"""
pages/2_Cash_Pairing.py — Cash Pairing (Statements of Account Reconciliation)

Executive + operations dashboard for:
- Pairing cash transactions to statements of account (SoA)
- Measuring match coverage, confidence, and variance
- Producing an ops-ready exception queue
- Showing governance transparency for audit / controllership

Designed for finance ops, ceded re ops, and client governance conversations.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from components.governance import show_governance_panel
from services.ingestion import DataPaths, load_canonical
from services.cash_engine import CashPairingConfig, pair_cash_to_statements
from services.reconciliation import cash_recon_summary, build_cash_exception_queue


# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(page_title="Cash Pairing — Reconciliation", layout="wide")

st.title("Cash Pairing & Reconciliation")
st.caption("Operational governance view of ceded reinsurance cash clearing (synthetic demo data)")


# -------------------------------------------------
# Executive Brief (Top-of-page explanation)
# -------------------------------------------------
with st.expander("Executive Brief — What this is, why it matters, and how to use it", expanded=True):
    st.markdown(
        """
### What this demo is
A **single logical reconciliation layer** that links **cash transactions** to **Statements of Account (SoA)** so teams can
**close faster**, reduce manual effort, and maintain **audit-ready traceability**.

### Why we need this
In many ceded re organizations, cash clearing becomes a **spreadsheet + email process** because:
- cash comes from banks / treasury,
- SoA comes from brokers / reinsurers,
- references are inconsistent or missing,
- partial payments and timing create ambiguity.

The result: slow close cycles, unreconciled balances, and exceptions that are hard to prioritize.

### What questions it answers (plain English)
- “**Which payments are already cleared vs not cleared?**”
- “**What is the reason an item is not cleared?**”
- “**Which items should ops work first?** (highest risk / highest materiality)”
- “**If we change tolerance / window, what happens?**”
- “**Can I show an auditor the reason a match was made?**”

### How to use this page (30 seconds)
1) Adjust **amount tolerance** and **invoice date window**  
2) Review **coverage KPIs** and match quality signals  
3) Use the **exception queue** as the operational worklist  
4) Drill into drivers (counterparty, variance, confidence)  
5) Export evidence packs (results + queue)

### What makes this approach different
- **Confidence-scored matching** (not just matched/unmatched)
- **Transparent reason codes** (why a match was made is visible)
- **Ops-ready exception queue** (prioritized by confidence, variance, and materiality)
- **Governance-first** (repeatable rules, validation, consistent schema)
        """
    )


# -------------------------------------------------
# Governance Panel
# -------------------------------------------------
show_governance_panel()


# -------------------------------------------------
# Sidebar Controls
# -------------------------------------------------
st.sidebar.header("Configuration")

root_default = os.getenv("DEMO_ROOT", str(Path(__file__).resolve().parents[1]))
root = st.sidebar.text_input("Demo root folder", value=root_default)

tol = st.sidebar.number_input(
    "Amount tolerance (USD)",
    min_value=0.0, max_value=5000.0, value=250.0, step=50.0
)
window = st.sidebar.slider("Invoice date window (days)", 7, 120, 45, 1)
min_conf = st.sidebar.slider("Min confidence for 'OK' (exceptions below this)", 0.0, 1.0, 0.25, 0.05)

st.sidebar.markdown("---")
show_only_exceptions = st.sidebar.checkbox("Show exceptions only", value=False)

match_type_filter = st.sidebar.multiselect(
    "Filter match types",
    options=["exact_reference", "candidate", "unmatched"],
    default=["exact_reference", "candidate", "unmatched"]
)

sort_mode = st.sidebar.selectbox(
    "Sort results by",
    options=["priority desc", "confidence desc", "variance desc", "amount desc", "date desc"],
    index=0
)


# -------------------------------------------------
# Load Data
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(root_path: str):
    return load_canonical(DataPaths(Path(root_path)))

data = load_data(root)

required = {"cash", "statements"}
missing = [k for k in required if k not in data]
if missing:
    st.error(f"Missing required tables: {missing}. Check ingestion pipeline / raw CSVs.")
    st.stop()


# -------------------------------------------------
# Pairing Logic
# -------------------------------------------------
cfg = CashPairingConfig(amount_tolerance_usd=float(tol), date_window_days=int(window))

with st.spinner("Pairing cash transactions to statements..."):
    pairs = pair_cash_to_statements(data["cash"], data["statements"], cfg=cfg)

# Validate expected output shape (demo-friendly)
for col in ["match_type", "confidence"]:
    if col not in pairs.columns:
        st.error(f"Pairing output missing required column: {col}")
        st.stop()

# Standardize / derive common analytic fields
if "txn_date" not in pairs.columns and "payment_date" in pairs.columns:
    pairs["txn_date"] = pairs["payment_date"]

# Variance (prefer engine-provided amount_diff_usd if exists)
if "variance_usd" not in pairs.columns:
    if "amount_diff_usd" in pairs.columns:
        pairs["variance_usd"] = pd.to_numeric(pairs["amount_diff_usd"], errors="coerce")
    else:
        pairs["variance_usd"] = np.nan

# Materiality bucket for ops prioritization
amt_col = "amount_usd" if "amount_usd" in pairs.columns else None
if "materiality_bucket" not in pairs.columns:
    if amt_col:
        a = pd.to_numeric(pairs[amt_col], errors="coerce").abs()
        pairs["materiality_bucket"] = pd.cut(
            a,
            bins=[-1, 5_000, 25_000, 100_000, 1_000_000, float("inf")],
            labels=["<5k", "5–25k", "25–100k", "100k–1m", ">1m"]
        )
    else:
        pairs["materiality_bucket"] = "unknown"

pairs["within_tolerance"] = pd.to_numeric(pairs["variance_usd"], errors="coerce").fillna(0) <= float(tol)

# Priority score (simple + explainable)
# Higher priority when: unmatched OR low confidence OR high variance OR higher amount
conf = pd.to_numeric(pairs["confidence"], errors="coerce").fillna(0.0)
var = pd.to_numeric(pairs["variance_usd"], errors="coerce").fillna(0.0)
amt = pd.to_numeric(pairs[amt_col], errors="coerce").abs().fillna(0.0) if amt_col else pd.Series(0.0, index=pairs.index)

amt_norm = (amt / max(float(amt.max()), 1.0)).clip(0, 1)
pairs["priority_score"] = (
    (pairs["match_type"].eq("unmatched")).astype(int) * 3.0
    + (conf < float(min_conf)).astype(int) * 2.0
    + (var > float(tol)).astype(int) * 2.0
    + amt_norm * 1.0
).round(2)


# -------------------------------------------------
# Summary KPIs
# -------------------------------------------------
summary = cash_recon_summary(pairs, amount_tolerance_usd=float(tol))

st.header("1. Reconciliation Coverage Overview")
st.markdown(
    """
This section summarizes how much cash can be **automatically cleared** and what remains as **exceptions**.
In a real engagement, this becomes a repeatable control for finance ops and ceded re operations.
"""
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Matched", f"{summary.matched:,}")
k2.metric("Unmatched", f"{summary.unmatched:,}")
k3.metric("Variance > tolerance", f"{summary.variance_count:,}")
k4.metric("Avg confidence", f"{summary.avg_confidence:.2f}")

st.info(
    "Key Insight: The goal is **high-confidence automation** plus a **controlled exception workflow** — not blind auto-matching."
)

st.divider()


# -------------------------------------------------
# Section 2 — Match Drivers & Quality Signals
# -------------------------------------------------
st.header("2. Match Drivers & Quality Signals")
st.markdown(
    """
These diagnostics explain *why* cash is or isn’t clearing.  
They also show how controls (tolerance + date window + confidence thresholds) impact automation vs conservatism.
"""
)

breakdown = pairs["match_type"].value_counts(dropna=False).reset_index()
breakdown.columns = ["match_type", "count"]

conf_bins = pd.cut(
    conf,
    bins=[-0.001, 0.25, 0.5, 0.75, 0.9, 1.0],
    labels=["0–0.25", "0.25–0.5", "0.5–0.75", "0.75–0.9", "0.9–1.0"],
)
conf_tbl = conf_bins.value_counts().sort_index().reset_index()
conf_tbl.columns = ["confidence_band", "count"]

c1, c2 = st.columns([0.5, 0.5])

with c1:
    st.subheader("Match type breakdown")
    st.bar_chart(breakdown.set_index("match_type")["count"], use_container_width=True)
    with st.expander("Details"):
        st.dataframe(breakdown, use_container_width=True, hide_index=True)

with c2:
    st.subheader("Confidence distribution")
    st.bar_chart(conf_tbl.set_index("confidence_band")["count"], use_container_width=True)
    with st.expander("Details"):
        st.dataframe(conf_tbl, use_container_width=True, hide_index=True)

st.subheader("Variance vs Confidence (risk quadrant)")
st.caption("Low confidence + high variance = highest operational risk items that should be worked first.")

quad = pairs.copy()
quad["confidence"] = conf
quad["variance_usd"] = var
quad_plot = quad[["confidence", "variance_usd", "match_type"]].dropna()

if len(quad_plot) > 0:
    st.scatter_chart(quad_plot, x="confidence", y="variance_usd", color="match_type", use_container_width=True)

st.warning(
    "Governance Note: These thresholds are controllable levers — tighten for audit conservatism or loosen to increase automation."
)

st.divider()


# -------------------------------------------------
# Section 3 — Exception Queue (ops-ready workflow)
# -------------------------------------------------
st.header("3. Exception Queue (Operational Worklist)")
st.markdown(
    """
This is the **ops-ready queue**: items that are **unmatched**, **outside tolerance**, or **below confidence threshold**.
In production, this becomes a workflow with ownership, SLA, and resolution notes.
"""
)

queue = build_cash_exception_queue(pairs, min_confidence=float(min_conf))

# Add priority score and materiality if not present in queue
if "priority_score" not in queue.columns and "cash_txn_id" in queue.columns and "cash_txn_id" in pairs.columns:
    enrich = pairs[["cash_txn_id", "priority_score", "materiality_bucket"]].drop_duplicates("cash_txn_id")
    queue = queue.merge(enrich, on="cash_txn_id", how="left")

# Filtered view for drill-down
view = pairs.copy()
view = view[view["match_type"].isin(match_type_filter)]

if show_only_exceptions:
    view = view[
        (view["match_type"].eq("unmatched"))
        | (conf < float(min_conf))
        | (var > float(tol))
    ]

# Sorting logic
if sort_mode == "priority desc":
    view = view.sort_values(["priority_score", "variance_usd"], ascending=[False, False])
elif sort_mode == "confidence desc":
    view = view.sort_values(["confidence"], ascending=False)
elif sort_mode == "variance desc":
    view = view.sort_values(["variance_usd"], ascending=False)
elif sort_mode == "amount desc" and amt_col:
    view = view.sort_values([amt_col], ascending=False)
elif sort_mode == "date desc" and "txn_date" in view.columns:
    view = view.sort_values(["txn_date"], ascending=False)

# Queue KPIs
qc1, qc2, qc3 = st.columns(3)
qc1.metric("Exceptions", f"{len(queue):,}")
qc2.metric("Outside tolerance", f"{int((var > float(tol)).sum()):,}")
qc3.metric("Below min confidence", f"{int((conf < float(min_conf)).sum()):,}")

st.subheader("Exception queue")
st.dataframe(queue.sort_values(["priority_score"], ascending=False) if "priority_score" in queue.columns else queue,
             use_container_width=True, hide_index=True)

# Counterparty driver summary (what execs ask)
if "counterparty_id" in pairs.columns:
    st.subheader("Top exception drivers (by counterparty)")
    exc_mask = (pairs["match_type"].eq("unmatched")) | (conf < float(min_conf)) | (var > float(tol))
    exc = pairs[exc_mask].copy()
    if len(exc) > 0:
        drivers = (
            exc.groupby("counterparty_id", dropna=False)
            .agg(
                exceptions=("cash_txn_id", "count") if "cash_txn_id" in exc.columns else ("match_type", "count"),
                avg_confidence=("confidence", "mean"),
                total_amount_usd=(amt_col, "sum") if amt_col else ("confidence", "count"),
            )
            .reset_index()
            .sort_values(["exceptions"], ascending=False)
            .head(10)
        )
        st.dataframe(drivers, use_container_width=True, hide_index=True)
    else:
        st.write("No exceptions under current settings.")

st.subheader("Drill-down (filtered pairing results)")
st.dataframe(view, use_container_width=True, hide_index=True)

# Exports
st.markdown("### Exports")
col_a, col_b = st.columns(2)
with col_a:
    st.download_button(
        "Download pairing results (CSV)",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="cash_pairing_results.csv",
        mime="text/csv",
        use_container_width=True,
    )
with col_b:
    st.download_button(
        "Download exception queue (CSV)",
        data=queue.to_csv(index=False).encode("utf-8"),
        file_name="cash_exception_queue.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.divider()


# -------------------------------------------------
# Section 4 — What this enables (executive framing)
# -------------------------------------------------
st.header("4. What This Enables for Synpulse + Clients")

st.markdown(
    """
**For finance / controllership**
- Faster close cycles and fewer manual reconciliations  
- Repeatable control evidence for auditors  
- Clear prioritization of exceptions by materiality and confidence  

**For ceded re operations**
- Reduced leakage and fewer “unknown balance” disputes  
- Better counterparty management through exception transparency  

**For consulting delivery**
- A reusable accelerator: plug in client data, validate schema, produce worklists + governance outputs
"""
)

# Recommended next steps (dynamic narrative)
rec_actions = []
if summary.unmatched > 0:
    rec_actions.append("Review unmatched items: missing references, missing SoA, or counterparty ID mismatches.")
if summary.variance_count > 0:
    rec_actions.append("Investigate variance drivers: partial payments, FX, fees, or timing differences.")
if summary.avg_confidence < 0.5:
    rec_actions.append("Tune tolerance/window or strengthen reference logic to improve confidence without increasing risk.")

if rec_actions:
    st.subheader("Recommended next steps (based on this run)")
    for a in rec_actions:
        st.write(f"- {a}")

st.success(
    "One-sentence takeaway: This demo turns cash reconciliation from a spreadsheet process into a governed, confidence-scored operational workflow."
)

st.caption("Reinsurance Analytics × Synpulse — Ceded Re Governance Platform")
