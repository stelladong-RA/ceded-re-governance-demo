"""
pages/1_CEC.py — Ceded Exposure & Credit Control (CEC)

Executive + governance dashboard for:
- Portfolio exposure oversight (single logical layer above source systems)
- Counterparty concentration risk
- Credit utilization vs limits/ratings
- Stress testing sensitivities
- Audit & governance transparency (validation, lineage, assumptions)

Designed for client, board, and regulatory conversations.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from components.governance import show_governance_panel
from components.charts import (
    plot_concentration,
    plot_exposure_by_region,
    plot_utilization,
    plot_stress_impact,
)

from services.ingestion import load_canonical, DataPaths
from services.exposure_engine import (
    ExposureConfig,
    build_portfolio_exposure,
    top_concentrations,
    exposure_heatmap_table,
)
from services.credit_engine import (
    CreditConfig,
    build_counterparty_limits,
    compute_utilization,
    stress_scenario,
)

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(page_title="CEC — Exposure & Credit Control", layout="wide")

st.title("Ceded Exposure & Credit Control (CEC)")
st.caption("Strategic governance view of ceded reinsurance risk (synthetic demo data)")

# -------------------------------------------------
# Sidebar Controls (match Demo 2/3 pattern)
# -------------------------------------------------
st.sidebar.header("Configuration")

root_default = os.getenv("DEMO_ROOT", str(Path(__file__).resolve().parents[1]))
root = st.sidebar.text_input("Demo root folder", value=root_default)

st.sidebar.subheader("Exposure settings")
basis = st.sidebar.selectbox(
    "Exposure basis",
    options=["limit", "sum_insured", "written_premium"],
    index=0,
)

st.sidebar.subheader("Stress settings")
shock = st.sidebar.slider("Stress shock (%)", min_value=0, max_value=50, value=10, step=5)

st.sidebar.subheader("Diagnostics")
show_raw = st.sidebar.checkbox("Show raw tables (debug)", value=False)
show_portfolio_sample = st.sidebar.checkbox("Show portfolio sample rows", value=True)

# -------------------------------------------------
# Load Data
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(root_path: str) -> dict[str, pd.DataFrame]:
    return load_canonical(DataPaths(Path(root_path)))

with st.spinner("Loading canonical data..."):
    data = load_data(root)

# Demo-friendly guards
required = {"pas", "placements", "counterparties"}
missing = [k for k in required if k not in data]
if missing:
    st.error(f"Missing required tables: {missing}. Check your ingestion pipeline / raw CSVs.")
    st.stop()

# -------------------------------------------------
# Build Portfolio Layer (governed “single logical layer”)
# -------------------------------------------------
cfg = ExposureConfig(ceded_basis=basis)

portfolio = build_portfolio_exposure(
    pas=data["pas"],
    placements=data["placements"],
    counterparties=data["counterparties"],
    cfg=cfg,
)

# -------------------------------------------------
# Helpers for robust display (stable UX even if schemas evolve)
# -------------------------------------------------
def _safe_col(df: pd.DataFrame, col: str) -> bool:
    return isinstance(df, pd.DataFrame) and (col in df.columns)

def _ensure_util_columns(util_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes utilization dataframe column names so charts/tables are stable.
    """
    if util_df is None or util_df.empty:
        return util_df

    df = util_df.copy()

    # exposures: ceded_exposure_usd vs total_ceded_exposure_usd
    if "total_ceded_exposure_usd" not in df.columns and "ceded_exposure_usd" in df.columns:
        df.rename(columns={"ceded_exposure_usd": "total_ceded_exposure_usd"}, inplace=True)

    # risk flag: risk_flag vs flag
    if "flag" not in df.columns and "risk_flag" in df.columns:
        df.rename(columns={"risk_flag": "flag"}, inplace=True)

    # utilization: ensure numeric
    if "utilization_pct" in df.columns:
        df["utilization_pct"] = pd.to_numeric(df["utilization_pct"], errors="coerce")

    return df

def _portfolio_kpis(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"rows": 0, "matched": 0, "unmatched": 0, "ceded_total": 0.0, "avg_quality": None}

    rows = len(df)
    matched = int((df["join_confidence"] == "matched").sum()) if _safe_col(df, "join_confidence") else None
    unmatched = int((df["join_confidence"] != "matched").sum()) if _safe_col(df, "join_confidence") else None
    ceded_total = float(df["ceded_exposure_usd"].sum()) if _safe_col(df, "ceded_exposure_usd") else 0.0
    avg_quality = float(df["data_quality_score"].mean()) if _safe_col(df, "data_quality_score") else None
    return {"rows": rows, "matched": matched, "unmatched": unmatched, "ceded_total": ceded_total, "avg_quality": avg_quality}

def _data_sources_summary(tables: dict) -> pd.DataFrame:
    """
    Quick summary table for meeting realism: rows, cols, and key fields present.
    """
    key_cols = {
        "pas": ["policy_id", "program_id", "sum_insured_usd", "written_premium_usd", "region", "peril"],
        "placements": ["treaty_id", "policy_id", "program_id", "counterparty_id", "layer", "limit_usd", "ceded_limit_usd", "ceded_premium_usd"],
        "counterparties": ["counterparty_id", "counterparty_name", "rating", "base_limit_usd"],
        "claims": ["claim_id", "policy_id", "paid_usd"],
        "cash": ["cash_id", "counterparty_id", "txn_date", "amount_usd"],
        "statements": ["statement_id", "counterparty_id", "invoice_date", "amount_due_usd"],
    }

    rows = []
    for name, df in tables.items():
        if not isinstance(df, pd.DataFrame):
            continue
        present, missing = [], []
        for c in key_cols.get(name, []):
            (present if c in df.columns else missing).append(c)
        rows.append(
            {
                "Table": name,
                "Rows": int(len(df)),
                "Cols": int(df.shape[1]),
                "Key fields present": ", ".join(present) if present else "",
                "Key fields missing": ", ".join(missing) if missing else "",
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["Table"])
    return out

# -------------------------------------------------
# SECTION 0 — Executive Brief (match Demo 2/3 style)
# -------------------------------------------------
with st.expander("Executive Brief — What this is, why it matters, and how to use it", expanded=True):
    st.markdown(
        f"""
### What this demo is
A **governed, executive-facing view** of ceded exposure that answers:
- **Who are we ceded to?**
- **How concentrated are we?**
- **Are we within credit appetite/limits?**
- **What changes under stress?**

It creates a **single logical layer above source systems** (PAS, placements, finance, etc.) so responses are **faster and more defensible**.

### Why we need this 
Most organizations keep ceded exposure in **multiple systems and spreadsheets**. The same board question can produce different answers.
CEC provides a consistent, repeatable view with **validation and traceability**.

### What questions it answers
- “Which counterparties are we most exposed to?”
- “Do any reinsurers exceed limits or approach thresholds?”
- “What breaks first if exposure increases by {shock}%?”
- “Can we defend these numbers to audit / regulators?”

### How to use this page 
1) Set **Exposure basis** and **Stress shock** in the sidebar  
2) Review **Portfolio overview KPIs** (sanity + coverage)  
3) Show **Concentration** (top counterparties)  
4) Show **Credit utilization** (exposure vs limits)  
5) Show **Stress** (breaches under shock)  
6) Optionally show **Unified records** + debug tables for realism

### What makes this approach different (why it’s better than “just dashboards”)
- **Governance-first:** validation + lineage + assumptions are visible, not hidden.
- **System-agnostic:** sits above PAS/finance/placement tools; doesn’t require ripping/replacing.
- **Board-ready outputs:** concentration + utilization + stress are the exact governance questions executives ask.
- **Transparent defaults:** when data is incomplete, assumptions are explicit (not silent).
"""
    )

# Quick “what data did we load” snapshot (kept in same layout)
st.subheader("Data loaded for this run")
ds = _data_sources_summary(data)
st.dataframe(ds, use_container_width=True, hide_index=True)

# Portfolio KPIs (fast meeting anchors)
k = _portfolio_kpis(portfolio)
k0, k1, k2, k3, k4 = st.columns(5)
k0.metric("Portfolio rows", f"{k['rows']:,}")
k1.metric("Matched", f"{k['matched']:,}" if k["matched"] is not None else "—")
k2.metric("Unmatched", f"{k['unmatched']:,}" if k["unmatched"] is not None else "—")
k3.metric("Total ceded exposure", f"${k['ceded_total']:,.0f}")
k4.metric("Avg data quality", f"{k['avg_quality']:.1f}" if k["avg_quality"] is not None else "—")

st.info("Tip for the live demo: open **Governance & Audit Trail** next to show validation + refresh time, then walk the numbered sections.")

st.divider()

# -------------------------------------------------
# Governance Panel (align with Demo 2/3 pattern)
# -------------------------------------------------
show_governance_panel()

st.divider()

# -------------------------------------------------
# SECTION 1 — Portfolio Overview
# -------------------------------------------------
st.header("1. Portfolio Overview")
st.markdown(
    """
This is the consolidated exposure layer (policy × placement × counterparty).  
It replaces fragmented reporting across PAS, placement tools, and spreadsheets.
"""
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total records", f"{len(portfolio):,}")
c2.metric("Matched records", f"{(portfolio['join_confidence']=='matched').sum():,}" if _safe_col(portfolio, "join_confidence") else "—")
c3.metric("Unmatched records", f"{(portfolio['join_confidence']!='matched').sum():,}" if _safe_col(portfolio, "join_confidence") else "—")
c4.metric("Total ceded exposure", f"${portfolio['ceded_exposure_usd'].sum():,.0f}" if _safe_col(portfolio, "ceded_exposure_usd") else "—")

st.info("Key insight: once this layer is built, governance questions become fast, repeatable, and defensible—without manual reconciliation.")

st.divider()

# -------------------------------------------------
# SECTION 2 — Counterparty Concentration
# -------------------------------------------------
st.header("2. Counterparty Concentration Risk")
st.markdown(
    """
Highlights dependency on individual reinsurers and concentration risk.  
This is a common **board / risk committee** view.
"""
)

top = top_concentrations(portfolio, by="counterparty_name", n=10)
st.plotly_chart(plot_concentration(top), use_container_width=True)
st.dataframe(top, use_container_width=True, hide_index=True)

st.warning("Governance note: high reliance on a small set of reinsurers increases credit risk and renewal/negotiation risk.")

st.divider()

# -------------------------------------------------
# SECTION 3 — Exposure Distribution (Accumulation Lens)
# -------------------------------------------------
st.header("3. Exposure Distribution by Region")
st.markdown(
    """
Shows geographic distribution to support accumulation monitoring, capital planning, and governance checks.  
(You can swap Region for Peril/LOB in future iterations.)
"""
)

heat = exposure_heatmap_table(portfolio, index="counterparty_name", columns="region")
st.plotly_chart(plot_exposure_by_region(heat), use_container_width=True)
st.dataframe(heat, use_container_width=True)

st.info("Management use: supports accumulation reviews, regional underwriting strategy, and capital allocation discussions.")

st.divider()

# -------------------------------------------------
# SECTION 4 — Credit Utilization
# -------------------------------------------------
st.header("4. Counterparty Credit Utilization")
st.markdown(
    """
Compares exposure against credit limits (in production: derived from credit policy + ratings + appetite).  
This is where **risk flags** become actionable.
"""
)

limits = build_counterparty_limits(data["counterparties"], CreditConfig())
util = compute_utilization(portfolio, limits)
util = _ensure_util_columns(util)

st.plotly_chart(plot_utilization(util), use_container_width=True)
st.dataframe(util, use_container_width=True, hide_index=True)

st.error("Risk indicator: utilization above 80% typically triggers review (credit committee / governance escalation).")

st.divider()

# -------------------------------------------------
# SECTION 5 — Stress Scenario Analysis
# -------------------------------------------------
st.header("5. Stress Scenario Analysis")
st.markdown(
    f"""
Simulates a **{shock}% exposure uplift** to represent adverse movements (loss escalation, model drift, or portfolio growth).  
This highlights which counterparties breach thresholds first.
"""
)

stressed = stress_scenario(util, shock_pct=float(shock))

# Normalize columns if upstream engine differs
if stressed is not None and not stressed.empty:
    if "total_ceded_exposure_usd" not in stressed.columns and "ceded_exposure_usd" in stressed.columns:
        stressed.rename(columns={"ceded_exposure_usd": "total_ceded_exposure_usd"}, inplace=True)
    if "flag" not in stressed.columns and "risk_flag" in stressed.columns:
        stressed.rename(columns={"risk_flag": "flag"}, inplace=True)

st.plotly_chart(plot_stress_impact(stressed), use_container_width=True)

stress_cols = [
    "counterparty_name",
    "rating",
    "credit_limit_usd",
    "total_ceded_exposure_usd",
    "utilization_pct",
    "flag",
    "stressed_exposure_usd",
    "stressed_utilization_pct",
    "stressed_flag",
]
stress_cols = [c for c in stress_cols if _safe_col(stressed, c)]

st.dataframe(
    stressed[stress_cols] if stress_cols else stressed,
    use_container_width=True,
    hide_index=True,
)

st.warning("Board-level view: identifies reinsurers that could breach limits under moderate stress—supports proactive governance actions.")

st.divider()

# -------------------------------------------------
# SECTION 6 — Unified Portfolio Records (Audit Appendix)
# -------------------------------------------------
st.header("6. Unified Portfolio Records")
st.markdown(
    """
Sample integrated records used for auditability and drill-down.  
This is the “evidence layer” behind every chart and KPI above.
"""
)

sample_cols = [
    "policy_id",
    "program_id",
    "line_of_business",
    "peril",
    "region",
    "treaty_id",
    "layer",
    "counterparty_id",
    "counterparty_name",
    "rating",
    "basis_amount_usd",
    "ceded_share",
    "ceded_exposure_usd",
    "join_confidence",
    "cession_source",
    "data_quality_score",
]
sample_cols = [c for c in sample_cols if _safe_col(portfolio, c)]

if show_portfolio_sample:
    st.dataframe(
        portfolio[sample_cols].head(50) if sample_cols else portfolio.head(50),
        use_container_width=True,
        hide_index=True,
    )

st.success("Audit trail: every exposure figure is traceable to underlying rows (policy/placement/counterparty), with visible assumptions and validation status.")

# -------------------------------------------------
# Optional Debug Drill-down (matches Demo 2/3 pattern)
# -------------------------------------------------
if show_raw:
    with st.expander("Optional drill-down — raw & canonical tables (debug realism)"):
        st.write("These are synthetic demo tables. In production, these map to client sources with schema validation.")
        for k, df in data.items():
            st.subheader(k)
            st.dataframe(df.head(50), use_container_width=True)

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.caption("Reinsurance Analytics — Ceded Re Governance Platform")
