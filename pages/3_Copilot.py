"""
pages/3_Copilot.py — Consultant Copilot (Narratives + Q&A)

Executive + consulting dashboard for:
- Generating board/client-ready narratives from governed KPIs (CEC + Cash)
- Demonstrating controlled Q&A with explicit “evidence pack” context
- Running safely in deterministic fallback mode when no LLM key exists

Designed for consulting delivery: faster synthesis, consistent messaging,
and audit-ready traceability back to underlying metrics.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from components.governance import show_governance_panel

from services.ingestion import DataPaths, load_canonical
from services.exposure_engine import ExposureConfig, build_portfolio_exposure
from services.credit_engine import CreditConfig, build_counterparty_limits, compute_utilization
from services.cash_engine import CashPairingConfig, pair_cash_to_statements
from services.reconciliation import cash_recon_summary, portfolio_data_quality
from services.llm_service import build_copilot_context, draft_exec_summary, LLMConfig


# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(page_title="Copilot — Consultant Narrative Layer", layout="wide")

st.title("Consultant Copilot — Narratives + Q&A")
st.caption("Controlled narrative layer over governed Ceded Re KPIs (synthetic demo data)")


# -------------------------------------------------
# Executive Brief (Top-of-page explanation)
# -------------------------------------------------
with st.expander("Executive Brief — What this is, why it matters, and how to use it", expanded=True):
    st.markdown(
        """
### What this demo is
A **consultant copilot** that converts governed analytics (CEC + Cash Pairing) into an **executive-ready narrative** with:
- **Key metrics** (what happened / what’s material)
- **Governance risks** (what could go wrong)
- **Recommended actions** (what to do next)
- A visible **Evidence Pack** (exact inputs used)

### Why we need this
Even when analytics exist, teams still spend time translating metrics into:
- steering committee updates,
- board risk narratives,
- audit/control explanations,
- pilot proposals and “what data do you need?”

This page demonstrates a repeatable narrative layer that is **fast, consistent, and defensible**.

### Guardrails (why this is credible)
- Uses **pre-computed KPIs** and selected evidence only (no guessing).
- Output comes with an **Evidence Pack** showing what was used.
- Runs safely in **deterministic mode** if no LLM key exists.

### How to use this page 
1) Select a prompt (Section 1)  
2) Generate a narrative (Section 2)  
3) Review the Evidence Pack (Section 3)  
4) Export narrative + evidence for a deck / memo (Section 4)

### One-sentence takeaway
**Copilot turns governed Ceded Re analytics into board-ready consulting outputs in minutes, with a visible evidence trail.**
        """
    )


# -------------------------------------------------
# Governance Panel (same position as Demo 1 & 2)
# -------------------------------------------------
show_governance_panel()


# -------------------------------------------------
# Sidebar Controls (same concept as Demo 1 & 2)
# -------------------------------------------------
st.sidebar.header("Configuration")

root_default = os.getenv("DEMO_ROOT", str(Path(__file__).resolve().parents[1]))
root = st.sidebar.text_input("Demo root folder", value=root_default)

st.sidebar.subheader("Copilot mode (controlled)")
provider = os.getenv("LLM_PROVIDER", "none")
model = os.getenv("LLM_MODEL", "gpt-4o-mini")
cfg = LLMConfig(provider=provider, model=model)

mode_label = (
    "Deterministic fallback (no external LLM)"
    if cfg.provider in ("none", "", None)
    else f"LLM enabled ({cfg.provider} / {cfg.model})"
)
if cfg.provider in ("none", "", None):
    st.sidebar.info(f"Mode: {mode_label}")
else:
    st.sidebar.success(f"Mode: {mode_label}")

st.sidebar.subheader("Scope of evidence pack")
include_credit_table = st.sidebar.checkbox("Include credit utilization evidence (top 10)", value=True)
include_cash = st.sidebar.checkbox("Include cash pairing KPIs", value=True)
include_data_quality = st.sidebar.checkbox("Include data-quality KPIs", value=True)

st.sidebar.subheader("Output style")
tone = st.sidebar.selectbox("Tone", ["Board-ready", "Client steering committee", "Ops-focused"], index=0)
length = st.sidebar.selectbox("Length", ["Concise", "Standard", "Detailed"], index=1)

st.sidebar.subheader("Presentation helper")
show_talk_track = st.sidebar.checkbox("Show meeting talk track", value=True)

st.divider()


# -------------------------------------------------
# Load Data (governed KPI context)
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(root_path: str):
    return load_canonical(DataPaths(Path(root_path)))


with st.spinner("Loading and preparing governed KPI context..."):
    data = load_data(root)

required_tables = {"pas", "placements", "counterparties", "cash", "statements"}
missing = [t for t in required_tables if t not in data]
if missing:
    st.error(f"Missing required tables: {missing}. Check your raw CSVs and pipeline.")
    st.stop()

portfolio = build_portfolio_exposure(
    pas=data["pas"],
    placements=data["placements"],
    counterparties=data["counterparties"],
    cfg=ExposureConfig(),
)

dq_full = portfolio_data_quality(portfolio)

limits = build_counterparty_limits(data["counterparties"], CreditConfig())
util = compute_utilization(portfolio, limits)

pairs = None
cash_kpis_full = {}
if include_cash:
    pairs = pair_cash_to_statements(data["cash"], data["statements"], CashPairingConfig())
    cash_kpis_full = cash_recon_summary(pairs).__dict__

dq_used_for_display = dq_full if include_data_quality else {}
cash_kpis_used_for_display = cash_kpis_full if include_cash else {}

dq_used_for_context = dq_full if include_data_quality else {"rows_total": dq_full.get("rows_total", 0)}
credit_table_for_context = util if include_credit_table else util.head(0)
cash_kpis_for_context = cash_kpis_full if include_cash else {}

evidence_pack = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "mode": mode_label,
    "tone": tone,
    "length": length,
    "included": {
        "data_quality_kpis": bool(include_data_quality),
        "credit_utilization_top10": bool(include_credit_table),
        "cash_pairing_kpis": bool(include_cash),
    },
    "exposure_data_quality": dq_used_for_display,
    "credit_utilization_top10": (
        util.head(10).to_dict(orient="records") if include_credit_table else []
    ),
    "cash_recon_kpis": cash_kpis_used_for_display,
}

ctx = build_copilot_context(
    exposure_kpis=dq_used_for_context,
    credit_table=credit_table_for_context,
    cash_recon_kpis=cash_kpis_for_context,
)


# -------------------------------------------------
# SECTION 1 — Suggested Prompts
# -------------------------------------------------
st.header("1. Suggested Prompts")

st.markdown(
    """
Use these prompts to drive a **consistent** meeting flow:
- Steering committee summary (balanced)
- Board-ready risk view (decision-oriented)
- Ops + controls view (workflow + exceptions)
- Pilot proposal (next steps)
- Data readiness (what we need from the client)
"""
)

DEFAULT_PROMPTS = {
    "Steering committee summary": (
        "Summarize key governance risks and recommended next steps for a Ceded Re exposure control assessment."
    ),
    "Board-ready risk view": (
        "Create a board-ready summary of counterparty concentration, credit utilization, and stress sensitivities. "
        "Include 3 risks and 3 actions."
    ),
    "Ops + controls view": (
        "Summarize operational reconciliation risks in cash pairing, highlight exception drivers, "
        "and propose an operating cadence for resolution."
    ),
    "Pilot proposal": (
        "Draft a 30-day pilot plan with success metrics and minimal data requirements for "
        "Ceded Exposure Control and Cash Reconciliation."
    ),
    "Data readiness": (
        "Assess data readiness based on join rates and completeness. List gaps, risks, and how to remediate in a pilot."
    ),
}

if "copilot_question" not in st.session_state:
    st.session_state["copilot_question"] = DEFAULT_PROMPTS["Steering committee summary"]

colp1, colp2, colp3 = st.columns(3)
with colp1:
    if st.button("Steering committee summary", use_container_width=True):
        st.session_state["copilot_question"] = DEFAULT_PROMPTS["Steering committee summary"]
with colp2:
    if st.button("Board-ready risk view", use_container_width=True):
        st.session_state["copilot_question"] = DEFAULT_PROMPTS["Board-ready risk view"]
with colp3:
    if st.button("Ops + controls view", use_container_width=True):
        st.session_state["copilot_question"] = DEFAULT_PROMPTS["Ops + controls view"]

colp4, colp5, colp6 = st.columns(3)
with colp4:
    if st.button("Pilot proposal", use_container_width=True):
        st.session_state["copilot_question"] = DEFAULT_PROMPTS["Pilot proposal"]
with colp5:
    if st.button("Data readiness", use_container_width=True):
        st.session_state["copilot_question"] = DEFAULT_PROMPTS["Data readiness"]
with colp6:
    pass

question = st.text_area(
    "Prompt / question",
    value=st.session_state["copilot_question"],
    height=90,
)

st.divider()


# -------------------------------------------------
# SECTION 2 — Generate Narrative
# -------------------------------------------------
st.header("2. Generate Narrative")

st.markdown(
    """
This generates an **executive-ready narrative** with:
- key metrics,
- top risks,
- recommended actions,
- and clarification questions.

In the meeting: click **Generate**, then scroll to **Evidence Pack** to show defensibility.
"""
)

gen_col1, gen_col2 = st.columns([0.65, 0.35])
with gen_col1:
    if show_talk_track:
        st.info(
            "Talk track: “This narrative is generated only from governed KPIs — then we show the exact evidence used.”"
        )
with gen_col2:
    generate = st.button("Generate narrative", type="primary", use_container_width=True)

if generate:
    with st.spinner("Drafting executive narrative..."):
        style_hints = (
            f"\n\nStyle requirements:\n"
            f"- Tone: {tone}\n"
            f"- Length: {length}\n"
            f"- Must include: (i) key metrics, (ii) top governance risks, (iii) recommended next steps, (iv) 3 questions to confirm.\n"
            f"- Avoid jargon; be precise and decision-oriented.\n"
        )
        result = draft_exec_summary(question=question + style_hints, context=ctx, cfg=cfg)
    st.session_state["copilot_result"] = result

result = st.session_state.get("copilot_result", {"summary": ""})

st.text_area("Executive-ready narrative", value=result.get("summary", ""), height=280)

st.divider()


# -------------------------------------------------
# SECTION 3 — Evidence Pack (Defensibility)
# -------------------------------------------------
st.header("3. Evidence Pack (What the Copilot Used)")

st.markdown(
    """
This section makes the output **defensible**:
it shows the exact KPI context used to produce the narrative — useful for governance, audit, and consulting credibility.
"""
)

c1, c2 = st.columns([0.55, 0.45])

with c1:
    st.subheader("Exposure & data-quality KPIs")
    if include_data_quality:
        st.json(dq_used_for_display)
    else:
        st.info("Not included (toggle in sidebar).")

    st.subheader("Cash reconciliation KPIs")
    if include_cash:
        st.json(cash_kpis_used_for_display)
    else:
        st.info("Not included (toggle in sidebar).")

with c2:
    st.subheader("Credit utilization (top 10)")
    if include_credit_table:
        st.dataframe(util.head(10), use_container_width=True, hide_index=True)
    else:
        st.info("Not included (toggle in sidebar).")

st.divider()


# -------------------------------------------------
# SECTION 4 — Optional Drill-Down (Demo Realism)
# -------------------------------------------------
st.header("4. Optional Drill-Down (Underlying Tables)")

with st.expander("Show underlying tables (synthetic demo realism)", expanded=False):
    st.write("In production, these tables map to client sources with schema validation and governance controls.")
    tab1, tab2, tab3 = st.tabs(["Portfolio (sample)", "Cash pairs (sample)", "Utilization (full)"])

    with tab1:
        st.dataframe(portfolio.head(50), use_container_width=True, hide_index=True)

    with tab2:
        if pairs is not None:
            st.dataframe(pairs.head(50), use_container_width=True, hide_index=True)
        else:
            st.info("Cash pairing not included.")

    with tab3:
        st.dataframe(util, use_container_width=True, hide_index=True)

st.divider()


# -------------------------------------------------
# SECTION 5 — Exports (Deck / Memo)
# -------------------------------------------------
st.header("5. Exports (Deck / Memo)")

st.markdown("Export narrative and evidence pack to paste into a deck, memo, or steering committee update.")

narrative_text = result.get("summary", "") or ""
export_bundle = {
    "question": question,
    "narrative": narrative_text,
    "evidence_pack": evidence_pack,
}

colx1, colx2 = st.columns(2)
with colx1:
    st.download_button(
        "Download narrative (TXT)",
        data=narrative_text.encode("utf-8"),
        file_name="copilot_narrative.txt",
        mime="text/plain",
        use_container_width=True,
    )

with colx2:
    st.download_button(
        "Download narrative + evidence pack (JSON)",
        data=json.dumps(export_bundle, indent=2).encode("utf-8"),
        file_name="copilot_narrative_evidence.json",
        mime="application/json",
        use_container_width=True,
    )

st.divider()


# -------------------------------------------------
# SECTION 6 — How this becomes production
# -------------------------------------------------
st.header("6. How This Becomes a Real Copilot in a Client Program")

st.markdown(
    """
In a client deployment, the copilot typically sits on top of:

- **Canonical Ceded Re data model** (placements, exposure, claims, finance, cash)
- **Governed analytics** (lineage, validations, assumptions, audit evidence)
- **Secure LLM gateway** with strict access controls and redaction policies
- **Role-based outputs**: board summary, steering committee pack, ops worklists
- **Feedback loop**: consultant edits captured to improve templates and policies

This demo is intentionally conservative: designed to be safe, repeatable, and auditable.
"""
)

st.success("One-sentence takeaway: Copilot converts governed metrics into consulting outputs — fast, consistent, and defensible.")
st.caption("Reinsurance Analytics — Ceded Re Governance Platform")
