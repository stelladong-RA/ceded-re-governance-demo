# app.py
from __future__ import annotations

import os
from pathlib import Path
import streamlit as st

from utils.loaders import load_settings


# -------------------------------------------------
# App Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Synpulse Ceded Re Demo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

settings = load_settings()

# Demo root precedence:
# 1) settings.yaml (demo_root)
# 2) env var DEMO_ROOT
# 3) repo folder containing app.py
default_root = str(Path(__file__).resolve().parent)
demo_root = settings.get("demo_root") or os.getenv("DEMO_ROOT") or default_root

mode = settings.get("mode", "demo")
version = settings.get("version", "synpulse-demo-v1")


# -------------------------------------------------
# Session defaults (safe, stable UX)
# -------------------------------------------------
if "meeting_mode" not in st.session_state:
    st.session_state["meeting_mode"] = True

if "show_governance_default" not in st.session_state:
    st.session_state["show_governance_default"] = True


# -------------------------------------------------
# Small UI helper
# -------------------------------------------------
def _pill(label: str, value: str):
    st.markdown(
        f"""
        <div style="
            display:inline-block;
            padding:6px 10px;
            margin:4px 6px 4px 0;
            border-radius:999px;
            border:1px solid rgba(49,51,63,0.18);
            background:rgba(49,51,63,0.04);
            font-size:13px;">
            <b>{label}:</b> {value}
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------------------------------------
# Sidebar (client-facing + consistent with demos)
# -------------------------------------------------
st.sidebar.title("Synpulse × Reinsurance Analytics")
st.sidebar.caption("Ceded Re Governance Platform — Demo Environment")
st.sidebar.markdown("---")

st.sidebar.subheader("Today’s demo flow")
st.sidebar.markdown(
    """
1) **CEC — Exposure & Credit Control**  
2) **Cash Pairing — Reconciliation**  
3) **Consultant Copilot — Narrative + Q&A**  
"""
)

st.sidebar.markdown("---")
st.sidebar.subheader("Controls")

# Keep root visible for portability and for the team
demo_root_input = st.sidebar.text_input(
    "Demo root folder",
    value=demo_root,
    help="Path to the demo repository root (used by all pages via DataPaths).",
)

# Meeting mode keeps the landing page clean and prevents “internal notes” showing by default
st.sidebar.toggle("Meeting mode (hide internal notes)", key="meeting_mode")
st.sidebar.toggle("Show governance expanders by default", key="show_governance_default")

st.sidebar.markdown("---")
st.sidebar.subheader("Environment")

st.sidebar.info(
    f"""
**Mode:** {mode}  
**Version:** {version}  
**Data:** synthetic / illustrative  
"""
)

st.sidebar.caption(
    "Tip: Start with **CEC** to establish governance backbone, then show ops value in **Cash Pairing**, then packaging in **Copilot**."
)


# -------------------------------------------------
# Landing Page (matches Demo 1/2/3 structure)
# -------------------------------------------------
st.title("Ceded Reinsurance Analytics Platform")
st.caption("Strategic demonstration environment — synthetic data (illustrative only).")

with st.expander("Executive Brief — What this is, why it matters, and how to use it", expanded=True):
    st.markdown(
        """
### What this platform is
A **single, governed analytics layer** for Ceded Re operations that consolidates exposure, credit, and cash reconciliation
into **board-ready views** and **ops-ready worklists**.

### Why it matters
In most organizations, ceded re information is fragmented across PAS, placements/brokers, finance, and spreadsheets.
That creates slow close cycles, inconsistent answers, and audit friction.
This demo shows how the same outcomes can be delivered **faster, repeatably, and defensibly**.

### How to use this app 
1) **Demo 1 — CEC:** establish the portfolio “single source of truth” + governance  
2) **Demo 2 — Cash Pairing:** show operational controls + exception workflow  
3) **Demo 3 — Copilot:** show packaging into executive narrative + evidence pack

### What makes it different
- **Governance-first:** validation + lineage + assumptions are visible  
- **Above-systems posture:** integrates without rip-and-replace  
- **Operational outputs:** not just dashboards—worklists and controls  
- **Consulting-ready:** reusable accelerator structure for client programs  
"""
    )

st.markdown("---")


# -------------------------------------------------
# Executive Snapshot (tight, professional)
# -------------------------------------------------
st.subheader("Executive snapshot")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Modules", "3")
    st.caption("CEC • Cash Pairing • Copilot")

with c2:
    st.metric("Primary outcomes", "Governance + Efficiency")
    st.caption("Exposure control, credit risk, reconciliation speed")

with c3:
    st.metric("Integration posture", "Above systems")
    st.caption("PAS / placements / finance — no rip-and-replace")

with c4:
    st.metric("Delivery model", "Pilot-ready")
    st.caption("Repeatable accelerators + evidence packs")

st.markdown("---")


# -------------------------------------------------
# Where to start (clear CTA)
# -------------------------------------------------
st.subheader("Where to start")
st.markdown(
    """
Follow this sequence:

1. **CEC — Exposure & Credit Control**  
   Establish portfolio totals, counterparty concentration, credit utilization, and stress sensitivities.

2. **Cash Pairing — Reconciliation**  
   Demonstrate confidence-scored matching and an ops-ready exception queue.

3. **Consultant Copilot — Narrative + Q&A**  
   Show how governed KPIs become executive narratives with an evidence pack.
"""
)

st.markdown("---")


# -------------------------------------------------
# Governance posture (confidence builder)
# -------------------------------------------------
with st.expander(
    "Governance & audit posture (demo)",
    expanded=st.session_state.get("show_governance_default", True),
):
    st.markdown(
        """
This demo emphasizes governance because clients care about defensibility:

- **Traceability:** outputs are backed by the underlying integrated “portfolio layer”  
- **Assumptions disclosure:** derived fields and defaults are explicit  
- **Data health:** schema checks + join coverage support trust in results  
- **Repeatability:** reruns produce consistent outputs, enabling control evidence
"""
    )
    _pill("Demo root", demo_root_input)
    _pill("Mode", str(mode))
    _pill("Version", str(version))

st.markdown("---")


# -------------------------------------------------
# Footer
# -------------------------------------------------
st.caption("© Reinsurance Analytics | Demonstration Platform")
