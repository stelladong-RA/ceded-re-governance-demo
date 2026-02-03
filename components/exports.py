# components/exports.py
from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from typing import Dict, Optional

import pandas as pd
import streamlit as st


def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def render_export_block(
    *,
    portfolio: pd.DataFrame,
    concentration: pd.DataFrame,
    utilization: pd.DataFrame,
    stressed: pd.DataFrame,
    governance_path: Optional[str] = None,
    meeting_notes: Optional[str] = None,
):
    """
    Export utilities for a consulting-ready pack.
    Uses Streamlit download buttons (no filesystem writes).
    """
    st.subheader("Exports — Consulting Briefing Pack")
    st.caption("One-click downloads for reuse in workshops, decks, and follow-ups.")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button(
            "Download: Concentration (CSV)",
            data=_df_to_csv_bytes(concentration),
            file_name=f"cec_concentration_{ts}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "Download: Utilization (CSV)",
            data=_df_to_csv_bytes(utilization),
            file_name=f"cec_utilization_{ts}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c3:
        st.download_button(
            "Download: Stress (CSV)",
            data=_df_to_csv_bytes(stressed),
            file_name=f"cec_stress_{ts}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c4:
        st.download_button(
            "Download: Portfolio sample (CSV)",
            data=_df_to_csv_bytes(portfolio.head(200)),
            file_name=f"cec_portfolio_sample_{ts}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("### Executive one-pager (copy/paste into slides or Confluence)")

    one_pager = build_one_pager_md(
        portfolio=portfolio,
        utilization=utilization,
        stressed=stressed,
        governance_path=governance_path,
        meeting_notes=meeting_notes,
    )

    st.download_button(
        "Download: One-pager (Markdown)",
        data=one_pager.encode("utf-8"),
        file_name=f"CEC_Briefing_OnePager_{ts}.md",
        mime="text/markdown",
        use_container_width=True,
    )

    with st.expander("Preview one-pager"):
        st.code(one_pager, language="markdown")


def build_one_pager_md(
    *,
    portfolio: pd.DataFrame,
    utilization: pd.DataFrame,
    stressed: pd.DataFrame,
    governance_path: Optional[str],
    meeting_notes: Optional[str],
) -> str:
    total_rows = len(portfolio)
    total_exposure = float(portfolio["ceded_exposure_usd"].sum()) if "ceded_exposure_usd" in portfolio.columns else 0.0
    matched = int((portfolio["join_confidence"] == "matched").sum()) if "join_confidence" in portfolio.columns else 0
    match_rate = (matched / total_rows) if total_rows else 0.0

    # Top risks from stressed scenario
    top_flags = None
    if "stressed_utilization_pct" in stressed.columns and "counterparty_name" in stressed.columns:
        top_flags = stressed.sort_values("stressed_utilization_pct", ascending=False).head(5)[
            ["counterparty_name", "stressed_utilization_pct"]
        ]

    lines = []
    lines.append("# Ceded Exposure & Credit Control (CEC) — Demo Briefing\n")
    lines.append("## What this demonstrates\n")
    lines.append(
        "- A **single logical layer** above PAS + placements + counterparty data\n"
        "- Governance-ready outputs: **concentration**, **credit utilization**, and **stress testing**\n"
        "- A defensible posture: **join confidence**, assumptions, and traceability\n"
    )

    lines.append("## Snapshot KPIs\n")
    lines.append(f"- Portfolio rows: **{total_rows:,}**\n")
    lines.append(f"- Match rate (placements joined): **{match_rate*100:.1f}%** ({matched:,}/{total_rows:,})\n")
    lines.append(f"- Total ceded exposure (demo): **${total_exposure:,.0f}**\n")

    lines.append("\n## Key insights (demo)\n")
    lines.append("- Exposure is concentrated across a subset of counterparties.\n")
    lines.append("- Utilization can be assessed quickly against limits and ratings.\n")
    lines.append("- Under stress, utilization shifts help prioritize governance review.\n")

    if top_flags is not None and not top_flags.empty:
        lines.append("\n### Top stressed counterparties (illustrative)\n")
        for _, r in top_flags.iterrows():
            lines.append(f"- {r['counterparty_name']}: **{float(r['stressed_utilization_pct']):.1f}%** stressed utilization\n")

    lines.append("\n## Governance / audit posture\n")
    if governance_path:
        lines.append(f"- Governance report artifact: `{governance_path}`\n")
    lines.append("- Join confidence is shown on portfolio rows to support explainability.\n")
    lines.append("- This demo is synthetic; production implementations include lineage + controls.\n")

    if meeting_notes:
        lines.append("\n## Notes for discussion\n")
        lines.append(meeting_notes.strip() + "\n")

    lines.append("\n## Suggested next step (pilot)\n")
    lines.append(
        "- Run a 2–4 week proof-of-value on one program: define scope, map data sources, validate reconciliations, and deliver a governance pack.\n"
    )

    return "".join(lines)
