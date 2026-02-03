# components/governance.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import streamlit as st
import pandas as pd


def _load_governance(root: Path) -> Dict[str, Any] | None:
    path = root / "data" / "processed" / "_governance.json"
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _status_badge(ok: bool) -> str:
    return "🟢 OK" if ok else "🔴 Attention"


def _health_color(pct: float | None) -> str:
    if pct is None:
        return "⚪ N/A"
    if pct >= 0.98:
        return "🟢"
    if pct >= 0.90:
        return "🟡"
    return "🔴"


def show_governance_panel():
    """
    Renders an executive-grade governance / audit trail panel.
    """

    root = Path(st.session_state.get("demo_root", "."))
    gov = _load_governance(root)

    if gov is None:
        st.warning("No governance report found. Run the ingestion pipeline.")
        return

    # ----------------------------
    # Header
    # ----------------------------
    st.markdown("## Governance & Audit Trail")
    st.caption("Data lineage, validation status, and transformation transparency")

    created = gov.get("created_at_utc", "n/a")
    version = gov.get("pipeline_version", "n/a")

    val = gov.get("validation_summary", {})
    errors = int(val.get("errors", 0))
    warnings = int(val.get("warnings", 0))

    healthy = errors == 0

    # ----------------------------
    # Executive KPIs
    # ----------------------------
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Pipeline status", _status_badge(healthy))

    with c2:
        st.metric("Last refresh (UTC)", created.split(".")[0])

    with c3:
        st.metric("Pipeline version", version)

    with c4:
        st.metric("Validation errors", errors)

    if not healthy:
        st.error("Validation errors detected. Governance outputs may be unreliable.")

    st.markdown("---")

    # ----------------------------
    # Data Quality Scorecard
    # ----------------------------
    st.subheader("Data Quality Scorecard")
    st.caption("Completeness of critical fields per source")

    tables = gov.get("tables", {}).get("canonical", {})
    key_health = gov.get("key_fields_health", {})

    rows = []

    for table, fields in key_health.items():
        if not fields:
            continue

        for col, pct in fields.items():
            rows.append(
                {
                    "Table": table,
                    "Field": col,
                    "Completeness": None if pct is None else round(pct * 100, 2),
                    "Status": _health_color(pct),
                }
            )

    if rows:
        scorecard = pd.DataFrame(rows)

        st.dataframe(
            scorecard,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Completeness": st.column_config.ProgressColumn(
                    "Completeness %",
                    min_value=0,
                    max_value=100,
                    format="%.1f",
                )
            },
        )
    else:
        st.info("No key-field health metrics available.")

    st.markdown("---")

    # ----------------------------
    # Derived / Assumptions
    # ----------------------------
    st.subheader("Derived Fields & Assumptions")
    st.caption("Fields created during canonicalization")

    derived = gov.get("derived_fields", {})

    if derived:
        for table, fields in derived.items():
            st.markdown(f"**{table.upper()}**")
            for f in fields:
                st.code(f, language="text")
    else:
        st.success("No derived fields. Canonical schema matches raw sources.")

    st.markdown("---")

    # ----------------------------
    # Join Coverage
    # ----------------------------
    st.subheader("Join Coverage Indicators")
    st.caption("How well sources connect across the data model")

    joins = gov.get("join_coverage", {})

    if joins:
        join_rows = []
        for name, pct in joins.items():
            join_rows.append(
                {
                    "Join": name.replace("_", " → "),
                    "Match rate %": None if pct is None else round(pct * 100, 1),
                    "Status": _health_color(pct),
                }
            )

        join_df = pd.DataFrame(join_rows)

        st.dataframe(join_df, use_container_width=True, hide_index=True)
    else:
        st.info("No join coverage metrics recorded.")

    st.markdown("---")

    # ----------------------------
    # Technical Lineage (Collapsed)
    # ----------------------------
    with st.expander("Technical Lineage & Table Profiles"):
        st.caption("Detailed metadata for auditors and IT")

        for layer in ["raw", "canonical", "processed"]:
            layer_tables = gov.get("tables", {}).get(layer, {})
            if not layer_tables:
                continue

            st.markdown(f"### {layer.capitalize()} Layer")

            for name, meta in layer_tables.items():
                st.markdown(f"**{name}**")
                st.json(meta)

    # ----------------------------
    # Footer
    # ----------------------------
    st.markdown("---")
    st.caption(
        "Governance artifacts generated automatically by the Synpulse demo pipeline. "
        "In production: versioned, access-controlled, and integrated with enterprise lineage tools."
    )
