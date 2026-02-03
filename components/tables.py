"""
components/tables.py

Reusable table renderers and formatting helpers.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd


def show_table(df: pd.DataFrame, title: str | None = None, height: int | None = None):
    if title:
        st.subheader(title)
    if df.empty:
        st.info("No rows to display.")
        return
    st.dataframe(df, use_container_width=True, height=height, hide_index=True)


def kpi_row(items: list[tuple[str, str]]):
    """items: list of (label, value)"""
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)
