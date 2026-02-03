"""
components/filters.py

Reusable Streamlit filter widgets.
Keep these widgets thin and return values; no business logic here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import streamlit as st
import pandas as pd


@dataclass(frozen=True)
class PortfolioFilters:
    program_ids: List[str]
    lines_of_business: List[str]
    regions: List[str]
    perils: List[str]
    counterparties: List[str]


def multiselect_with_all(label: str, options: list, default_all: bool = True, help: str | None = None):
    """Common pattern: allow user to pick subset or All."""
    if default_all:
        default = options
    else:
        default = options[:1] if options else []
    return st.multiselect(label, options=options, default=default, help=help)


def portfolio_filters_sidebar(portfolio_df: pd.DataFrame) -> PortfolioFilters:
    """
    Sidebar filters for the unified portfolio table.
    Expects columns: program_id, line_of_business, region, peril, counterparty_name
    """
    st.sidebar.subheader("Filters")

    program_opts = sorted([x for x in portfolio_df["program_id"].dropna().unique().tolist()])
    lob_opts = sorted([x for x in portfolio_df["line_of_business"].dropna().unique().tolist()])
    region_opts = sorted([x for x in portfolio_df["region"].dropna().unique().tolist()])
    peril_opts = sorted([x for x in portfolio_df["peril"].dropna().unique().tolist()])
    cp_opts = sorted([x for x in portfolio_df["counterparty_name"].dropna().unique().tolist()])

    program_ids = multiselect_with_all("Program", program_opts, help="Filter by ceded program / book.")
    lobs = multiselect_with_all("Line of Business", lob_opts)
    regions = multiselect_with_all("Region", region_opts)
    perils = multiselect_with_all("Peril", peril_opts)
    cps = multiselect_with_all("Counterparty", cp_opts)

    return PortfolioFilters(
        program_ids=program_ids,
        lines_of_business=lobs,
        regions=regions,
        perils=perils,
        counterparties=cps,
    )


def apply_portfolio_filters(portfolio_df: pd.DataFrame, f: PortfolioFilters) -> pd.DataFrame:
    """Apply a PortfolioFilters object to a portfolio DataFrame."""
    df = portfolio_df.copy()
    if f.program_ids:
        df = df[df["program_id"].isin(f.program_ids)]
    if f.lines_of_business:
        df = df[df["line_of_business"].isin(f.lines_of_business)]
    if f.regions:
        df = df[df["region"].isin(f.regions)]
    if f.perils:
        df = df[df["peril"].isin(f.perils)]
    if f.counterparties:
        df = df[df["counterparty_name"].isin(f.counterparties)]
    return df
