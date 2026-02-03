# services/exposure_engine.py

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
import numpy as np


# =====================================================
# Configuration
# =====================================================

@dataclass
class ExposureConfig:
    """
    Controls how ceded exposure is calculated.
    """
    ceded_basis: str = "limit"   # limit | sum_insured | written_premium


# =====================================================
# Internal Utilities
# =====================================================

def _find_cession_col(df: pd.DataFrame) -> str | None:
    """
    Find the best available cession/share column.
    """

    candidates = [
        "cession_pct",
        "ceded_share",
        "share_pct",
        "quota_share",
        "participation_pct",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    return None


def _compute_quality_score(df: pd.DataFrame) -> pd.Series:
    """
    Simple governance completeness score (0–100).
    """

    fields = [
        "policy_id",
        "treaty_id",
        "counterparty_id",
        "ceded_limit_usd",
        "rating",
    ]

    available = [f for f in fields if f in df.columns]

    if not available:
        return pd.Series(0, index=df.index)

    present = df[available].notna().mean(axis=1)

    return (present * 100).round(1)


def _safe_numeric(s: pd.Series) -> pd.Series:
    """
    Coerce to numeric safely.
    """
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


# =====================================================
# Core Builder
# =====================================================

def build_portfolio_exposure(
    pas: pd.DataFrame,
    placements: pd.DataFrame,
    counterparties: pd.DataFrame,
    cfg: ExposureConfig,
) -> pd.DataFrame:
    """
    Build unified ceded exposure portfolio.

    Output:
        One row per policy × placement × counterparty
        with governance metadata.
    """

    # -----------------------------
    # Join PAS + Placements
    # -----------------------------

    join_key = "policy_id" if "policy_id" in pas.columns else "program_id"

    df = pas.merge(
        placements,
        on=join_key,
        how="left",
        suffixes=("", "_plc"),
        indicator=True,
    )

    # Join confidence
    df["join_confidence"] = np.where(
        df["_merge"] == "both",
        "matched",
        "unmatched"
    )

    df.drop(columns="_merge", inplace=True)


    # -----------------------------
    # Join Counterparties
    # -----------------------------

    if "counterparty_id" in df.columns and "counterparty_id" in counterparties.columns:
        df = df.merge(
            counterparties,
            on="counterparty_id",
            how="left",
            suffixes=("", "_cp")
        )


    # -----------------------------
    # Select Exposure Basis
    # -----------------------------

    if cfg.ceded_basis == "limit" and "ceded_limit_usd" in df.columns:
        base = df["ceded_limit_usd"]

    elif cfg.ceded_basis == "sum_insured" and "sum_insured_usd" in df.columns:
        base = df["sum_insured_usd"]

    elif cfg.ceded_basis == "written_premium" and "written_premium_usd" in df.columns:
        base = df["written_premium_usd"]

    else:
        # Safe fallback
        base = df.get("ceded_limit_usd", 0)


    df["basis_amount_usd"] = _safe_numeric(base)

    # Store basis for auditability (IMPORTANT)
    df["ceded_basis"] = cfg.ceded_basis


    # -----------------------------
    # Compute Ceded Share
    # -----------------------------

    cession_col = _find_cession_col(df)

    if cession_col:

        share = _safe_numeric(df[cession_col])

        # Convert percent → decimal
        if share.dropna().between(0, 100).all():
            share = share / 100.0

        df["cession_source"] = cession_col

    else:
        # Demo-safe assumption
        share = pd.Series(0.25, index=df.index)
        df["cession_source"] = "demo_default_25pct"


    df["ceded_share"] = share


    # -----------------------------
    # Compute Ceded Exposure
    # -----------------------------

    df["ceded_exposure_usd"] = (
        df["basis_amount_usd"] * df["ceded_share"]
    ).round(2)


    # -----------------------------
    # Governance / Quality
    # -----------------------------

    df["data_quality_score"] = _compute_quality_score(df)

    df["exposure_source_note"] = np.where(
        df["cession_source"] == "demo_default_25pct",
        "Demo assumption: 25% default share",
        "Client-provided share"
    )

    df["exposure_last_updated"] = pd.Timestamp.utcnow().isoformat()


    return df


# =====================================================
# Analytics Helpers
# =====================================================

def top_concentrations(
    portfolio: pd.DataFrame,
    by: str,
    n: int = 10
) -> pd.DataFrame:
    """
    Top exposure concentrations by dimension.
    """

    if by not in portfolio.columns:
        return pd.DataFrame()

    out = (
        portfolio
        .groupby(by, dropna=False)["ceded_exposure_usd"]
        .sum()
        .reset_index()
        .sort_values("ceded_exposure_usd", ascending=False)
        .head(n)
    )

    return out


def exposure_heatmap_table(
    portfolio: pd.DataFrame,
    index: str,
    columns: str,
) -> pd.DataFrame:
    """
    Pivot table for heatmap visualization.
    """

    if index not in portfolio.columns or columns not in portfolio.columns:
        return pd.DataFrame()

    pivot = pd.pivot_table(
        portfolio,
        values="ceded_exposure_usd",
        index=index,
        columns=columns,
        aggfunc="sum",
        fill_value=0,
    )

    return pivot.round(0)
