# services/credit_engine.py

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
import numpy as np


# -----------------------------
# Config
# -----------------------------

@dataclass
class CreditConfig:
    default_limit: float = 250_000_000
    high_risk_threshold: float = 90.0     # %
    watch_threshold: float = 70.0         # %


# -----------------------------
# Limits
# -----------------------------

def build_counterparty_limits(
    counterparties: pd.DataFrame,
    cfg: CreditConfig
) -> pd.DataFrame:
    """
    Build credit limits per reinsurer based on rating.
    """

    df = counterparties.copy()

    # Rating multipliers (illustrative)
    rating_map = {
        "AAA": 1.5,
        "AA": 1.3,
        "A": 1.1,
        "BBB": 1.0,
        "BB": 0.8,
        "B": 0.6,
        "CCC": 0.4,
    }

    df["rating_factor"] = df["rating"].map(rating_map).fillna(0.9)

    base = df.get("base_limit_usd", cfg.default_limit)

    df["credit_limit_usd"] = base * df["rating_factor"]

    return df[
        [
            "counterparty_id",
            "counterparty_name",
            "rating",
            "credit_limit_usd",
        ]
    ]


# -----------------------------
# Utilization
# -----------------------------

def compute_utilization(
    portfolio: pd.DataFrame,
    limits: pd.DataFrame,
    cfg: CreditConfig = CreditConfig()
) -> pd.DataFrame:
    """
    Compute counterparty credit utilization (robust version).
    """

    df_port = portfolio.copy()
    df_lim = limits.copy()


    # -----------------------------
    # Normalize join keys
    # -----------------------------

    df_port["counterparty_id"] = (
        df_port["counterparty_id"]
        .astype(str)
        .str.strip()
    )

    df_lim["counterparty_id"] = (
        df_lim["counterparty_id"]
        .astype(str)
        .str.strip()
    )


    # -----------------------------
    # Aggregate exposure (ID only)
    # -----------------------------

    agg = (
        df_port
        .groupby("counterparty_id", dropna=False)["ceded_exposure_usd"]
        .sum()
        .reset_index()
        .rename(columns={
            "ceded_exposure_usd": "total_ceded_exposure_usd"
        })
    )


    # -----------------------------
    # Join limits + metadata
    # -----------------------------

    df = agg.merge(
        df_lim,
        on="counterparty_id",
        how="left"
    )


    # -----------------------------
    # Ensure display fields exist
    # -----------------------------

    if "counterparty_name" not in df.columns:
        df["counterparty_name"] = df["counterparty_id"]

    if "rating" not in df.columns:
        df["rating"] = "NR"


    # -----------------------------
    # Defaults
    # -----------------------------

    df["credit_limit_usd"] = df["credit_limit_usd"].fillna(cfg.default_limit)


    # -----------------------------
    # Utilization (%)
    # -----------------------------

    df["utilization_pct"] = np.where(
        df["credit_limit_usd"] > 0,
        df["total_ceded_exposure_usd"] / df["credit_limit_usd"] * 100,
        np.nan
    ).round(1)


    # -----------------------------
    # Risk flags
    # -----------------------------

    df["flag"] = df["utilization_pct"].apply(
        lambda x: _risk_flag(x, cfg)
    )


    return df.sort_values(
        "utilization_pct",
        ascending=False
    )

def _risk_flag(x: float, cfg: CreditConfig) -> str:
    if pd.isna(x):
        return "Unknown"

    if x >= cfg.high_risk_threshold:
        return "High Risk"
    elif x >= cfg.watch_threshold:
        return "Watchlist"
    else:
        return "Normal"


# -----------------------------
# Stress Testing
# -----------------------------

def stress_scenario(
    utilization: pd.DataFrame,
    shock_pct: float,
    cfg: CreditConfig = CreditConfig()
) -> pd.DataFrame:
    """
    Apply exposure stress scenario.
    """

    df = utilization.copy()

    shock_factor = 1.0 + shock_pct / 100.0


    # -----------------------------
    # Apply shock
    # -----------------------------

    df["stressed_exposure_usd"] = (
        df["total_ceded_exposure_usd"] * shock_factor
    )


    df["stressed_utilization_pct"] = np.where(
        df["credit_limit_usd"] > 0,
        df["stressed_exposure_usd"] / df["credit_limit_usd"] * 100,
        np.nan
    ).round(1)


    # -----------------------------
    # Stress flags
    # -----------------------------

    df["stressed_flag"] = df["stressed_utilization_pct"].apply(
        lambda x: _risk_flag(x, cfg)
    )


    return df
