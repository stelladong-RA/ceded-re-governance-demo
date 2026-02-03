"""
services/reconciliation.py

Reconciliation and exception queues across:
- Statements vs Cash (cash pairing)
- Claims vs Recoverables (demo stub)
- Portfolio governance checks (missing joins, missing counterparties, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd


# =====================================================
# Cash pairing reconciliation summary
# =====================================================

@dataclass(frozen=True)
class ReconciliationSummary:
    matched: int
    unmatched: int
    variance_count: int
    avg_confidence: float


def cash_recon_summary(
    pairs: pd.DataFrame,
    amount_tolerance_usd: float = 250.0
) -> ReconciliationSummary:
    """
    Compute high-level cash pairing KPIs.

    Expected columns (from services.cash_engine.pair_cash_to_statements):
      - match_type
      - confidence
      - variance_usd
    """
    if pairs is None or pairs.empty:
        return ReconciliationSummary(matched=0, unmatched=0, variance_count=0, avg_confidence=0.0)

    # Treat any non-unmatched as "paired" (includes reference_not_found/candidate/exact_reference)
    matched = int((pairs["match_type"] != "unmatched").sum()) if "match_type" in pairs.columns else 0
    unmatched = int((pairs["match_type"] == "unmatched").sum()) if "match_type" in pairs.columns else 0

    # Variance only meaningful for paired rows
    if "variance_usd" in pairs.columns and "match_type" in pairs.columns:
        variance_count = int(((pairs["match_type"] != "unmatched") & (pairs["variance_usd"] > amount_tolerance_usd)).sum())
    else:
        variance_count = 0

    avg_conf = float(pairs.get("confidence", pd.Series(dtype=float)).fillna(0.0).mean()) if len(pairs) else 0.0

    return ReconciliationSummary(
        matched=matched,
        unmatched=unmatched,
        variance_count=variance_count,
        avg_confidence=avg_conf,
    )


# =====================================================
# Cash exception queue (ops-ready)
# =====================================================

def build_cash_exception_queue(
    pairs: pd.DataFrame,
    min_confidence: float = 0.25,
    amount_tolerance_usd: float = 250.0,
) -> pd.DataFrame:
    """
    Exceptions suitable for an operations queue:
    - Unmatched cash
    - Reference provided but not found
    - Low confidence matches
    - High amount variance

    Expected columns:
      - cash_id
      - txn_date
      - counterparty_id
      - amount_usd
      - matched_statement_id
      - match_type
      - confidence
      - variance_usd
      - match_reason
    """
    if pairs is None or pairs.empty:
        return pd.DataFrame(
            columns=[
                "exception_type", "priority", "cash_id", "txn_date", "counterparty_id", "amount_usd",
                "matched_statement_id", "match_type", "confidence", "variance_usd", "match_reason"
            ]
        )

    q = pairs.copy()

    # Normalize missing columns to avoid KeyErrors
    for col in ["cash_id", "txn_date", "counterparty_id", "amount_usd", "matched_statement_id",
                "match_type", "confidence", "variance_usd", "match_reason"]:
        if col not in q.columns:
            q[col] = pd.NA

    q["exception_type"] = "None"

    # 1) Unmatched
    q.loc[q["match_type"] == "unmatched", "exception_type"] = "Unmatched cash"

    # 2) Reference provided but not found (common operational issue)
    q.loc[q["match_type"] == "reference_not_found", "exception_type"] = "Statement reference not found"

    # 3) Low confidence (candidate matches)
    q.loc[(q["match_type"].isin(["candidate"])) & (q["confidence"].fillna(0) < float(min_confidence)), "exception_type"] = "Low-confidence match"

    # 4) Amount variance (paired but outside tolerance)
    q.loc[(q["match_type"] != "unmatched") & (q["variance_usd"].fillna(0) > float(amount_tolerance_usd)), "exception_type"] = "Amount variance"

    q = q[q["exception_type"] != "None"].copy()

    # Priority scoring (simple, demo-friendly)
    # Higher amount and lower confidence = higher priority
    q["priority"] = (
        (q["amount_usd"].fillna(0).abs() / 1_000_000.0).clip(0, 5) +
        (1.0 - q["confidence"].fillna(0)).clip(0, 1) * 2.0 +
        (q["variance_usd"].fillna(0) / max(float(amount_tolerance_usd), 1.0)).clip(0, 5)
    ).round(2)

    cols = [
        "exception_type", "priority", "cash_id", "txn_date", "counterparty_id", "amount_usd",
        "matched_statement_id", "match_type", "confidence", "variance_usd", "match_reason"
    ]

    return q[cols].sort_values(["exception_type", "priority"], ascending=[True, False]).reset_index(drop=True)


# =====================================================
# Portfolio governance (used by Demo 1)
# =====================================================

def portfolio_data_quality(portfolio: pd.DataFrame) -> Dict[str, float]:
    """
    Quick data-quality KPIs for consulting discussion:
    - placement join match rate
    - missing counterparty metadata
    """
    if portfolio is None or portfolio.empty:
        return {
            "rows_total": 0.0,
            "placement_match_rate": 0.0,
            "missing_counterparty_rate": 0.0,
        }

    total = float(max(len(portfolio), 1))
    matched = float((portfolio.get("join_confidence") == "matched").sum()) if "join_confidence" in portfolio.columns else 0.0
    missing_cp = float(portfolio.get("counterparty_name").isna().sum()) if "counterparty_name" in portfolio.columns else 0.0

    return {
        "rows_total": total,
        "placement_match_rate": matched / total,
        "missing_counterparty_rate": missing_cp / total,
    }
