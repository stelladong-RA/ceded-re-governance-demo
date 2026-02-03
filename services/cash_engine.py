"""
services/cash_engine.py

Cash Pairing demo logic:
- Match cash transactions to Statements of Account (SOA)
- Produce a match table with confidence + reason codes
- Demo-safe: works even when some optional columns are missing
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# =====================================================
# Config
# =====================================================

@dataclass(frozen=True)
class CashPairingConfig:
    # Matching tolerances
    amount_tolerance_usd: float = 250.0
    date_window_days: int = 45  # cash within +/- window of invoice_date

    # Confidence weights (demo)
    w_exact_reference: float = 0.80
    w_counterparty_match: float = 0.15
    w_amount_close: float = 0.35
    w_date_close: float = 0.20
    w_reference_present_but_missing: float = 0.10  # reference provided but statement not found

    # Candidate selection
    max_candidates_per_cash: int = 50  # guardrail for demo data


# =====================================================
# Helpers
# =====================================================

def _to_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=False)

def _ensure_col(df: pd.DataFrame, col: str, default=None) -> pd.DataFrame:
    if col not in df.columns:
        df[col] = default
    return df

def _first_existing(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _as_str(series: pd.Series) -> pd.Series:
    # Keep NaN as NaN, convert non-null to string
    return series.where(series.isna(), series.astype(str))

def _safe_abs_days(a: pd.Series, b: pd.Series) -> pd.Series:
    a = _to_dt(a)
    b = _to_dt(b)
    return (a - b).abs().dt.days

def _clip01(x: pd.Series) -> pd.Series:
    return x.clip(lower=0.0, upper=1.0)


# =====================================================
# Main
# =====================================================

def pair_cash_to_statements(
    cash: pd.DataFrame,
    statements: pd.DataFrame,
    cfg: CashPairingConfig = CashPairingConfig(),
) -> pd.DataFrame:
    """
    Return row-level matches:
      cash_id -> matched_statement_id (best candidate), plus confidence score, variance, and reason.

    Required cash columns (canonical):
      - cash_id
      - counterparty_id
      - amount_usd
      - txn_date (or payment_date as fallback)

    Required statements columns (canonical):
      - statement_id
      - counterparty_id
      - invoice_date
      - amount_due_usd (or a reasonable fallback)
    """

    c = cash.copy()
    s = statements.copy()

    # --- Ensure minimal columns exist ---
    c = _ensure_col(c, "cash_id")
    c = _ensure_col(c, "counterparty_id")
    c = _ensure_col(c, "amount_usd")
    c = _ensure_col(c, "txn_date")
    c = _ensure_col(c, "payment_date")
    c = _ensure_col(c, "statement_id")
    c = _ensure_col(c, "currency", "USD")
    c = _ensure_col(c, "memo", "")
    c = _ensure_col(c, "direction", "in")
    c = _ensure_col(c, "bank_reference", "")

    s = _ensure_col(s, "statement_id")
    s = _ensure_col(s, "counterparty_id")
    s = _ensure_col(s, "invoice_date")
    s = _ensure_col(s, "due_date")
    s = _ensure_col(s, "currency", "USD")

    # Amount due field (statements may vary)
    stmt_amt_col = _first_existing(s, ["amount_due_usd", "net_due_usd", "amount_usd", "balance_usd"])
    if stmt_amt_col is None:
        # demo-safe: create empty
        s["amount_due_usd"] = np.nan
        stmt_amt_col = "amount_due_usd"

    # --- Normalize dtypes ---
    c["cash_id"] = _as_str(c["cash_id"])
    c["counterparty_id"] = _as_str(c["counterparty_id"])
    c["statement_id"] = _as_str(c["statement_id"])

    s["statement_id"] = _as_str(s["statement_id"])
    s["counterparty_id"] = _as_str(s["counterparty_id"])

    c["amount_usd"] = pd.to_numeric(c["amount_usd"], errors="coerce")
    s[stmt_amt_col] = pd.to_numeric(s[stmt_amt_col], errors="coerce")

    # Use txn_date; fallback to payment_date
    c["txn_date"] = _to_dt(c["txn_date"])
    pay_dt = _to_dt(c["payment_date"])
    c["txn_date"] = c["txn_date"].fillna(pay_dt)

    s["invoice_date"] = _to_dt(s["invoice_date"])
    s["due_date"] = _to_dt(s["due_date"])

    # =====================================================
    # 1) Exact reference match (if cash.statement_id is present)
    # =====================================================
    s_ref = s[["statement_id", "counterparty_id", stmt_amt_col, "invoice_date", "due_date"]].copy()
    s_ref = s_ref.rename(columns={stmt_amt_col: "statement_amount_usd"})

    exact = c.merge(
        s_ref,
        on=["statement_id", "counterparty_id"],
        how="left",
        suffixes=("", "_stmt"),
        indicator=True,
    )

    exact["has_reference"] = exact["statement_id"].notna() & (exact["statement_id"].astype(str).str.len() > 0)
    exact["is_exact_ref_match"] = exact["_merge"].eq("both") & exact["statement_amount_usd"].notna()

    exact["variance_usd"] = (exact["amount_usd"] - exact["statement_amount_usd"]).abs()
    exact["days_diff"] = _safe_abs_days(exact["txn_date"], exact["invoice_date"])

    exact = exact.drop(columns=["_merge"])

    # =====================================================
    # 2) Candidate matching for non-exact rows
    #    Match by counterparty + amount/date proximity
    # =====================================================
    no_exact = exact[~exact["is_exact_ref_match"]].copy()

    # Build candidate pairs within same counterparty
    cand = no_exact.merge(
        s_ref[["statement_id", "counterparty_id", "statement_amount_usd", "invoice_date"]],
        on="counterparty_id",
        how="left",
        suffixes=("", "_cand"),
    )

    # Compute candidate diffs
    cand["variance_usd"] = (cand["amount_usd"] - cand["statement_amount_usd"]).abs()
    cand["days_diff"] = _safe_abs_days(cand["txn_date"], cand["invoice_date"])

    # Filter within tolerances
    tol_amt = max(float(cfg.amount_tolerance_usd), 1.0)
    tol_days = max(int(cfg.date_window_days), 1)

    cand = cand[
        (cand["days_diff"].isna() | (cand["days_diff"] <= tol_days)) &
        (cand["variance_usd"].isna() | (cand["variance_usd"] <= tol_amt))
    ].copy()

    # Guardrail: cap candidate explosion per cash row
    cand = cand.sort_values(["cash_id", "variance_usd", "days_diff"], ascending=[True, True, True])
    cand["cand_rank"] = cand.groupby("cash_id").cumcount() + 1
    cand = cand[cand["cand_rank"] <= cfg.max_candidates_per_cash].copy()

    # Score candidates (0..1)
    cand["score_amount"] = _clip01(1.0 - (cand["variance_usd"] / tol_amt))
    cand["score_date"] = _clip01(1.0 - (cand["days_diff"] / tol_days))
    cand["score_counterparty"] = 1.0  # always true in this candidate set

    cand["candidate_score"] = (
        cfg.w_counterparty_match * cand["score_counterparty"] +
        cfg.w_amount_close * cand["score_amount"] +
        cfg.w_date_close * cand["score_date"]
    ).fillna(0.0)

    # Pick best candidate per cash_id
    best = (
        cand.sort_values(["cash_id", "candidate_score", "variance_usd", "days_diff"], ascending=[True, False, True, True])
            .groupby("cash_id", as_index=False)
            .head(1)
            [["cash_id", "statement_id", "candidate_score", "variance_usd", "days_diff"]]
            .rename(columns={"statement_id": "candidate_statement_id"})
    )

    cand_counts = (
        cand.groupby("cash_id", as_index=False)
            .size()
            .rename(columns={"size": "candidate_count"})
    )

    # Merge best candidate back into no_exact rows
    no_exact = no_exact.merge(best, on="cash_id", how="left")
    no_exact = no_exact.merge(cand_counts, on="cash_id", how="left")
    no_exact["candidate_count"] = no_exact["candidate_count"].fillna(0).astype(int)

    # =====================================================
    # 3) Final selection + confidence + reason
    # =====================================================

    out = exact.merge(
        no_exact[["cash_id", "candidate_statement_id", "candidate_score", "candidate_count"]],
        on="cash_id",
        how="left",
        suffixes=("", "_cand"),
    )

    out["matched_statement_id"] = np.where(
        out["is_exact_ref_match"],
        out["statement_id"],
        out["candidate_statement_id"],
    )

    out["match_type"] = np.select(
        [
            out["is_exact_ref_match"],
            out["matched_statement_id"].notna(),
            out["has_reference"] & out["matched_statement_id"].isna(),
        ],
        [
            "exact_reference",
            "candidate",
            "reference_not_found",
        ],
        default="unmatched",
    )

    # Confidence
    out["confidence"] = np.select(
        [
            out["match_type"].eq("exact_reference"),
            out["match_type"].eq("candidate"),
            out["match_type"].eq("reference_not_found"),
        ],
        [
            float(cfg.w_exact_reference),
            out["candidate_score"].fillna(0.0).astype(float),
            float(cfg.w_reference_present_but_missing),
        ],
        default=0.0,
    ).clip(0.0, 1.0)

    # Match reason (human-readable, Teams/ops-friendly)
    def _reason(row) -> str:
        mt = row.get("match_type")
        if mt == "exact_reference":
            if pd.notna(row.get("variance_usd")) and row["variance_usd"] <= cfg.amount_tolerance_usd:
                return "Matched by statement reference (within tolerance)"
            return "Matched by statement reference (amount variance flagged)"
        if mt == "candidate":
            n = int(row.get("candidate_count") or 0)
            if pd.notna(row.get("variance_usd")) and row["variance_usd"] <= cfg.amount_tolerance_usd:
                return f"Matched by counterparty + amount/date proximity (best of {n} candidates)"
            return f"Matched by counterparty + date proximity (amount variance flagged; best of {n} candidates)"
        if mt == "reference_not_found":
            return "Statement reference provided but not found in SOA"
        return "No suitable SOA candidate within tolerance window"

    out["match_reason"] = out.apply(_reason, axis=1)

    # Recompute variance/days for candidate matches (because exact rows had stmt fields)
    # If candidate chosen, variance_usd/days_diff should reflect candidate comparison.
    # We already carried best's variance/days in best; keep those if exact was not used.
    out["variance_usd"] = np.where(
        out["match_type"].eq("exact_reference"),
        out["variance_usd"],
        out["variance_usd_cand"] if "variance_usd_cand" in out.columns else out["variance_usd"],
    )

    out["days_diff"] = np.where(
        out["match_type"].eq("exact_reference"),
        out["days_diff"],
        out["days_diff_cand"] if "days_diff_cand" in out.columns else out["days_diff"],
    )

    # =====================================================
    # Output columns (UI-friendly)
    # =====================================================
    keep = [
        "cash_id",
        "txn_date",
        "payment_date",
        "counterparty_id",
        "amount_usd",
        "currency",
        "direction",
        "bank_reference",
        "memo",
        "statement_id",            # original reference if supplied
        "matched_statement_id",
        "match_type",
        "confidence",
        "variance_usd",
        "days_diff",
        "candidate_count",
        "match_reason",
    ]

    for col in keep:
        if col not in out.columns:
            out[col] = np.nan

    return out[keep].sort_values(["match_type", "confidence"], ascending=[True, False]).reset_index(drop=True)
