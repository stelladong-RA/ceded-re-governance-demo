"""
utils/validators.py

Lightweight validation + normalization for demo datasets.
Goals:
- Prevent common demo breakages (dtype mismatch on joins, bad dates, numeric strings)
- Produce actionable governance flags (missing columns, null rates, join coverage)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd


# =====================================================
# Types
# =====================================================

@dataclass(frozen=True)
class ValidationIssue:
    table: str
    severity: str  # "error" | "warning"
    message: str


# =====================================================
# Small helpers
# =====================================================

def _as_string_id(s: pd.Series) -> pd.Series:
    """
    Convert an ID-like series to string consistently.
    Handles floats that look like 101.0 -> "101".
    """
    if s is None:
        return s
    # Work with pandas "string" dtype (nullable)
    s2 = s.astype("string")

    # If values look numeric, remove trailing .0
    # (common when CSVs infer IDs as floats)
    s2 = s2.str.replace(r"\.0$", "", regex=True)

    # Trim
    return s2.str.strip()


def _coerce_numeric(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


def _coerce_date(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")


def require_columns(df: pd.DataFrame, table: str, cols: List[str]) -> List[ValidationIssue]:
    missing = [c for c in cols if c not in df.columns]
    if not missing:
        return []
    return [ValidationIssue(table=table, severity="error", message=f"Missing columns: {missing}")]


def require_one_of(df: pd.DataFrame, table: str, col_options: List[str], label: str) -> List[ValidationIssue]:
    """
    Require at least one column in col_options exists.
    """
    exists = [c for c in col_options if c in df.columns]
    if exists:
        return []
    return [ValidationIssue(table=table, severity="error", message=f"Missing required field '{label}'. Need one of: {col_options}")]


def non_null(df: pd.DataFrame, table: str, col: str, max_null_rate: float = 0.25, severity: str = "warning") -> List[ValidationIssue]:
    if col not in df.columns:
        return []
    null_rate = float(df[col].isna().mean())
    if null_rate <= max_null_rate:
        return []
    return [ValidationIssue(table=table, severity=severity, message=f"High null rate for '{col}': {null_rate:.0%} (threshold {max_null_rate:.0%})")]


def join_coverage_warning(
    left: pd.DataFrame,
    right: pd.DataFrame,
    table: str,
    left_key: str,
    right_key: str,
    label: str,
    min_match_rate: float = 0.85
) -> List[ValidationIssue]:
    """
    Quick governance signal: how well does one table link to another?
    """
    if left is None or right is None:
        return []
    if left_key not in left.columns or right_key not in right.columns:
        return []
    if len(left) == 0:
        return []

    left_vals = left[left_key].dropna().astype("string")
    right_vals = set(right[right_key].dropna().astype("string").unique())

    if len(left_vals) == 0:
        return []

    match_rate = float(left_vals.isin(right_vals).mean())

    if match_rate >= min_match_rate:
        return []

    return [
        ValidationIssue(
            table=table,
            severity="warning",
            message=f"Low join coverage for {label}: {match_rate:.0%} matched (threshold {min_match_rate:.0%})."
        )
    ]


# =====================================================
# Core normalization hook
# =====================================================

def validate_dataframe(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """
    Lightweight validation + normalization hook for pipeline + pages.

    What it does:
      - normalize column names
      - trim whitespace on string columns
      - coerce *_id columns to string (prevents merge dtype issues)
      - coerce known numeric/date columns where present
    """
    if df is None:
        raise ValueError(f"{table}: dataframe is None")

    # Normalize column names
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Trim object/string columns
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype("string").str.strip()

    # Coerce any *_id columns to string
    for c in df.columns:
        if c.endswith("_id") or c in {"policy_id", "program_id", "treaty_id", "statement_id", "cash_id", "claim_id"}:
            df[c] = _as_string_id(df[c])

    # Common numeric columns (coerce if present)
    numeric_cols = [
        "amount_usd", "amount_due_usd", "net_due_usd",
        "limit_usd", "ceded_limit_usd", "ceded_premium_usd",
        "sum_insured_usd", "written_premium_usd",
        "paid_usd", "paid_amount_usd", "loss_usd",
        "cession_pct", "share_pct", "ceded_share",
    ]
    for c in numeric_cols:
        _coerce_numeric(df, c)

    # Common dates
    date_cols = ["txn_date", "payment_date", "invoice_date", "due_date", "inception_date", "expiry_date"]
    for c in date_cols:
        _coerce_date(df, c)

    return df


# =====================================================
# Demo table validation (governance story)
# =====================================================

def validate_demo_tables(tables: Dict[str, pd.DataFrame]) -> List[ValidationIssue]:
    """
    Basic schema checks for demo tables.
    Keep this simple — enough to support governance narratives.
    """
    issues: List[ValidationIssue] = []

    # Guard: if table missing entirely
    required_tables = ["pas", "placements", "counterparties", "cash", "statements"]
    for t in required_tables:
        if t not in tables or tables[t] is None:
            issues.append(ValidationIssue(table=t, severity="error", message="Table missing or not loaded"))
            return issues  # can't proceed meaningfully

    # Normalize inputs first (so downstream checks reflect actual usage)
    pas = validate_dataframe(tables["pas"], "pas")
    plc = validate_dataframe(tables["placements"], "placements")
    cps = validate_dataframe(tables["counterparties"], "counterparties")
    cash = validate_dataframe(tables["cash"], "cash")
    stmts = validate_dataframe(tables["statements"], "statements")

    # --- PAS ---
    issues += require_columns(pas, "pas", ["policy_id", "program_id", "line_of_business", "region", "peril"])
    issues += non_null(pas, "pas", "policy_id", max_null_rate=0.0, severity="error")
    issues += non_null(pas, "pas", "program_id", max_null_rate=0.0, severity="warning")

    # --- placements ---
    # for Demo 1 we often join to PAS by policy_id; require it for realism
    issues += require_columns(plc, "placements", ["treaty_id", "program_id", "policy_id", "counterparty_id", "layer"])

    # Require either limit_usd OR ceded_limit_usd (people use both)
    issues += require_one_of(plc, "placements", ["limit_usd", "ceded_limit_usd"], label="placement_limit")
    # Premium is optional but recommended
    if "ceded_premium_usd" not in plc.columns:
        issues.append(ValidationIssue(table="placements", severity="warning", message="Missing recommended column: ceded_premium_usd"))

    issues += non_null(plc, "placements", "treaty_id", max_null_rate=0.0, severity="error")
    issues += non_null(plc, "placements", "counterparty_id", max_null_rate=0.0, severity="error")

    # --- counterparties ---
    issues += require_columns(cps, "counterparties", ["counterparty_id", "counterparty_name", "rating"])
    issues += non_null(cps, "counterparties", "counterparty_id", max_null_rate=0.0, severity="error")

    # --- cash ---
    issues += require_columns(cash, "cash", ["cash_id", "counterparty_id", "amount_usd", "txn_date"])
    issues += non_null(cash, "cash", "cash_id", max_null_rate=0.0, severity="error")
    issues += non_null(cash, "cash", "counterparty_id", max_null_rate=0.0, severity="warning")

    # --- statements ---
    issues += require_columns(stmts, "statements", ["statement_id", "counterparty_id", "invoice_date", "amount_due_usd"])
    issues += non_null(stmts, "statements", "statement_id", max_null_rate=0.0, severity="error")

    # --- join coverage governance signals ---
    issues += join_coverage_warning(plc, pas, "placements", "policy_id", "policy_id", "placements → pas (policy_id)")
    issues += join_coverage_warning(plc, cps, "placements", "counterparty_id", "counterparty_id", "placements → counterparties (counterparty_id)")
    issues += join_coverage_warning(cash, cps, "cash", "counterparty_id", "counterparty_id", "cash → counterparties (counterparty_id)")
    issues += join_coverage_warning(stmts, cps, "statements", "counterparty_id", "counterparty_id", "statements → counterparties (counterparty_id)")

    return issues


def summarize_issues(issues: List[ValidationIssue]) -> Dict[str, int]:
    return {
        "errors": sum(1 for i in issues if i.severity == "error"),
        "warnings": sum(1 for i in issues if i.severity == "warning"),
        "total": len(issues),
    }
