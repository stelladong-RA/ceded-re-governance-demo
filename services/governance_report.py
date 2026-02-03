# services/governance_report.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

import hashlib

def _sha1_of_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def _key_field_health(df: pd.DataFrame, fields: list[str]) -> dict:
    out = {}
    for f in fields:
        if f in df.columns:
            out[f] = float(1.0 - df[f].isna().mean())
        else:
            out[f] = None
    return out

def _join_rate(left: pd.DataFrame, right: pd.DataFrame, on: str) -> float | None:
    if left is None or right is None:
        return None
    if on not in left.columns or on not in right.columns:
        return None
    if len(left) == 0:
        return None
    matched = left[on].isin(set(right[on].dropna().unique())).mean()
    return float(matched)


@dataclass
class GovernanceReport:
    created_at_utc: str
    pipeline_version: str
    validation_summary: Dict[str, Any]
    derived_fields: Dict[str, List[str]]
    tables: Dict[str, Any]


def _missingness(df: pd.DataFrame) -> Dict[str, float]:
    if df is None or df.empty:
        return {}
    return {c: float(df[c].isna().mean()) for c in df.columns}


def _table_profile(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None:
        return {"rows": 0, "cols": 0, "columns": [], "missingness": {}}
    return {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "columns": list(df.columns),
        "missingness": _missingness(df),
    }


def write_governance_json(
    root: Path,
    raw_tables: Dict[str, pd.DataFrame],
    canonical_tables: Dict[str, pd.DataFrame],
    processed: Dict[str, pd.DataFrame],
    validation_summary: Dict[str, Any],
    pipeline_version: str = "demo-v1",
) -> Path:
    """
    Writes data/processed/_governance.json describing the demo run.
    """
    
    config_dir = root / "config"
    mappings_hash = _sha1_of_file(config_dir / "mappings.yaml")
    settings_hash = _sha1_of_file(config_dir / "settings.yaml")

    # Key fields for demo-critical completeness
    key_fields = {
        "pas": ["policy_id", "program_id", "line_of_business", "region", "uw_year", "sum_insured_usd"],
        "placements": ["treaty_id", "policy_id", "counterparty_id", "ceded_limit_usd", "ceded_premium_usd"],
        "cash": ["cash_id", "counterparty_id", "payment_date", "amount_usd"],
        "statements": ["statement_id", "counterparty_id", "invoice_date", "amount_due_usd"],
    }
            
    key_fields_health = {}
    for tname, fields in key_fields.items():
        df = canonical_tables.get(tname)
        if df is None:
            df = raw_tables.get(tname)
        if df is not None:
            key_fields_health[tname] = _key_field_health(df, fields)

    # Simple join coverage indicators (edit keys to match your schema)
    join_coverage = {
    "cash_to_counterparties": _join_rate(canonical_tables.get("cash"), canonical_tables.get("counterparties"), "counterparty_id"),
    "statements_to_counterparties": _join_rate(canonical_tables.get("statements"), canonical_tables.get("counterparties"), "counterparty_id"),
    "placements_to_pas": _join_rate(canonical_tables.get("placements"), canonical_tables.get("pas"), "policy_id"),
}

    assumptions_mappings = {
        "settings_yaml_sha1": settings_hash,
        "mappings_yaml_sha1": mappings_hash,
        "derived_field_rules": {
            "cash.cash_id": "md5(treaty_id|counterparty|payment_date|amount_usd|currency) (fallback to row index)",
            "statements.amount_due_usd": "amount_due (else amount_usd, else balance_usd, else null)",
        }
    }

    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # detect derived fields from raw->canonical
    derived_fields: Dict[str, List[str]] = {}
    for name, raw_df in raw_tables.items():
        can_df = canonical_tables.get(name)
        if raw_df is None or can_df is None:
            continue
        raw_cols = set(map(str, raw_df.columns))
        can_cols = set(map(str, can_df.columns))
        added = sorted(list(can_cols - raw_cols))
        if added:
            derived_fields[name] = added

    report = GovernanceReport(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        pipeline_version=pipeline_version,
        validation_summary=validation_summary,
        derived_fields=derived_fields,
        tables={
            "raw": {k: _table_profile(v) for k, v in raw_tables.items()},
            "canonical": {k: _table_profile(v) for k, v in canonical_tables.items()},
            "processed": {k: _table_profile(v) for k, v in processed.items()},
        },
    )

    

    out_path = processed_dir / "_governance.json"

    payload = report.__dict__
    payload["key_fields_health"] = key_fields_health
    payload["join_coverage"] = join_coverage
    payload["assumptions_mappings"] = assumptions_mappings

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return out_path
