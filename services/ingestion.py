# services/ingestion.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd
import hashlib


@dataclass(frozen=True)
class DataPaths:
    """
    Resolves standard demo paths relative to a repo root.
    Expected structure:
      <root>/data/raw/*.csv
      <root>/data/processed/*.parquet   (optional for UI)
      <root>/config/*.yaml              (optional)
    """
    root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def config_dir(self) -> Path:
        return self.root / "config"


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _derive_cash_id(df: pd.DataFrame) -> pd.DataFrame:
    # Required for cash pairing demo
    if df.empty:
        return df
    if "cash_id" in df.columns:
        return df

    # Create a stable id based on key fields (fallback to index)
    keys = []
    for k in ["treaty_id", "counterparty_id", "payment_date", "amount_usd", "currency"]:
        if k in df.columns:
            keys.append(k)

    if not keys:
        df["cash_id"] = [f"cash_{i}" for i in range(len(df))]
        return df

    def mk(row):
        raw = "|".join(str(row.get(k, "")) for k in keys)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]

    df["cash_id"] = df.apply(mk, axis=1)
    return df


def _derive_amount_due_usd(df: pd.DataFrame) -> pd.DataFrame:
    # Required for statements / reconciliation demo
    if df.empty:
        return df
    if "amount_due_usd" in df.columns:
        return df

    # Prefer amount_due if present, else amount_usd, else balance_usd
    if "amount_due" in df.columns:
        df["amount_due_usd"] = df["amount_due"]
    elif "amount_usd" in df.columns:
        df["amount_due_usd"] = df["amount_usd"]
    elif "balance_usd" in df.columns:
        df["amount_due_usd"] = df["balance_usd"]
    else:
        df["amount_due_usd"] = pd.NA
    return df


def load_canonical(paths: DataPaths) -> Dict[str, pd.DataFrame]:
    """
    Loads demo raw CSVs and applies lightweight canonicalization needed
    by Streamlit pages. (Not meant to be production ETL.)
    """
    raw = {
        "pas": _safe_read_csv(paths.raw_dir / "pas.csv"),
        "claims": _safe_read_csv(paths.raw_dir / "claims.csv"),
        "placements": _safe_read_csv(paths.raw_dir / "placements.csv"),
        "cash": _safe_read_csv(paths.raw_dir / "cash.csv"),
        "statements": _safe_read_csv(paths.raw_dir / "statements.csv"),
        "counterparties": _safe_read_csv(paths.raw_dir / "counterparties.csv"),
    }

    # minimal derived fields used by later demos
    raw["cash"] = _derive_cash_id(raw["cash"])
    raw["statements"] = _derive_amount_due_usd(raw["statements"])

    return raw
