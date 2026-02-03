# utils/loaders.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import pandas as pd
import yaml


# -----------------------------
# Paths
# -----------------------------
@dataclass
class DemoPaths:
    """
    Centralized paths so every module refers to the same folders.
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
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def exports_dir(self) -> Path:
        return self.root / "exports"

    @property
    def config_dir(self) -> Path:
        return self.root / "config"

    def ensure(self) -> "DemoPaths":
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        return self


# -----------------------------
# YAML loaders
# -----------------------------
def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_settings(root: Path | None = None) -> dict:
    """
    Loads config/settings.yaml
    """
    root = root or Path(__file__).resolve().parents[1]
    return _load_yaml(root / "config" / "settings.yaml")


def load_mappings(root: Path | None = None) -> dict:
    """
    Loads config/mappings.yaml
    """
    root = root or Path(__file__).resolve().parents[1]
    return _load_yaml(root / "config" / "mappings.yaml")


# -----------------------------
# Raw table loader
# -----------------------------
def load_raw_tables(paths: DemoPaths) -> Dict[str, pd.DataFrame]:
    """
    Loads the standard demo CSV files from data/raw.
    Required filenames:
      pas.csv, claims.csv, placements.csv, cash.csv, statements.csv, counterparties.csv
    """
    required = {
        "pas": "pas.csv",
        "claims": "claims.csv",
        "placements": "placements.csv",
        "cash": "cash.csv",
        "statements": "statements.csv",
        "counterparties": "counterparties.csv",
    }

    tables: Dict[str, pd.DataFrame] = {}
    for key, fname in required.items():
        fpath = paths.raw_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Missing required raw file: {fpath}")
        tables[key] = pd.read_csv(fpath)

    return tables
