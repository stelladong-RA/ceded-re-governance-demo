# services/pipeline.py
from __future__ import annotations

import pandas as pd
from pathlib import Path

# Ingestion / Canonical
from services.ingestion import DataPaths, load_canonical

# Governance
from services.governance_report import write_governance_json

# Validation
from utils.loaders import DemoPaths, load_raw_tables
from utils.validators import (
    validate_demo_tables,
    summarize_issues,
    validate_dataframe,
)


# ---------------------------------------------------
# Helpers
# ---------------------------------------------------

def _coalesce(df: pd.DataFrame, candidates: list[str], default=None):
    """Return first existing column in candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return default


# ---------------------------------------------------
# Exposure Builder
# ---------------------------------------------------

def build_exposure(tables: dict) -> pd.DataFrame:

    pas = validate_dataframe(tables["pas"].copy(), "pas")
    placements = validate_dataframe(tables["placements"].copy(), "placements")
    claims = validate_dataframe(tables.get("claims", pd.DataFrame()), "claims")

    pas_key = _coalesce(pas, ["policy_id", "program_id"])
    plc_key = _coalesce(placements, ["policy_id", "program_id"])
    clm_key = _coalesce(claims, ["policy_id", "program_id"])

    df = pas.copy()

    # Join claims
    if pas_key and clm_key and pas_key == clm_key:
        df = df.merge(claims, on=pas_key, how="left", suffixes=("", "_claim"))

    # Join placements
    if pas_key and plc_key and pas_key == plc_key:
        df = df.merge(placements, on=pas_key, how="left", suffixes=("", "_plc"))

    gross_col = _coalesce(
        df, ["gross_exposure", "sum_insured_usd", "written_premium_usd"]
    )
    share_col = _coalesce(df, ["share_pct", "ceded_share", "share"])

    # Compute ceded exposure
    if gross_col and share_col:

        if df[share_col].dropna().between(0, 100).all():
            df["ceded_exposure_usd"] = (
                df[gross_col].astype(float)
                * (df[share_col].astype(float) / 100.0)
            )
        else:
            df["ceded_exposure_usd"] = (
                df[gross_col].astype(float) * df[share_col].astype(float)
            )

    else:
        df["ceded_exposure_usd"] = pd.NA

    df["exposure_source_note"] = (
        "Demo exposure: best-effort joins; validate mappings in client pilot"
    )

    return df


# ---------------------------------------------------
# Recoverables Builder
# ---------------------------------------------------

def build_recoverables(tables: dict) -> pd.DataFrame:

    claims = validate_dataframe(tables.get("claims", pd.DataFrame()), "claims")
    placements = validate_dataframe(tables["placements"], "placements")
    cash = validate_dataframe(tables["cash"], "cash")
    statements = validate_dataframe(tables["statements"], "statements")

    df = claims.copy()

    # Join placements
    if "policy_id" in df.columns and "policy_id" in placements.columns:
        df = df.merge(
            placements,
            on="policy_id",
            how="left",
            suffixes=("", "_plc"),
        )

    # Join cash / statements
    if "treaty_id" in df.columns and "treaty_id" in cash.columns:
        df = df.merge(cash, on="treaty_id", how="left", suffixes=("", "_cash"))

    if "treaty_id" in df.columns and "treaty_id" in statements.columns:
        df = df.merge(
            statements,
            on="treaty_id",
            how="left",
            suffixes=("", "_stmt"),
        )

    paid_col = _coalesce(df, ["paid_usd", "paid_amount"])
    share_col = _coalesce(df, ["share_pct", "ceded_share"])
    cash_col = _coalesce(df, ["amount_usd", "cash_amount_usd"])

    # Expected recovery
    if paid_col and share_col:

        if df[share_col].dropna().between(0, 100).all():
            df["expected_recovery_usd"] = (
                df[paid_col] * (df[share_col] / 100.0)
            )
        else:
            df["expected_recovery_usd"] = df[paid_col] * df[share_col]

    else:
        df["expected_recovery_usd"] = pd.NA

    # Cash vs outstanding
    if cash_col:

        df["cash_received_usd"] = df[cash_col]

        df["outstanding_recoverable_usd"] = (
            df["expected_recovery_usd"].fillna(0)
            - df["cash_received_usd"].fillna(0)
        )

    else:
        df["cash_received_usd"] = pd.NA
        df["outstanding_recoverable_usd"] = pd.NA

    df["recoverables_source_note"] = (
        "Demo recoverables: simplified expected vs cash logic"
    )

    return df


# ---------------------------------------------------
# Portfolio Builder
# ---------------------------------------------------

def build_portfolio(
    exposure: pd.DataFrame,
    recoverables: pd.DataFrame,
    tables: dict,
) -> pd.DataFrame:

    counterparties = validate_dataframe(
        tables["counterparties"].copy(), "counterparties"
    )

    cp_id = _coalesce(counterparties, ["counterparty_id"])
    cp_name = _coalesce(counterparties, ["counterparty_name", "name"])

    df = exposure.copy()

    # Join counterparty metadata
    if cp_id and cp_id in df.columns:
        df = df.merge(counterparties, on=cp_id, how="left")

    elif cp_name and cp_name in df.columns:
        df = df.merge(counterparties, on=cp_name, how="left")

    # Aggregate recoverables
    if "outstanding_recoverable_usd" in recoverables.columns:

        rec_key = None

        if cp_id and cp_id in recoverables.columns:
            rec_key = cp_id
        elif cp_name and cp_name in recoverables.columns:
            rec_key = cp_name

        if rec_key:
            rec_agg = (
                recoverables
                .groupby(rec_key)["outstanding_recoverable_usd"]
                .sum()
                .reset_index()
            )

            df = df.merge(rec_agg, on=rec_key, how="left")

    df["kpi_note"] = (
        "Demo KPIs: simplified rollups; production aligns to accounting"
    )

    return df


# ---------------------------------------------------
# Main Pipeline
# ---------------------------------------------------

def run_pipeline() -> None:

    # Project root
    root = Path(__file__).resolve().parents[1]

    # Paths
    demo_paths = DemoPaths(root=root).ensure()
    data_paths = DataPaths(root)

    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------
    # Load
    # -----------------------------------

    print("🔄 Loading raw tables...")
    raw_tables = load_raw_tables(demo_paths)

    print("🔄 Loading canonical tables...")
    canonical_tables = load_canonical(data_paths)

    # -----------------------------------
    # Validate
    # -----------------------------------

    print("🔎 Validating tables...")

    issues = validate_demo_tables(canonical_tables)
    summary = summarize_issues(issues)

    print(f"Validation summary: {summary}")

    if summary["errors"] > 0:
        print("⚠️ Proceeding in demo-safe mode.")

    # -----------------------------------
    # Build Layers
    # -----------------------------------

    print("🔧 Building exposure.parquet ...")
    exposure = build_exposure(canonical_tables)

    print("🔧 Building recoverables.parquet ...")
    recoverables = build_recoverables(canonical_tables)

    print("🔧 Building portfolio.parquet ...")
    portfolio = build_portfolio(
        exposure,
        recoverables,
        canonical_tables,
    )

    # -----------------------------------
    # Save
    # -----------------------------------

    print("💾 Writing parquet files...")

    exposure.to_parquet(out_dir / "exposure.parquet", index=False)
    recoverables.to_parquet(out_dir / "recoverables.parquet", index=False)
    portfolio.to_parquet(out_dir / "portfolio.parquet", index=False)

    print("✅ Data written.")

    # -----------------------------------
    # Governance
    # -----------------------------------

    gov_path = write_governance_json(
        root=root,
        raw_tables=raw_tables,
        canonical_tables=canonical_tables,
        processed={
            "exposure": exposure,
            "recoverables": recoverables,
            "portfolio": portfolio,
        },
        validation_summary=summary,
        pipeline_version="synpulse-demo-v1",
    )

    print(f"🧾 Governance report: {gov_path}")
    print(f"📁 Output: {out_dir}")


# ---------------------------------------------------
# Entrypoint
# ---------------------------------------------------

if __name__ == "__main__":
    run_pipeline()
