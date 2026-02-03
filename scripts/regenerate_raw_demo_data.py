# scripts/regenerate_raw_demo_data.py
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Config
# ============================================================

@dataclass
class GenConfig:
    seed: int = 42

    # Sizes (tune freely)
    n_policies: int = 260
    n_counterparties: int = 14

    # Placements per policy
    min_layers: int = 2
    max_layers: int = 4

    # Claims frequency
    claim_rate: float = 0.35  # proportion of policies with at least one claim
    max_claims_per_policy: int = 2

    # Statements & cash
    n_statement_months: int = 10
    cash_exact_ref_rate: float = 0.65      # cash rows referencing statement_id
    cash_candidate_rate: float = 0.25      # no reference, but matchable by amount/date
    cash_unmatched_rate: float = 0.10      # intentionally unmatched for exception queue

    # Time range
    start_date: str = "2025-01-01"  # earliest inception/invoice
    end_date: str = "2026-01-31"    # latest invoice/payment


# ============================================================
# Helpers
# ============================================================

def _rng(cfg: GenConfig) -> np.random.Generator:
    return np.random.default_rng(cfg.seed)

def _root() -> Path:
    return Path(__file__).resolve().parents[1]

def _raw_dir() -> Path:
    d = _root() / "data" / "raw"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s)

def _random_date(r: np.random.Generator, a: datetime, b: datetime) -> datetime:
    if b <= a:
        return a
    delta = (b - a).days
    return a + timedelta(days=int(r.integers(0, delta + 1)))

def _id(prefix: str, i: int, width: int = 6) -> str:
    return f"{prefix}{i:0{width}d}"

def _pick_weighted(r: np.random.Generator, items: List[str], weights: List[float]) -> str:
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    return items[int(r.choice(len(items), p=w))]

def _clip(x, lo=None, hi=None):
    if lo is not None:
        x = max(lo, x)
    if hi is not None:
        x = min(hi, x)
    return x

def _fmt_money(x: float) -> float:
    # store as numeric in CSV; format on UI
    return float(round(x, 2))

def _ensure_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# Generators
# ============================================================

def gen_counterparties(cfg: GenConfig) -> pd.DataFrame:
    r = _rng(cfg)

    # Realistic reinsurer-style names (synthetic)
    base_names = [
        "Ariel Re", "PartnerRe", "SCOR", "Swiss Re", "Munich Re", "Hannover Re",
        "Gen Re", "RenaissanceRe", "SiriusPoint", "Everest Re", "Axis Re",
        "Chubb Tempest Re", "MS Reinsurance", "Conduit Re", "Validus Re", "Aspen Re"
    ]
    r.shuffle(base_names)
    names = base_names[: cfg.n_counterparties]

    ratings = ["AAA", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BB+"]
    rating_weights = [0.05, 0.12, 0.10, 0.16, 0.20, 0.16, 0.12, 0.07, 0.02]

    rows = []
    for i, nm in enumerate(names, start=1):
        cp_id = _id("CP", i, width=3)
        rating = _pick_weighted(r, ratings, rating_weights)

        # base_limit tends to be higher for better ratings (not strictly)
        base_limit = float(r.normal(300_000_000, 90_000_000))
        base_limit = _clip(base_limit, 120_000_000, 650_000_000)

        region = _pick_weighted(r, ["US", "EU", "UK", "APAC"], [0.35, 0.25, 0.15, 0.25])

        rows.append({
            "counterparty_id": cp_id,
            "counterparty_name": nm,
            "rating": rating,
            "base_limit_usd": int(base_limit),
            "home_region": region,
        })

    return pd.DataFrame(rows)


def gen_pas(cfg: GenConfig) -> pd.DataFrame:
    r = _rng(cfg)
    start = _dt(cfg.start_date)
    end = _dt(cfg.end_date)

    lobs = ["Property Cat", "Property", "Casualty", "Specialty"]
    lob_w = [0.40, 0.30, 0.20, 0.10]

    perils = ["Hurricane", "Earthquake", "Wildfire", "Flood", "Severe Convective Storm"]
    peril_w = [0.25, 0.18, 0.22, 0.20, 0.15]

    regions = ["US", "EU", "UK", "APAC", "Canada", "LATAM"]
    region_w = [0.42, 0.20, 0.10, 0.12, 0.08, 0.08]

    # program pools so many policies share a program_id
    n_programs = max(18, int(cfg.n_policies / 10))
    program_ids = [_id("PRG", i, width=4) for i in range(1, n_programs + 1)]

    rows = []
    for i in range(1, cfg.n_policies + 1):
        policy_id = _id("POL", i, width=6)
        program_id = str(r.choice(program_ids))

        lob = _pick_weighted(r, lobs, lob_w)
        peril = _pick_weighted(r, perils, peril_w)
        region = _pick_weighted(r, regions, region_w)

        inception = _random_date(r, start, end - timedelta(days=365))
        expiry = inception + timedelta(days=365)

        # exposures: skewed distribution
        sum_insured = float(r.lognormal(mean=math.log(12_000_000), sigma=0.8))
        sum_insured = _clip(sum_insured, 2_000_000, 250_000_000)

        written_premium = sum_insured * float(r.uniform(0.006, 0.03))
        limit_usd = sum_insured * float(r.uniform(0.20, 0.80))
        retention = limit_usd * float(r.uniform(0.03, 0.15))

        uw_year = inception.year
        rows.append({
            "policy_id": policy_id,
            "program_id": program_id,
            "insured_name": f"Insured {_id('C', i, width=5)}",
            "line_of_business": lob,
            "region": region,
            "state_province": _pick_weighted(r, ["CA","FL","TX","NY","IL","WA","NA"], [0.13,0.12,0.12,0.10,0.09,0.08,0.36]),
            "inception_date": inception.date().isoformat(),
            "expiry_date": expiry.date().isoformat(),
            "peril": peril,
            "sum_insured_usd": int(sum_insured),
            "written_premium_usd": int(written_premium),
            "limit_usd": int(limit_usd),
            "retention_usd": int(retention),
            "currency": "USD",
            "uw_year": int(uw_year),
            "exposure_metric": _pick_weighted(r, ["TIV","Premium","Limit"], [0.55,0.25,0.20]),
        })

    return pd.DataFrame(rows)


def gen_placements(cfg: GenConfig, pas: pd.DataFrame, cps: pd.DataFrame) -> pd.DataFrame:
    r = _rng(cfg)

    cp_ids = cps["counterparty_id"].astype(str).tolist()

    rows = []
    tid = 1

    # For realism, assign treaty per program and layers per treaty.
    # But keep row-level policy_id to guarantee high join coverage.
    for _, p in pas.iterrows():
        policy_id = str(p["policy_id"])
        program_id = str(p["program_id"])
        lob = str(p["line_of_business"])
        region = str(p["region"])

        n_layers = int(r.integers(cfg.min_layers, cfg.max_layers + 1))

        # Create a treaty umbrella per policy (demo simplification)
        treaty_id = _id("TR", tid, width=5)
        tid += 1

        # Choose counterparties (avoid repeats)
        chosen = list(r.choice(cp_ids, size=n_layers, replace=False))

        # limit anchor based on policy limit
        policy_limit = float(p.get("limit_usd", 20_000_000))
        base_limit = policy_limit * float(r.uniform(0.7, 1.3))
        base_limit = _clip(base_limit, 3_000_000, 180_000_000)

        # cession % by layer - lower layers often higher share, keep within 5%–45%
        cessions = sorted([float(r.uniform(0.08, 0.40)) for _ in range(n_layers)], reverse=True)

        for layer_idx in range(1, n_layers + 1):
            counterparty_id = str(chosen[layer_idx - 1])

            layer_limit = base_limit * float(r.uniform(0.6, 1.1))
            layer_limit = _clip(layer_limit, 2_000_000, 250_000_000)

            cession_pct = cessions[layer_idx - 1] * 100.0  # store as percent for readability
            ceded_limit = layer_limit * (cession_pct / 100.0)

            # premium rate depends on lob/peril loosely
            rate = float(r.uniform(0.02, 0.06))
            if "Cat" in lob:
                rate *= float(r.uniform(1.1, 1.5))
            if region in ["US", "Canada"] and str(p["peril"]) in ["Hurricane", "Wildfire"]:
                rate *= float(r.uniform(1.1, 1.4))

            ceded_premium = ceded_limit * rate

            rows.append({
                # Keys for joins
                "treaty_id": treaty_id,
                "program_id": program_id,
                "policy_id": policy_id,
                "counterparty_id": counterparty_id,

                # Placement attributes
                "layer": int(layer_idx),
                "limit_usd": _fmt_money(layer_limit),
                "ceded_limit_usd": _fmt_money(ceded_limit),
                "ceded_premium_usd": _fmt_money(ceded_premium),
                "cession_pct": _fmt_money(cession_pct),

                # Helpful narrative fields (optional)
                "treaty_type": _pick_weighted(r, ["XoL", "QS", "Surplus"], [0.55, 0.35, 0.10]),
                "broker": _pick_weighted(r, ["Aon", "Guy Carpenter", "Marsh", "Willis", "Direct"], [0.22,0.20,0.18,0.16,0.24]),
                "placement_status": _pick_weighted(r, ["Bound", "Signed", "Quoted"], [0.70, 0.20, 0.10]),
            })

    df = pd.DataFrame(rows)
    return df


def gen_claims(cfg: GenConfig, pas: pd.DataFrame) -> pd.DataFrame:
    r = _rng(cfg)
    start = _dt(cfg.start_date)
    end = _dt(cfg.end_date)

    rows = []
    cid = 1

    for _, p in pas.iterrows():
        if float(r.random()) > cfg.claim_rate:
            continue

        n_claims = int(r.integers(1, cfg.max_claims_per_policy + 1))
        for _ in range(n_claims):
            policy_id = str(p["policy_id"])
            program_id = str(p["program_id"])
            loss_date = _random_date(r, start, end)

            # claim severity: skewed
            paid = float(r.lognormal(mean=math.log(250_000), sigma=1.0))
            paid = _clip(paid, 25_000, 35_000_000)

            rows.append({
                "claim_id": _id("CLM", cid, width=6),
                "policy_id": policy_id,
                "program_id": program_id,
                "loss_date": loss_date.date().isoformat(),
                "paid_amount_usd": _fmt_money(paid),
                "claim_status": _pick_weighted(r, ["Open", "Closed", "Reopened"], [0.35, 0.60, 0.05]),
                "peril": str(p["peril"]),
                "region": str(p["region"]),
                "line_of_business": str(p["line_of_business"]),
            })
            cid += 1

    return pd.DataFrame(rows)


def gen_statements(cfg: GenConfig, cps: pd.DataFrame, placements: pd.DataFrame) -> pd.DataFrame:
    r = _rng(cfg)
    start = _dt(cfg.start_date)
    end = _dt(cfg.end_date)

    # Use premiums to drive statement sizes per counterparty
    prem_by_cp = (
        placements.groupby("counterparty_id", dropna=False)["ceded_premium_usd"]
        .sum()
        .reset_index()
    )
    prem_map = {str(k): float(v) for k, v in zip(prem_by_cp["counterparty_id"], prem_by_cp["ceded_premium_usd"])}

    # Create monthly-ish statements
    months = cfg.n_statement_months
    rows = []
    sid = 1

    cp_ids = cps["counterparty_id"].astype(str).tolist()
    for cp in cp_ids:
        base = prem_map.get(cp, float(r.uniform(2_000_000, 8_000_000)))
        # split base across months
        for m in range(months):
            inv_date = _random_date(r, start, end)
            due_date = inv_date + timedelta(days=int(r.integers(15, 45)))

            amt = base / months * float(r.uniform(0.75, 1.35))
            amt = _clip(amt, 50_000, 8_000_000)

            # For cash_engine compatibility, keep both amount_due_usd and net_due_usd
            rows.append({
                "statement_id": _id("SOA-", sid, width=6),
                "counterparty_id": cp,
                "invoice_date": inv_date.date().isoformat(),
                "due_date": due_date.date().isoformat(),
                "amount_due_usd": _fmt_money(amt),
                "net_due_usd": _fmt_money(amt),
                "currency": "USD",
                "statement_type": _pick_weighted(r, ["Premium", "Adjustment", "Claim"], [0.72, 0.18, 0.10]),
                "notes": "",
            })
            sid += 1

    return pd.DataFrame(rows)


def gen_cash(cfg: GenConfig, cps: pd.DataFrame, statements: pd.DataFrame) -> pd.DataFrame:
    r = _rng(cfg)
    start = _dt(cfg.start_date)
    end = _dt(cfg.end_date)

    # Ratios
    exact_n = int(len(statements) * cfg.cash_exact_ref_rate)
    candidate_n = int(len(statements) * cfg.cash_candidate_rate)
    unmatched_n = int(len(statements) * cfg.cash_unmatched_rate)
    total_n = max(1, exact_n + candidate_n + unmatched_n)

    # Sample statements for exact reference matches
    stmt_sample = statements.sample(n=min(exact_n, len(statements)), random_state=cfg.seed)
    stmt_ids = stmt_sample["statement_id"].astype(str).tolist()

    rows = []
    cash_i = 1

    # --- Exact reference matches ---
    for _, s in stmt_sample.iterrows():
        cp = str(s["counterparty_id"])
        invoice_date = pd.to_datetime(s["invoice_date"])
        due = pd.to_datetime(s["due_date"])
        txn_date = invoice_date + timedelta(days=int(r.integers(1, 25)))
        payment_date = txn_date + timedelta(days=int(r.integers(0, 6)))

        # amount close to due with small variance
        due_amt = float(s["net_due_usd"])
        amt = due_amt + float(r.normal(0, due_amt * 0.01))
        amt = _clip(amt, 1_000, 20_000_000)

        rows.append({
            "cash_id": _id("CASH", cash_i, width=9),
            "cash_txn_id": _id("CASH", cash_i, width=9),  # some code expects cash_txn_id
            "counterparty_id": cp,
            "txn_date": txn_date.date().isoformat(),
            "payment_date": payment_date.date().isoformat(),
            "amount_usd": _fmt_money(amt),
            "direction": "Inbound",
            "bank_reference": f"SOA-{str(s['statement_id'])}",
            "statement_id": str(s["statement_id"]),
            "notes": "Exact reference payment",
        })
        cash_i += 1

    # --- Candidate matches (no reference, but matchable by counterparty + amount/date) ---
    cand_pool = statements.sample(n=min(candidate_n, len(statements)), random_state=cfg.seed + 7)
    for _, s in cand_pool.iterrows():
        cp = str(s["counterparty_id"])
        invoice_date = pd.to_datetime(s["invoice_date"])
        txn_date = invoice_date + timedelta(days=int(r.integers(1, 35)))
        payment_date = txn_date + timedelta(days=int(r.integers(0, 5)))

        due_amt = float(s["net_due_usd"])
        amt = due_amt + float(r.normal(0, due_amt * 0.02))
        amt = _clip(amt, 1_000, 20_000_000)

        rows.append({
            "cash_id": _id("CASH", cash_i, width=9),
            "cash_txn_id": _id("CASH", cash_i, width=9),
            "counterparty_id": cp,
            "txn_date": txn_date.date().isoformat(),
            "payment_date": payment_date.date().isoformat(),
            "amount_usd": _fmt_money(amt),
            "direction": "Inbound",
            "bank_reference": f"PAY-{int(r.integers(100000, 999999))}",
            "statement_id": "",  # intentionally blank -> candidate matching
            "notes": "No reference; should match by proximity",
        })
        cash_i += 1

    # --- Unmatched cash (exceptions) ---
    cp_ids = cps["counterparty_id"].astype(str).tolist()
    for _ in range(unmatched_n):
        cp = str(r.choice(cp_ids))
        txn_date = _random_date(r, start, end)
        payment_date = txn_date + timedelta(days=int(r.integers(0, 7)))

        amt = float(r.uniform(5_000, 250_000))  # small random amounts unlikely to match big statements
        rows.append({
            "cash_id": _id("CASH", cash_i, width=9),
            "cash_txn_id": _id("CASH", cash_i, width=9),
            "counterparty_id": cp,
            "txn_date": txn_date.date().isoformat(),
            "payment_date": payment_date.date().isoformat(),
            "amount_usd": _fmt_money(amt),
            "direction": "Inbound",
            "bank_reference": f"MISC-{int(r.integers(100000, 999999))}",
            "statement_id": "",
            "notes": "Intentionally unmatched (exception queue)",
        })
        cash_i += 1

    df = pd.DataFrame(rows)

    # Ensure required columns exist even if empty in some rows
    for col in ["statement_id", "notes", "direction", "bank_reference", "payment_date"]:
        if col not in df.columns:
            df[col] = ""

    return df


# ============================================================
# Main
# ============================================================

def main():
    cfg = GenConfig()
    out_dir = _raw_dir()

    print(f"📁 Writing demo raw CSVs to: {out_dir}")
    print(f"🎲 Seed: {cfg.seed}")

    cps = gen_counterparties(cfg)
    pas = gen_pas(cfg)
    placements = gen_placements(cfg, pas=pas, cps=cps)
    claims = gen_claims(cfg, pas=pas)
    statements = gen_statements(cfg, cps=cps, placements=placements)
    cash = gen_cash(cfg, cps=cps, statements=statements)

    # Sort for nicer diffs
    cps = cps.sort_values("counterparty_id")
    pas = pas.sort_values("policy_id")
    placements = placements.sort_values(["treaty_id", "layer"])
    claims = claims.sort_values(["policy_id", "claim_id"])
    statements = statements.sort_values(["counterparty_id", "invoice_date"])
    cash = cash.sort_values(["counterparty_id", "txn_date"])

    # Write
    cps.to_csv(out_dir / "counterparties.csv", index=False)
    pas.to_csv(out_dir / "pas.csv", index=False)
    placements.to_csv(out_dir / "placements.csv", index=False)
    claims.to_csv(out_dir / "claims.csv", index=False)
    statements.to_csv(out_dir / "statements.csv", index=False)
    cash.to_csv(out_dir / "cash.csv", index=False)

    # Quick sanity prints
    print("✅ Done.")
    print(f"counterparties: {len(cps):,}")
    print(f"pas:           {len(pas):,}")
    print(f"placements:    {len(placements):,}")
    print(f"claims:        {len(claims):,}")
    print(f"statements:    {len(statements):,}")
    print(f"cash:          {len(cash):,}")

    # Join coverage checks (what your governance warnings are about)
    pas_policy = set(pas["policy_id"].astype(str))
    plc_policy = placements["policy_id"].astype(str)
    pol_match = float(plc_policy.isin(pas_policy).mean()) if len(plc_policy) else 1.0

    cp_set = set(cps["counterparty_id"].astype(str))
    plc_cp_match = float(placements["counterparty_id"].astype(str).isin(cp_set).mean()) if len(placements) else 1.0
    cash_cp_match = float(cash["counterparty_id"].astype(str).isin(cp_set).mean()) if len(cash) else 1.0
    stmt_cp_match = float(statements["counterparty_id"].astype(str).isin(cp_set).mean()) if len(statements) else 1.0

    print("\n🔎 Expected join coverage:")
    print(f"placements → pas (policy_id):         {pol_match:.0%}")
    print(f"placements → counterparties:          {plc_cp_match:.0%}")
    print(f"cash → counterparties:                {cash_cp_match:.0%}")
    print(f"statements → counterparties:          {stmt_cp_match:.0%}")


if __name__ == "__main__":
    main()
