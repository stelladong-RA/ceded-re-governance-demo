"""
services/llm_service.py

LLM helper utilities for the "Consultant Copilot" demo.

Design goals:
- Always works (deterministic fallback when no API key)
- Safe (no sensitive logging; avoids dumping large tables)
- Explainable (structured outputs + assumptions + optional citations)
- Upgradeable (drop-in wiring for OpenAI/Azure later)

For production: integrate via your platform policies (Azure OpenAI / OpenAI / Anthropic),
add prompt templates, evaluation, guardrails, and audit logging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import os
import math


# =========================================================
# Config
# =========================================================

@dataclass(frozen=True)
class LLMConfig:
    provider: str = "none"  # {"none","openai","azure_openai"}
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_context_rows: int = 12  # cap table rows included in context


# =========================================================
# Provider detection
# =========================================================

def is_llm_enabled(cfg: Optional[LLMConfig] = None) -> bool:
    """
    Returns True if env vars indicate an LLM provider is configured.
    Keeps demo behavior simple and predictable.
    """
    cfg = cfg or LLMConfig()

    if cfg.provider == "azure_openai":
        return bool(os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"))
    if cfg.provider == "openai":
        return bool(os.getenv("OPENAI_API_KEY"))
    # If provider unspecified/none, allow "auto" enable if any key exists
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY"))


# =========================================================
# Small utilities (safe + robust)
# =========================================================

def _safe_pct(x: Any) -> str:
    try:
        v = float(x)
        if math.isnan(v):
            return "N/A"
        return f"{v:.0%}"
    except Exception:
        return "N/A"


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v):
            return default
        return v
    except Exception:
        return default


def _as_records(table: Any, max_rows: int = 12) -> List[Dict[str, Any]]:
    """
    Convert a DataFrame-like or list-like into a list of dicts safely.
    Never returns more than max_rows records.
    """
    if table is None:
        return []

    # Pandas DataFrame
    try:
        import pandas as pd  # type: ignore
        if isinstance(table, pd.DataFrame):
            if len(table) == 0:
                return []
            # keep only first max_rows rows
            return table.head(max_rows).to_dict(orient="records")
    except Exception:
        pass

    # list[dict]
    if isinstance(table, list):
        if not table:
            return []
        if isinstance(table[0], dict):
            return table[:max_rows]

    # dict with "rows"
    if isinstance(table, dict) and "rows" in table and isinstance(table["rows"], list):
        rows = table["rows"]
        if rows and isinstance(rows[0], dict):
            return rows[:max_rows]

    return []


def _summarize_credit_table(credit_table: Any) -> Dict[str, Any]:
    """
    Extract compact, exec-usable signals from utilization table.
    Expected columns (best effort):
    - counterparty_name, rating, credit_limit_usd, utilization_pct, risk_flag
    """
    rows = _as_records(credit_table, max_rows=50)
    if not rows:
        return {
            "available": False,
            "top_utilization": [],
            "high_risk_count": 0,
            "watch_count": 0,
        }

    # Normalize keys we care about
    def get(row, key, default=None):
        return row.get(key, default)

    # Sort by utilization desc if available
    def util_val(r):
        return _safe_float(get(r, "utilization_pct", 0.0), 0.0)

    rows_sorted = sorted(rows, key=util_val, reverse=True)

    top = []
    high = 0
    watch = 0
    for r in rows_sorted[:10]:
        flag = str(get(r, "risk_flag", get(r, "flag", "")) or "")
        if "High" in flag:
            high += 1
        if "Watch" in flag:
            watch += 1
        top.append({
            "counterparty_name": get(r, "counterparty_name", "Unknown"),
            "rating": get(r, "rating", "N/A"),
            "utilization_pct": _safe_float(get(r, "utilization_pct", 0.0), 0.0),
            "credit_limit_usd": _safe_float(get(r, "credit_limit_usd", 0.0), 0.0),
            "risk_flag": flag or "Normal",
        })

    return {
        "available": True,
        "top_utilization": top,
        "high_risk_count": high,
        "watch_count": watch,
    }


def _summarize_cash_kpis(cash_recon_kpis: Dict[str, Any]) -> Dict[str, Any]:
    matched = int(_safe_float(cash_recon_kpis.get("matched", 0), 0))
    unmatched = int(_safe_float(cash_recon_kpis.get("unmatched", 0), 0))
    variance_count = int(_safe_float(cash_recon_kpis.get("variance_count", 0), 0))
    avg_conf = _safe_float(cash_recon_kpis.get("avg_confidence", 0.0), 0.0)

    total = matched + unmatched
    match_rate = (matched / total) if total > 0 else 0.0

    return {
        "matched": matched,
        "unmatched": unmatched,
        "variance_count": variance_count,
        "avg_confidence": avg_conf,
        "match_rate": match_rate,
    }


# =========================================================
# Context builder
# =========================================================

def build_copilot_context(
    exposure_kpis: Dict[str, Any],
    credit_table: Any,
    cash_recon_kpis: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a concise, structured context packet for the copilot.

    This is intentionally "LLM-friendly" but also supports deterministic fallback.
    """
    credit_summary = _summarize_credit_table(credit_table)
    cash_summary = _summarize_cash_kpis(cash_recon_kpis)

    rows_total = int(_safe_float(exposure_kpis.get("rows_total", 0), 0))
    placement_match_rate = _safe_float(exposure_kpis.get("placement_match_rate", 0.0), 0.0)
    missing_cp_rate = _safe_float(exposure_kpis.get("missing_counterparty_rate", 0.0), 0.0)

    # Build highlights grounded in metrics (no fluff)
    highlights: List[str] = []
    highlights.append(f"Unified exposure layer prepared ({rows_total:,} rows).")
    highlights.append(f"Placement match rate: {_safe_pct(placement_match_rate)}.")
    highlights.append(f"Missing counterparty enrichment: {_safe_pct(missing_cp_rate)}.")

    if credit_summary["available"]:
        # Show top 1–2 counterparties by utilization as a hook
        top = credit_summary["top_utilization"][:2]
        if top:
            highlights.append(
                "Top credit utilization: "
                + ", ".join([f"{t['counterparty_name']} ({t['utilization_pct']:.0%})" for t in top])
                + "."
            )

    highlights.append(f"Cash auto-match rate: {_safe_pct(cash_summary['match_rate'])} (avg confidence {cash_summary['avg_confidence']:.2f}).")

    # Risks (tie to governance)
    risks: List[str] = []
    if missing_cp_rate > 0.05:
        risks.append("Counterparty master-data gaps may weaken governance reporting and audit defensibility.")
    if credit_summary["high_risk_count"] > 0:
        risks.append("At least one counterparty appears in a high-utilization band and may require credit committee review.")
    if cash_summary["unmatched"] > 0:
        risks.append("Unmatched cash items create operational leakage and slower close cycles.")
    if cash_summary["variance_count"] > 0:
        risks.append("Material variances suggest reference quality issues, partial payments, or timing mismatches.")

    # Recommended actions (consulting-forward)
    actions: List[str] = [
        "Confirm client data sourcing scope (PAS, placements, finance/cash, statements) and agree canonical field mapping.",
        "Run a short proof-of-value (2–4 weeks) on one ceded program to validate join coverage, controls, and reporting outputs.",
        "Define governance thresholds (match rates, confidence cutoffs, utilization flags) aligned to client policy and audit needs.",
    ]

    # Assumptions + citations (useful for audit narrative)
    assumptions: List[str] = [
        "Synthetic demo data is illustrative; production values will align to client accounting and credit policy.",
        "Credit limits shown are demo-calculated and should be replaced by client-approved limits and ratings sources.",
        "Cash pairing uses rules-based proximity matching; production workflows should include user feedback loops and exceptions SLAs.",
    ]

    citations: List[str] = [
        "Demo dataset (synthetic) — for illustration only",
        "Governance report (_governance.json) — lineage and validation status",
    ]

    return {
        "highlights": highlights,
        "risks": risks,
        "actions": actions,
        "assumptions": assumptions,
        "citations": citations,
        "metrics": {
            "exposure_kpis": {
                "rows_total": rows_total,
                "placement_match_rate": placement_match_rate,
                "missing_counterparty_rate": missing_cp_rate,
            },
            "cash_recon": cash_summary,
            "credit_summary": credit_summary,
        },
        # Compact table payload only (avoid dumping large frames)
        "tables": {
            "credit_top_utilization": credit_summary.get("top_utilization", [])[:10],
        },
    }


# =========================================================
# Narrative generation (fallback + future LLM wiring)
# =========================================================

def draft_exec_summary(
    question: str,
    context: Dict[str, Any],
    cfg: LLMConfig = LLMConfig(),
) -> Dict[str, Any]:
    """
    Returns an executive-ready narrative.

    If no LLM is configured, returns a deterministic, board-friendly summary
    so the demo always works.
    """
    enabled = is_llm_enabled(cfg)

    # Always produce something high quality, even without LLM
    if (not enabled) or cfg.provider == "none":
        return _deterministic_exec_summary(question=question, context=context)

    # If you later wire an SDK, keep this contract unchanged:
    # return {"mode": "...", "question": question, "summary": "...", "citations": [...]}
    return {
        "mode": "configured_but_not_implemented",
        "question": question,
        "summary": (
            "LLM is enabled, but provider integration is intentionally not wired in this demo.\n\n"
            "Recommendation: keep the deterministic mode for the meeting, and only enable an LLM once "
            "data policies, redaction, and audit logging are in place."
        ),
        "citations": context.get("citations", []),
    }


def _deterministic_exec_summary(question: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic executive narrative (professional tone).
    """
    highlights: List[str] = context.get("highlights", [])
    risks: List[str] = context.get("risks", [])
    actions: List[str] = context.get("actions", [])
    assumptions: List[str] = context.get("assumptions", [])
    metrics: Dict[str, Any] = context.get("metrics", {})

    exposure = metrics.get("exposure_kpis", {})
    cash = metrics.get("cash_recon", {})
    credit_summary = metrics.get("credit_summary", {})

    lines: List[str] = []

    # Title / intent
    lines.append("Executive Summary (demo mode — deterministic, no external LLM)")
    lines.append("")
    lines.append(f"Prompt: {question}")
    lines.append("")

    # What this is
    lines.append("What this shows")
    lines.append(
        "- A governed, end-to-end view that connects ceded exposure + counterparty credit + cash reconciliation into a single control narrative."
    )
    lines.append("")

    # Key metrics (numbers first)
    lines.append("Key metrics observed")
    lines.append(f"- Exposure records in unified layer: {int(_safe_float(exposure.get('rows_total', 0), 0)):,}")
    lines.append(f"- Placement match rate: {_safe_pct(exposure.get('placement_match_rate', 0.0))}")
    lines.append(f"- Missing counterparty enrichment: {_safe_pct(exposure.get('missing_counterparty_rate', 0.0))}")
    lines.append(f"- Cash matched: {int(_safe_float(cash.get('matched', 0), 0)):,} / Unmatched: {int(_safe_float(cash.get('unmatched', 0), 0)):,}")
    lines.append(f"- Cash match rate: {_safe_pct(cash.get('match_rate', 0.0))} | Avg confidence: {_safe_float(cash.get('avg_confidence', 0.0), 0.0):.2f}")
    if credit_summary.get("available"):
        lines.append(f"- Credit signals: {credit_summary.get('high_risk_count', 0)} high-risk, {credit_summary.get('watch_count', 0)} watchlist (top utilization sample).")
    lines.append("")

    # Highlights
    lines.append("Highlights")
    if highlights:
        for h in highlights:
            lines.append(f"- {h}")
    else:
        lines.append("- N/A")
    lines.append("")

    # Risks
    lines.append("Governance risks / gaps")
    if risks:
        for r in risks:
            lines.append(f"- {r}")
    else:
        lines.append("- No material governance risks flagged by current thresholds.")
    lines.append("")

    # Recommended actions
    lines.append("Recommended next steps")
    if actions:
        for a in actions:
            lines.append(f"- {a}")
    else:
        lines.append("- Confirm scope and proceed with proof-of-value.")
    lines.append("")

    # Questions (good for the meeting)
    lines.append("Suggested questions to confirm with the client / Synpulse team")
    lines.append("- What is the authoritative source for counterparty limits and ratings, and how often are they refreshed?")
    lines.append("- What match confidence threshold is acceptable for auto-clear vs exception workflow (audit policy)?")
    lines.append("- Which programs / LoBs are the priority for a proof-of-value, and what success metrics matter (time-to-close, match rate, leakage)?")
    lines.append("")

    # Assumptions & citations
    lines.append("Assumptions")
    for a in assumptions[:5]:
        lines.append(f"- {a}")
    lines.append("")

    citations = context.get("citations", [])
    if citations:
        lines.append("Citations / references (demo)")
        for c in citations[:10]:
            lines.append(f"- {c}")

    return {
        "mode": "fallback",
        "question": question,
        "summary": "\n".join(lines),
        "citations": citations,
    }
