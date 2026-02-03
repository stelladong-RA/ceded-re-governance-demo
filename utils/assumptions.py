"""
utils/assumptions.py

Central place to define and version "demo assumptions" that appear in governance panels and exports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Assumption:
    id: str
    text: str
    scope: str  # e.g. "CEC", "Cash Pairing", "Copilot"
    version: str = "v1"


def default_assumptions() -> List[Assumption]:
    """
    These assumptions are written in simple business language for demo governance.
    """
    return [
        Assumption(
            id="CEC-001",
            scope="CEC",
            text="Ceded exposure is calculated as a simplified basis (e.g., limit or sum insured) for demo purposes; client implementations align to agreed exposure definitions and accounting policies.",
        ),
        Assumption(
            id="CEC-002",
            scope="CEC",
            text="Counterparty ratings and credit limits are synthetic; real deployments ingest rating feeds and internal limit governance.",
        ),
        Assumption(
            id="CASH-001",
            scope="Cash Pairing",
            text="Cash pairing uses a deterministic matching approach (reference match, then proximity match within tolerances); production versions include configurable rules, workflows, and audit trails.",
        ),
        Assumption(
            id="COP-001",
            scope="Copilot",
            text="Copilot narratives are generated from the displayed KPIs; sensitive data controls and client security requirements are applied in production deployments.",
        ),
    ]


def assumptions_text(scope: str | None = None) -> List[str]:
    items = default_assumptions()
    if scope:
        items = [a for a in items if a.scope.lower() == scope.lower()]
    return [f"{a.id} ({a.version}) — {a.text}" for a in items]
