# scripts/check_raw.py
from __future__ import annotations

from pathlib import Path
import os

from utils.loaders import DemoPaths, load_raw_tables
from utils.validators import validate_demo_tables, summarize_issues


def main() -> None:
    """
    Validates raw demo CSVs using your current validators and prints issues.
    This script intentionally does NOT depend on services.ingestion.canonicalize_all
    (which no longer exists).
    """

    # Project root (repo root)
    root = Path(__file__).resolve().parents[1]

    # Allow override (optional)
    demo_root = Path(os.getenv("DEMO_ROOT", str(root)))

    paths = DemoPaths(root=demo_root).ensure()

    print("🔄 Loading raw tables...")
    raw_tables = load_raw_tables(paths)

    print("🔎 Validating raw tables...")
    issues = validate_demo_tables(raw_tables)
    summary = summarize_issues(issues)

    if summary["total"] == 0:
        print("RAW VALIDATION ISSUES:")
        print("✅ none")
        return

    print("RAW VALIDATION ISSUES:")
    for issue in issues:
        # issue likely a dataclass with fields: table, severity, message
        print(f"- {issue}")

    print(f"\nSummary: {summary}")


if __name__ == "__main__":
    main()
