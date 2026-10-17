"""
validate_dataset.py

Validate a core-problems JSON file against the expected schema.

Usage:
    python scripts/validate_dataset.py
    python scripts/validate_dataset.py --path data/core_problems_template.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_FIELDS = ["id", "domain", "english", "french", "msa", "code_switched_tn", "expected_answer"]
LANGUAGE_FIELDS = ["english", "french", "msa", "code_switched_tn"]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_entry(entry: dict, seen_ids: set[str]) -> list[str]:
    """Return a list of failure reasons for *entry*. Empty list means PASS."""
    failures: list[str] = []

    # 1. Required fields present and non-empty
    for field in REQUIRED_FIELDS:
        if field not in entry:
            failures.append(f"missing field '{field}'")
        elif entry[field] is None or entry[field] == "":
            failures.append(f"field '{field}' is empty or None")

    # 2. Unique ID check
    entry_id = entry.get("id")
    if entry_id is not None and entry_id != "":
        if entry_id in seen_ids:
            failures.append(f"duplicate id '{entry_id}'")
        seen_ids.add(entry_id)

    return failures


def _check_answer_consistency(entries: list[dict]) -> dict[str, list[str]]:
    """
    For each group of entries sharing the same `id`, verify that all four
    language versions agree on `expected_answer`.

    Returns a mapping of id → list of inconsistency messages.
    Since IDs are supposed to be unique per entry, this mainly guards against
    any accidental duplicate-ID situation where answers diverge.
    """
    by_id: dict[str, list[str]] = defaultdict(list)
    for entry in entries:
        eid = entry.get("id", "")
        ans = entry.get("expected_answer", "")
        by_id[eid].append(ans)

    inconsistencies: dict[str, list[str]] = {}
    for eid, answers in by_id.items():
        unique_answers = set(answers)
        if len(unique_answers) > 1:
            inconsistencies[eid] = [
                f"expected_answer is inconsistent across entries with id='{eid}': {sorted(unique_answers)}"
            ]
    return inconsistencies


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a core-problems JSON file."
    )
    parser.add_argument(
        "--path",
        default=str(PROJECT_ROOT / "data" / "core_problems.json"),
        help="Path to the JSON file to validate (default: data/core_problems.json)",
    )
    args = parser.parse_args()

    json_path = Path(args.path)
    if not json_path.is_absolute():
        json_path = PROJECT_ROOT / json_path

    if not json_path.exists():
        print(f"ERROR: File not found: {json_path}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    try:
        with json_path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        print(f"ERROR: Invalid JSON in {json_path}: {exc}")
        sys.exit(1)

    if not isinstance(data, list):
        print("ERROR: JSON root must be an array.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Strip _schema documentation objects
    # ------------------------------------------------------------------
    problem_entries = [
        obj for obj in data
        if not (isinstance(obj, dict) and list(obj.keys()) == ["_schema"])
    ]

    if not problem_entries:
        print("No problem entries found yet. Template is ready.")
        sys.exit(0)

    print(f"Validating {json_path}")
    print(f"Total entries found (excluding _schema): {len(problem_entries)}\n")

    # ------------------------------------------------------------------
    # Per-entry validation
    # ------------------------------------------------------------------
    seen_ids: set[str] = set()
    entry_failures: dict[int, tuple[str, list[str]]] = {}

    for idx, entry in enumerate(problem_entries):
        entry_id = entry.get("id", f"<entry {idx}>")
        failures = _validate_entry(entry, seen_ids)
        if failures:
            entry_failures[idx] = (entry_id, failures)

    # ------------------------------------------------------------------
    # Cross-entry answer consistency
    # ------------------------------------------------------------------
    inconsistencies = _check_answer_consistency(problem_entries)
    for eid, msgs in inconsistencies.items():
        # Attach inconsistency failures to the relevant entry index
        for idx, entry in enumerate(problem_entries):
            if entry.get("id") == eid:
                if idx in entry_failures:
                    entry_failures[idx][1].extend(msgs)
                else:
                    entry_failures[idx] = (eid, list(msgs))

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    passed: list[str] = []
    failed: list[tuple[str, list[str]]] = []

    for idx, entry in enumerate(problem_entries):
        entry_id = entry.get("id", f"<entry {idx}>")
        if idx in entry_failures:
            failed.append(entry_failures[idx])
        else:
            passed.append(entry_id)

    if passed:
        print("PASSED entries:")
        for eid in passed:
            print(f"  ✓ {eid}")
    else:
        print("PASSED entries: (none)")

    print()

    if failed:
        print("FAILED entries:")
        for eid, reasons in failed:
            print(f"  ✗ {eid}")
            for reason in reasons:
                print(f"      - {reason}")
    else:
        print("FAILED entries: (none)")

    print()

    if failed:
        print(f"Overall status: FAIL  ({len(failed)} of {len(problem_entries)} entries failed)")
        sys.exit(1)
    else:
        print(f"Overall status: PASS  (all {len(problem_entries)} entries valid)")
        sys.exit(0)


if __name__ == "__main__":
    main()
