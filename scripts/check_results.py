"""Check that an evaluation result file is a complete model × condition matrix.

Usage:
    python scripts/check_results.py --input results/raw_outputs.csv
    python scripts/check_results.py --input results/raw_outputs.csv --models qwen_3b
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "core_problems.json"
CONDITIONS = ["english", "french", "msa", "code_switched_tn"]


def dataset_ids() -> set[str]:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    return {row["id"] for row in data if "_schema" not in row}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/raw_outputs.csv")
    parser.add_argument("--models", default=None, help="Comma-separated expected model keys")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        print(f"ERROR: missing result file: {path}")
        sys.exit(1)

    df = pd.read_csv(path, dtype=str)
    required = {"problem_id", "model", "condition", "model_response", "expected_answer"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: missing columns: {sorted(missing)}")
        sys.exit(1)

    ids = dataset_ids()
    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else sorted(df["model"].dropna().unique())
    expected = {(pid, model, condition) for pid in ids for model in models for condition in CONDITIONS}
    actual = set(zip(df["problem_id"], df["model"], df["condition"]))
    duplicates = df.duplicated(["problem_id", "model", "condition"]).sum()
    missing_rows = expected - actual
    extra_rows = actual - expected
    errors = df["model_response"].fillna("").str.startswith("[GENERATION_ERROR", na=False).sum()

    print(f"Rows: {len(df)} | expected combinations: {len(expected)}")
    print(f"Duplicate combinations: {duplicates}")
    print(f"Generation errors: {errors}")
    print(f"Missing combinations: {len(missing_rows)}")
    print(f"Unexpected combinations: {len(extra_rows)}")
    if missing_rows:
        print("First missing:", sorted(missing_rows)[:10])
    if extra_rows:
        print("First unexpected:", sorted(extra_rows)[:10])

    if duplicates or missing_rows or extra_rows or errors:
        print("STATUS: FAIL — complete the run and resolve generation errors before publication.")
        sys.exit(1)
    print("STATUS: PASS — complete result matrix with no generation errors.")


if __name__ == "__main__":
    main()
