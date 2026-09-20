"""
inspect_controls.py
-------------------
Prints a diagnostic report for the three control benchmark CSVs produced by
fetch_controls.py.

Usage:
    python scripts/inspect_controls.py
"""

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# File registry
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_DIR = PROJECT_ROOT / "data" / "control_benchmarks"

# (filename, question column, answer column)
FILES: list[tuple[str, str, str]] = [
    ("codemixbench_mmlu_hineng.csv", "sentence", "answer"),
    ("codemixbench_mmlu_spaeng.csv", "sentence", "answer"),
    ("codemixqa.csv",                "problem",  "answer"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIVIDER = "=" * 70
THIN    = "─" * 70


def null_or_empty_counts(df: pd.DataFrame) -> pd.Series:
    """For object/string columns, count rows that are null OR empty string."""
    counts = {}
    for col in df.select_dtypes(include="object").columns:
        counts[col] = df[col].isna().sum() + (df[col] == "").sum()
    return pd.Series(counts, dtype=int)


def inspect_file(path: Path, question_col: str, answer_col: str) -> dict | None:
    """
    Print diagnostics for one CSV and return a summary dict, or None if the
    file does not exist.
    """
    rel = path.relative_to(PROJECT_ROOT)
    print(f"\n{DIVIDER}")
    print(f"  FILE: {rel}")
    print(DIVIDER)

    if not path.exists():
        print(f"  [WARNING] File not found — run fetch_controls.py first.\n")
        return None

    df = pd.read_csv(path)
    n_rows, n_cols = df.shape

    # 1. Shape ---------------------------------------------------------------
    print(f"\n[1] Shape")
    print(f"    Rows    : {n_rows:,}")
    print(f"    Columns : {n_cols}")

    # 2. Column names + dtypes -----------------------------------------------
    print(f"\n[2] Columns & dtypes")
    col_info = df.dtypes.rename("dtype").reset_index().rename(columns={"index": "column"})
    for _, row in col_info.iterrows():
        print(f"    {row['column']:<30} {str(row['dtype'])}")

    # 3. First 3 rows (transposed) -------------------------------------------
    print(f"\n[3] First 3 rows (transposed — columns as rows)")
    print(THIN)
    preview = df.head(3).T.to_string()
    # indent each line for readability
    for line in preview.splitlines():
        print(f"    {line}")
    print(THIN)

    # 4. Null / empty counts for string columns ------------------------------
    empty = null_or_empty_counts(df)
    if empty.empty:
        print("\n[4] Null/empty counts  — no object/string columns found")
    else:
        print("\n[4] Null/empty values in string columns")
        for col, cnt in empty.items():
            flag = "  ← non-zero" if cnt > 0 else ""
            print(f"    {col:<30} {cnt:>6,}{flag}")

    # 5. Question / answer column identification -----------------------------
    print(f"\n[5] Key columns")
    q_present = question_col in df.columns
    a_present = answer_col   in df.columns
    print(f"    Question/prompt column : '{question_col}' {'✓' if q_present else '✗  NOT FOUND'}")
    print(f"    Answer column          : '{answer_col}'   {'✓' if a_present else '✗  NOT FOUND'}")

    return {"file": rel.name, "rows": n_rows, "cols": n_cols}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print(DIVIDER)
    print("  inspect_controls.py  —  control benchmark diagnostics")
    print(DIVIDER)

    summaries = []
    for filename, q_col, a_col in FILES:
        path = BENCHMARKS_DIR / filename
        result = inspect_file(path, q_col, a_col)
        if result is not None:
            summaries.append(result)

    # Comparison table -------------------------------------------------------
    print(f"\n{DIVIDER}")
    print("  SUMMARY — row counts across all three benchmarks")
    print(DIVIDER)

    if not summaries:
        print("  No files were loaded. Run fetch_controls.py first.\n")
        return

    header = f"  {'File':<45} {'Rows':>8}  {'Cols':>5}"
    print(header)
    print(f"  {'─' * 45} {'─' * 8}  {'─' * 5}")
    for s in summaries:
        print(f"  {s['file']:<45} {s['rows']:>8,}  {s['cols']:>5}")
    print()


if __name__ == "__main__":
    main()
