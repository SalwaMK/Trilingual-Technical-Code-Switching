"""
score_results.py
----------------
Scores raw model outputs from results/raw_outputs.csv and produces:

  results/scored_results.csv  — per-row scores (adds a `score` column)
  results/summary.csv         — pivot table: accuracy by (model × condition)

Scoring logic
-------------
The `score` column can take three values:

  "correct"             — model response contains the expected answer
                          (case-insensitive, stripped, fuzzy substring match)
  "incorrect"           — model response was generated but doesn't match
  "needs_manual_review" — expected answer is empty/missing, or the response
                          is a generation error, so automatic scoring isn't
                          reliable

For open-ended technical explanations where the expected answer is long prose,
mark those entries "needs_manual_review" in core_problems.json by leaving
expected_answer as an empty string — you can then score them manually.

Usage
-----
    python scripts/score_results.py
    python scripts/score_results.py --input results/raw_outputs.csv
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_INPUT = RESULTS_DIR / "raw_outputs.csv"
SCORED_OUTPUT = RESULTS_DIR / "scored_results.csv"
SUMMARY_OUTPUT = RESULTS_DIR / "summary.csv"

# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

_GENERATION_ERROR_RE = re.compile(r"^\[GENERATION_ERROR", re.IGNORECASE)


def _normalise(text: str) -> str:
    """Lowercase, strip whitespace, collapse internal spaces."""
    return re.sub(r"\s+", " ", str(text).lower().strip())


def _score_row(response: str, expected: str) -> str:
    """
    Return 'correct', 'incorrect', or 'needs_manual_review' for one row.

    Rules (evaluated in order):
    1. If expected is empty/NaN → needs_manual_review
    2. If response starts with [GENERATION_ERROR → needs_manual_review
    3. Exact match (after normalisation) → correct
    4. Fuzzy: normalised expected is a substring of normalised response → correct
    5. Otherwise → incorrect
    """
    # 1. Missing expected answer
    if not expected or _normalise(expected) in ("", "nan"):
        return "needs_manual_review"

    # 2. Generation error
    if _GENERATION_ERROR_RE.match(str(response)):
        return "needs_manual_review"

    norm_response = _normalise(response)
    norm_expected = _normalise(expected)

    # 3 & 4. Exact or fuzzy substring match
    if norm_expected == norm_response or norm_expected in norm_response:
        return "correct"

    # 5. No match
    return "incorrect"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def score(input_path: Path) -> pd.DataFrame:
    """Load raw outputs, add score column, return scored DataFrame."""
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        print("        Run scripts/run_eval.py first.")
        sys.exit(1)

    df = pd.read_csv(input_path, dtype=str)

    required = {"problem_id", "model", "condition", "model_response", "expected_answer"}
    missing = required - set(df.columns)
    if missing:
        print(f"[ERROR] raw_outputs.csv is missing columns: {missing}")
        sys.exit(1)

    # Fill NaN strings
    df["model_response"] = df["model_response"].fillna("")
    df["expected_answer"] = df["expected_answer"].fillna("")

    df["score"] = df.apply(
        lambda row: _score_row(row["model_response"], row["expected_answer"]),
        axis=1,
    )

    return df


def build_summary(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a model × condition pivot table of accuracy.

    Accuracy = correct / (correct + incorrect)
    Rows flagged 'needs_manual_review' are excluded from the denominator
    so they don't artificially lower the score.
    """
    # Work only on auto-scoreable rows
    auto = scored_df[scored_df["score"] != "needs_manual_review"].copy()

    if auto.empty:
        print("[warn] No auto-scoreable rows found — summary will be empty.")
        return pd.DataFrame()

    auto["is_correct"] = (auto["score"] == "correct").astype(int)

    # Group by model × condition
    grouped = (
        auto.groupby(["model", "condition"])["is_correct"]
        .agg(correct="sum", total="count")
        .reset_index()
    )
    grouped["accuracy"] = grouped["correct"] / grouped["total"]

    # Pivot: rows = condition, columns = model
    pivot = grouped.pivot(index="condition", columns="model", values="accuracy")

    # Reorder conditions to the canonical order if possible
    canonical = ["english", "french", "msa", "code_switched_tn"]
    ordered = [c for c in canonical if c in pivot.index] + [
        c for c in pivot.index if c not in canonical
    ]
    pivot = pivot.loc[ordered]

    # Round for readability
    pivot = pivot.round(3)

    return pivot


def print_summary(pivot: pd.DataFrame) -> None:
    """Pretty-print the accuracy pivot table."""
    divider = "=" * 70
    print(f"\n{divider}")
    print("  ACCURACY SUMMARY  (correct / auto-scoreable rows)")
    print(divider)

    if pivot.empty:
        print("  (no data)")
    else:
        # Format as percentage strings for console
        format_func = lambda v: f"{v:.1f}%" if pd.notna(v) else "—"
        display = (pivot * 100).map(format_func) if hasattr(pivot, "map") else (pivot * 100).applymap(format_func)
        print(display.to_string())

    print(divider)


def print_score_distribution(scored_df: pd.DataFrame) -> None:
    """Print per-model score breakdown."""
    print("\n  SCORE DISTRIBUTION by model")
    print("─" * 50)
    dist = (
        scored_df.groupby(["model", "score"])
        .size()
        .unstack(fill_value=0)
    )
    print(dist.to_string())
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Score raw evaluation outputs.")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help=f"Path to raw_outputs.csv (default: {DEFAULT_INPUT.relative_to(PROJECT_ROOT)})",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path

    print("=" * 70)
    print("score_results.py  —  Trilingual Code-Switching Scoring")
    print("=" * 70)
    print(f"Input  : {input_path}")
    print(f"Output : {SCORED_OUTPUT}")
    print(f"Summary: {SUMMARY_OUTPUT}")

    # Score
    scored = score(input_path)

    # Save scored results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scored.to_csv(SCORED_OUTPUT, index=False)
    print(f"\nSaved scored results → {SCORED_OUTPUT.relative_to(PROJECT_ROOT)}")
    print(f"Total rows: {len(scored):,}")

    # Score distribution
    print_score_distribution(scored)

    # Build and save summary
    pivot = build_summary(scored)
    if not pivot.empty:
        pivot.to_csv(SUMMARY_OUTPUT)
        print(f"Saved summary       → {SUMMARY_OUTPUT.relative_to(PROJECT_ROOT)}")

    # Print summary to console
    print_summary(pivot)

    # Manual review count
    n_manual = (scored["score"] == "needs_manual_review").sum()
    if n_manual > 0:
        manual_ids = scored[scored["score"] == "needs_manual_review"]["problem_id"].unique()
        print(f"\n  {n_manual} row(s) flagged for manual review:")
        for pid in manual_ids:
            print(f"    • {pid}")
        print()


if __name__ == "__main__":
    main()
