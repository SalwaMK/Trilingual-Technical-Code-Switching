"""
fetch_tunisia_tech.py

Load Farah21/Tunisia_Tech from HuggingFace (train + test splits),
apply quality and topic filters, and save the result to
data/tunisia_tech_filtered.csv (relative to the project root).

Run from the project root:
    python scripts/fetch_tunisia_tech.py
"""

from pathlib import Path

import pandas as pd
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Thresholds — adjust here without touching the filter logic below
# ---------------------------------------------------------------------------
QUALITY_THRESHOLD = 0.65
TECH_THRESHOLD = 0.05

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "tunisia_tech_filtered.csv"

TARGET_DIALECTS = {"mixed", "tunisian_arabizi"}


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load both splits and concatenate
    # ------------------------------------------------------------------
    print("Loading Farah21/Tunisia_Tech (train + test) …")
    raw = load_dataset("Farah21/Tunisia_Tech")

    train_df = raw["train"].to_pandas()
    test_df = raw["test"].to_pandas()
    df = pd.concat([train_df, test_df], ignore_index=True)

    total_rows = len(df)
    print(f"  Rows loaded  : {total_rows:,}  (train={len(train_df):,}, test={len(test_df):,})")
    print(f"  Columns      : {list(df.columns)}")

    # ------------------------------------------------------------------
    # 2. Filter by dialect
    # ------------------------------------------------------------------
    df = df[df["dialect"].isin(TARGET_DIALECTS)].copy()
    after_dialect = len(df)
    print(f"\nAfter dialect filter {sorted(TARGET_DIALECTS)!r}: {after_dialect:,} rows")

    # ------------------------------------------------------------------
    # 3. Filter by quality_score
    # ------------------------------------------------------------------
    df = df[df["quality_score"] >= QUALITY_THRESHOLD].copy()
    after_quality = len(df)
    print(f"After quality_score >= {QUALITY_THRESHOLD}        : {after_quality:,} rows")

    # ------------------------------------------------------------------
    # 4. Filter by tech_confidence
    # ------------------------------------------------------------------
    df = df[df["tech_confidence"] >= TECH_THRESHOLD].copy()
    after_tech = len(df)
    print(f"After tech_confidence >= {TECH_THRESHOLD}         : {after_tech:,} rows")

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved → {OUTPUT_PATH}")
    print(f"  Columns saved: {list(df.columns)}")
    print(
        f"\nSummary: {total_rows:,} loaded → {after_dialect:,} (dialect)"
        f" → {after_quality:,} (quality) → {after_tech:,} (tech) rows retained."
    )


if __name__ == "__main__":
    main()
