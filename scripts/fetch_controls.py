"""
fetch_controls.py
-----------------
Downloads three code-switching control benchmark datasets from HuggingFace
and saves 500-row random samples to data/control_benchmarks/.

Usage:
    python scripts/fetch_controls.py
"""

from pathlib import Path

import pandas as pd
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_SAMPLE = 500
RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "control_benchmarks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sample_and_save(ds_name: str, config: str | None, split: str, out_path: Path) -> None:
    """Load one HuggingFace dataset split, sample N_SAMPLE rows, and write a CSV."""
    label = f"{ds_name}" + (f" [{config}]" if config else "")
    try:
        print(f"\n{'─' * 60}")
        print(f"Loading : {label}  (split='{split}')")

        kwargs = {"split": split}
        if config:
            kwargs["name"] = config

        ds = load_dataset(ds_name, **kwargs)
        df = ds.to_pandas()
        original_rows = len(df)

        n = min(N_SAMPLE, original_rows)
        df_sample = df.sample(n=n, random_state=RANDOM_STATE).reset_index(drop=True)

        df_sample.to_csv(out_path, index=False)

        print(f"Dataset : {label}")
        print(f"Config  : {config or '(none)'}")
        print(f"Original rows : {original_rows:,}")
        print(f"Sampled rows  : {len(df_sample):,}")
        print(f"Saved to      : {out_path.relative_to(PROJECT_ROOT)}")

    except Exception as exc:
        print(f"\n[ERROR] Failed to fetch {label}: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("fetch_controls.py  —  downloading control benchmarks")
    print(f"Output directory : {OUT_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Sample size      : {N_SAMPLE} rows (random_state={RANDOM_STATE})")
    print("=" * 60)

    # 1. CodeMixBench — Hindi-English
    sample_and_save(
        ds_name="Tanushreeeeee/CodeMixBench",
        config="mmlu_hineng",
        split="test",
        out_path=OUT_DIR / "codemixbench_mmlu_hineng.csv",
    )

    # 2. CodeMixBench — Spanish-English
    sample_and_save(
        ds_name="Tanushreeeeee/CodeMixBench",
        config="mmlu_spaeng",
        split="test",
        out_path=OUT_DIR / "codemixbench_mmlu_spaeng.csv",
    )

    # 3. CodeMixQA
    sample_and_save(
        ds_name="gentaiscool/codemixqa",
        config=None,
        split="test",
        out_path=OUT_DIR / "codemixqa.csv",
    )

    print(f"\n{'=' * 60}")
    print("Done. Run  python scripts/inspect_controls.py  to verify.")
    print("=" * 60)


if __name__ == "__main__":
    main()
