"""
analyze.py
----------
Loads results/summary.csv and produces a grouped bar chart saved to
results/accuracy_by_condition.png.

  X-axis  : language condition  (english, french, msa, code_switched_tn)
  Bars    : one bar per model
  Y-axis  : accuracy (0.0 – 1.0, formatted as %)

Usage
-----
    python scripts/analyze.py
    python scripts/analyze.py --summary results/summary.csv --out results/accuracy_by_condition.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe on headless servers

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_SUMMARY = RESULTS_DIR / "summary.csv"
DEFAULT_OUT = RESULTS_DIR / "accuracy_by_condition.png"

# Canonical x-axis order
CONDITION_ORDER = ["english", "french", "msa", "code_switched_tn"]

# Human-readable labels for the x-axis ticks
CONDITION_LABELS = {
    "english": "English",
    "french": "French",
    "msa": "MSA (Arabic)",
    "code_switched_tn": "Code-Switched TN",
}

# Colour palette (one per model — cycles if more models than colours)
BAR_COLOURS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_summary(path: Path) -> pd.DataFrame:
    """Load summary CSV; index = condition, columns = model names."""
    if not path.exists():
        print(f"[ERROR] Summary file not found: {path}")
        print("        Run scripts/score_results.py first.")
        sys.exit(1)

    df = pd.read_csv(path, index_col=0)

    if df.empty:
        print("[ERROR] Summary file is empty — no data to plot.")
        sys.exit(1)

    return df


def reorder_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder rows to the canonical condition order."""
    ordered = [c for c in CONDITION_ORDER if c in df.index]
    extras = [c for c in df.index if c not in CONDITION_ORDER]
    return df.loc[ordered + extras]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_grouped_bar(df: pd.DataFrame, out_path: Path) -> None:
    """
    Draw a grouped bar chart from the accuracy pivot table.

    Parameters
    ----------
    df:
        DataFrame with conditions as index and model names as columns.
        Values are accuracy in [0, 1].
    out_path:
        Where to save the PNG.
    """
    df = reorder_conditions(df)

    conditions = list(df.index)
    models = list(df.columns)
    n_conditions = len(conditions)
    n_models = len(models)

    # Layout geometry
    group_width = 0.7          # total width devoted to bars in each group
    bar_width = group_width / max(n_models, 1)
    x = np.arange(n_conditions)

    fig, ax = plt.subplots(figsize=(max(8, n_conditions * 2.2), 5.5))

    for i, model in enumerate(models):
        colour = BAR_COLOURS[i % len(BAR_COLOURS)]
        offsets = x - group_width / 2 + (i + 0.5) * bar_width
        values = df[model].values.astype(float)

        bars = ax.bar(
            offsets,
            values,
            width=bar_width * 0.9,
            label=model,
            color=colour,
            edgecolor="white",
            linewidth=0.6,
        )

        # Value labels on top of each bar
        for bar, val in zip(bars, values):
            if np.isnan(val):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.01,
                f"{val:.0%}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#333333",
            )

    # Axes formatting
    ax.set_xticks(x)
    ax.set_xticklabels(
        [CONDITION_LABELS.get(c, c) for c in conditions],
        fontsize=11,
    )
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_xlabel("Language Condition", fontsize=12)
    ax.set_title(
        "Model Accuracy by Language Condition\n(Trilingual Technical Code-Switching Benchmark)",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.grid(axis="y", linestyle="--", alpha=0.4, color="#888888")
    ax.set_axisbelow(True)

    # Legend
    ax.legend(
        title="Model",
        title_fontsize=10,
        fontsize=10,
        loc="upper right",
        framealpha=0.85,
    )

    # Subtle spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Chart saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot accuracy by language condition.")
    parser.add_argument(
        "--summary",
        default=str(DEFAULT_SUMMARY),
        help=f"Path to summary.csv (default: {DEFAULT_SUMMARY.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help=f"Output PNG path (default: {DEFAULT_OUT.relative_to(PROJECT_ROOT)})",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.is_absolute():
        summary_path = PROJECT_ROOT / summary_path

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path

    print("=" * 60)
    print("analyze.py  —  Accuracy Chart Generator")
    print("=" * 60)
    print(f"Summary : {summary_path}")
    print(f"Output  : {out_path}")

    df = load_summary(summary_path)
    print(f"\nLoaded summary: {df.shape[0]} condition(s) × {df.shape[1]} model(s)")
    format_func = lambda v: f"{v:.1%}" if pd.notna(v) else "—"
    display = df.map(format_func) if hasattr(df, "map") else df.applymap(format_func)
    print(display.to_string())

    plot_grouped_bar(df, out_path)


if __name__ == "__main__":
    main()
