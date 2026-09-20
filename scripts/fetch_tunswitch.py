"""
fetch_tunswitch.py

Extract text transcripts from tunis-ai/TunSwitch.

The dataset is an audiofolder repo; text lives inside ZIP archives
(TunSwitchCS.zip, TunSwitchTO.zip) in SpeechBrain CSV format.
The transcript field is named `wrd`.

Strategy (in order):
  1. snapshot_download  → scan ZIPs for SpeechBrain CSVs with a `wrd` column
  2. load_dataset streaming fallback (audio only — no text available)
  3. Final fallback      → create an empty CSV with a header row

Run from the project root:
    python scripts/fetch_tunswitch.py
"""

import io
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "tunswitch_transcripts.csv"

REPO_ID = "tunis-ai/TunSwitch"

# Columns to keep from SpeechBrain CSVs when available
KEEP_COLS_PRIMARY = ["ID", "wrd"]
KEEP_COLS_EXTRA = ["spk_id"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_from_zip(zip_path: Path, tmp_root: Path) -> list[pd.DataFrame]:
    """Unzip *zip_path* into a temp subdir and return DataFrames that have a `wrd` column."""
    frames: list[pd.DataFrame] = []
    extract_dir = tmp_root / zip_path.stem
    extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
    except zipfile.BadZipFile as exc:
        print(f"    [warn] Could not unzip {zip_path.name}: {exc}")
        return frames

    for csv_file in sorted(extract_dir.rglob("*.csv")):
        try:
            candidate = pd.read_csv(csv_file, dtype=str)
        except Exception as exc:
            print(f"    [warn] Could not read {csv_file}: {exc}")
            continue

        if "wrd" not in candidate.columns:
            continue

        keep = [c for c in KEEP_COLS_PRIMARY + KEEP_COLS_EXTRA if c in candidate.columns]
        frames.append(candidate[keep].copy())
        print(f"    Found {len(candidate):,} rows in {csv_file.relative_to(extract_dir)}")

    return frames


def _try_snapshot_download() -> pd.DataFrame | None:
    """Primary path: snapshot_download → ZIP scan."""
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
    except ImportError:
        print("[step 1] huggingface_hub not installed — skipping snapshot_download.")
        return None

    print(f"[step 1] Attempting snapshot_download of {REPO_ID} …")
    try:
        repo_dir = Path(snapshot_download(repo_id=REPO_ID, repo_type="dataset"))
    except Exception as exc:
        print(f"[step 1] snapshot_download failed: {exc}")
        return None

    print(f"  Repo cached at: {repo_dir}")
    zip_files = sorted(repo_dir.rglob("*.zip"))
    if not zip_files:
        print("  No ZIP files found in repo snapshot.")
        return None

    print(f"  Found {len(zip_files)} ZIP file(s): {[z.name for z in zip_files]}")

    all_frames: list[pd.DataFrame] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        for zp in zip_files:
            frames = _collect_from_zip(zp, tmp_root)
            all_frames.extend(frames)

    if not all_frames:
        print("  No SpeechBrain CSVs with a `wrd` column found inside ZIPs.")
        return None

    combined = pd.concat(all_frames, ignore_index=True)

    # Deduplicate on ID if present
    if "ID" in combined.columns:
        before = len(combined)
        combined = combined.drop_duplicates(subset="ID")
        if len(combined) < before:
            print(f"  Deduplicated on ID: {before:,} → {len(combined):,} rows")

    # Normalise column names for the output file
    rename = {"ID": "id", "wrd": "transcript", "spk_id": "spk_id"}
    combined = combined.rename(columns={k: v for k, v in rename.items() if k in combined.columns})

    print(f"[step 1] SUCCESS — {len(combined):,} transcripts collected.")
    return combined


def _try_streaming_fallback() -> pd.DataFrame | None:
    """Secondary path: load_dataset streaming (audio only — no text)."""
    print("[step 2] Attempting load_dataset streaming fallback …")
    try:
        from datasets import load_dataset  # noqa: PLC0415

        ds = load_dataset(REPO_ID, split="test", streaming=True)
        # Peek at the first example to confirm schema
        first = next(iter(ds), None)
        if first is None:
            print("[step 2] Streaming dataset appears empty.")
            return None

        available_cols = list(first.keys())
        print(f"  Streaming columns available: {available_cols}")

        if "wrd" in available_cols or "transcript" in available_cols:
            # Rare lucky case — text is actually present
            text_col = "wrd" if "wrd" in available_cols else "transcript"
            rows = [{"id": ex.get("ID", ex.get("id", idx)), "transcript": ex[text_col]}
                    for idx, ex in enumerate(ds)]
            df = pd.DataFrame(rows)
            print(f"[step 2] SUCCESS — {len(df):,} rows with text found via streaming.")
            return df

        # Audio-only — no text column
        print(
            "[step 2] WARNING: Streaming split has no text column (audio-only). "
            "Text transcripts are only available in the ZIP archives inside the repo. "
            "Run with HuggingFace credentials to use snapshot_download (step 1)."
        )
        return None

    except Exception as exc:
        print(f"[step 2] Streaming fallback failed: {exc}")
        return None


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 — snapshot_download
    # ------------------------------------------------------------------
    result = _try_snapshot_download()

    # ------------------------------------------------------------------
    # Step 2 — streaming fallback
    # ------------------------------------------------------------------
    if result is None:
        result = _try_streaming_fallback()

    # ------------------------------------------------------------------
    # Step 3 — final fallback: empty CSV
    # ------------------------------------------------------------------
    if result is None:
        print(
            "[step 3] Both primary and fallback paths failed. "
            "Writing an empty CSV with header row.\n"
            "         To get transcripts, ensure network access and valid HuggingFace "
            "credentials, then re-run this script."
        )
        result = pd.DataFrame(columns=["id", "transcript"])

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved → {OUTPUT_PATH}  ({len(result):,} rows)")


if __name__ == "__main__":
    main()
