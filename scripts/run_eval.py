"""
run_eval.py
-----------
Evaluation runner for the Trilingual Technical Code-Switching benchmark.

For each problem in data/core_problems.json, runs all 4 language conditions
(english, french, msa, code_switched_tn) through the selected models and
saves raw outputs to results/raw_outputs.csv.

Features
--------
- Three backends: hf_api (default), huggingface (local GPU), ollama (local server)
- Resumable: already-evaluated (problem_id, model, condition) triples are skipped.
- --limit N: evaluate only the first N problems (quick smoke tests).
- --models: comma-separated model keys (default: atlas,qwen_3b,jais).
- --conditions: comma-separated language conditions (default: all four).

Usage
-----
    # Recommended: three primary HF API models, all problems
    python scripts/run_eval.py --models atlas,qwen_3b,jais

    # Quick 3-problem test
    python scripts/run_eval.py --models atlas --limit 3

    # Ollama models
    python scripts/run_eval.py --models deepseek,qwen_coder,qwen_mini

    # Local HF (requires GPU)
    python scripts/run_eval.py --models atlas_local,qwen_local
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — makes load_models importable regardless of cwd
# ---------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from load_models import (  # noqa: E402
    DEFAULT_MODELS,
    HF_API_MODELS,
    HF_LOCAL_MODELS,
    MODEL_CONFIGS,
    OLLAMA_MODELS,
    check_ollama,
    generate_response,
    hf_api_generate,
    is_hf_api,
    is_hf_local,
    is_ollama,
    load_model_by_name,
    ollama_generate,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_PATH = PROJECT_ROOT / "data" / "core_problems.json"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_PATH = RESULTS_DIR / "raw_outputs.csv"

LANGUAGE_CONDITIONS = ["english", "french", "msa", "code_switched_tn"]

OUTPUT_COLUMNS = [
    "problem_id",
    "domain",
    "model",
    "condition",
    "prompt_text",
    "model_response",
    "expected_answer",
]

PROMPT_TEMPLATE = (
    "{problem_statement}\n\n"
    "Answer concisely with just the answer (for example: '20V', 'O(log n)', or '254'). "
    "Do not explain."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_problems(path: Path, limit: int | None = None) -> list[dict]:
    """Load and return problem entries from the JSON file (strips _schema docs)."""
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    problems = [
        obj for obj in data
        if isinstance(obj, dict) and list(obj.keys()) != ["_schema"]
    ]

    if limit is not None:
        problems = problems[:limit]

    return problems


def load_existing_results(path: Path) -> set[tuple[str, str, str]]:
    """Return a set of (problem_id, model, condition) already evaluated."""
    if not path.exists():
        return set()

    try:
        df = pd.read_csv(path, dtype=str)
        required = {"problem_id", "model", "condition"}
        if not required.issubset(df.columns):
            print(f"[warn] {path.name} exists but missing columns {required - set(df.columns)}. Starting fresh.")
            return set()
        done = set(zip(df["problem_id"], df["model"], df["condition"]))
        print(f"[resume] Found {len(done)} existing rows — will skip these.")
        return done
    except Exception as exc:
        print(f"[warn] Could not read existing results ({exc}). Starting fresh.")
        return set()


def append_rows(rows: list[dict], path: Path) -> None:
    """Append a list of result dicts to the CSV (creates with header if needed)."""
    if not rows:
        return
    df_new = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    write_header = not path.exists()
    df_new.to_csv(path, mode="a", header=write_header, index=False)


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def run_eval(
    problems: list[dict],
    model_names: list[str],
    conditions: list[str],
    done: set[tuple[str, str, str]],
) -> None:
    """Iterate models → problems → conditions, generate responses, and persist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Check Ollama once if any Ollama model is requested
    ollama_ok: bool | None = None

    for model_name in model_names:
        cfg = MODEL_CONFIGS[model_name]
        model_id = cfg["model_id"]

        # Determine which (problem_id, condition) pairs this model still needs.
        needed = [
            (prob, cond)
            for prob in problems
            for cond in conditions
            if (prob["id"], model_name, cond) not in done
        ]

        if not needed:
            print(f"\n[{model_name}] All rows already present — skipping.")
            continue

        backend = cfg["backend"]
        backend_tag = {"hf_api": "[HF API]", "huggingface": "[HF local]", "ollama": "[Ollama]"}.get(backend, "")

        print(f"\n{'=' * 70}")
        print(f"Model : {model_name}  ({model_id})  {backend_tag}")
        print(f"Tasks : {len(needed)} prompt(s) to generate")
        print("=" * 70)

        model, tokenizer = None, None

        # ── Ollama pre-flight ────────────────────────────────────────────────
        if is_ollama(model_name):
            if ollama_ok is None:
                ollama_ok = check_ollama()
            if not ollama_ok:
                print("[ERROR] Ollama server not reachable. Start it with: ollama serve")
                print("        Skipping this model.")
                continue
            ollama_name = cfg["ollama_name"]

        # ── HuggingFace local: load weights into memory ──────────────────────
        elif is_hf_local(model_name):
            try:
                model, tokenizer = load_model_by_name(model_name)
            except Exception as exc:
                print(f"[ERROR] Could not load '{model_name}': {exc}")
                print("        Skipping this model.")
                continue

        # ── HF API: nothing to load (just HTTP calls) ────────────────────────
        # else: is_hf_api(model_name) — no setup needed

        # ── Generate ─────────────────────────────────────────────────────────
        batch: list[dict] = []

        for prob, cond in needed:
            pid = prob["id"]
            domain = prob.get("domain", "")
            prompt_text = prob.get(cond, "")
            expected = prob.get("expected_answer", "")

            if not prompt_text:
                print(f"  [warn] Problem '{pid}' has no text for condition '{cond}' — skipping.")
                continue

            full_prompt = PROMPT_TEMPLATE.format(problem_statement=prompt_text)

            print(f"  [{model_name}] {pid} / {cond} … ", end="", flush=True)
            try:
                if is_ollama(model_name):
                    response = ollama_generate(ollama_name, full_prompt)
                elif is_hf_local(model_name):
                    response = generate_response(model, tokenizer, full_prompt)
                else:  # hf_api
                    response = hf_api_generate(model_id, full_prompt)
                print("done")
            except Exception as exc:
                response = f"[GENERATION_ERROR: {exc}]"
                print(f"ERROR: {exc}")

            batch.append({
                "problem_id": pid,
                "domain": domain,
                "model": model_name,
                "condition": cond,
                "prompt_text": prompt_text,
                "model_response": response,
                "expected_answer": expected,
            })

            done.add((pid, model_name, cond))

            # Flush every 10 rows to protect against interruption
            if len(batch) >= 10:
                append_rows(batch, OUTPUT_PATH)
                batch = []

        # Final flush
        append_rows(batch, OUTPUT_PATH)

        # ── Free GPU memory (local HF only) ───────────────────────────────────
        if is_hf_local(model_name) and model is not None:
            try:
                import torch  # noqa: PLC0415
                del model, tokenizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        print(f"[{model_name}] Done. Results appended to {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the trilingual code-switching evaluation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate only the first N problems (useful for quick tests).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        metavar="KEYS",
        help=(
            f"Comma-separated model keys to evaluate. "
            f"HF local (default): {', '.join(HF_LOCAL_MODELS)}. "
            f"HF API: {', '.join(HF_API_MODELS)}. "
            f"Ollama: {', '.join(OLLAMA_MODELS)}. "
            f"Default: {','.join(DEFAULT_MODELS)}."
        ),
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default=",".join(LANGUAGE_CONDITIONS),
        metavar="CONDS",
        help=(
            f"Comma-separated language conditions. "
            f"Available: {', '.join(LANGUAGE_CONDITIONS)}. "
            f"Default: all."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validate model names
    requested_models = [m.strip() for m in args.models.split(",") if m.strip()]
    invalid_models = [m for m in requested_models if m not in MODEL_CONFIGS]
    if invalid_models:
        print(f"[ERROR] Unknown model key(s): {invalid_models}")
        print(f"        Valid keys: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    # Validate conditions
    requested_conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    invalid_conds = [c for c in requested_conditions if c not in LANGUAGE_CONDITIONS]
    if invalid_conds:
        print(f"[ERROR] Unknown condition(s): {invalid_conds}")
        print(f"        Valid conditions: {LANGUAGE_CONDITIONS}")
        sys.exit(1)

    # Load dataset
    if not DATA_PATH.exists():
        print(f"[ERROR] Dataset not found: {DATA_PATH}")
        print("        Create data/core_problems.json first.")
        sys.exit(1)

    problems = load_problems(DATA_PATH, limit=args.limit)
    if not problems:
        print("[ERROR] No problem entries found in the dataset (only _schema?).")
        sys.exit(1)

    print("=" * 70)
    print("run_eval.py  —  Trilingual Code-Switching Evaluation")
    print("=" * 70)
    print(f"Dataset   : {DATA_PATH.relative_to(PROJECT_ROOT)}  ({len(problems)} problems)")
    print(f"Models    : {requested_models}")
    print(f"Conditions: {requested_conditions}")
    print(f"Output    : {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print("=" * 70)

    done = load_existing_results(OUTPUT_PATH)

    run_eval(
        problems=problems,
        model_names=requested_models,
        conditions=requested_conditions,
        done=done,
    )

    # Final summary
    if OUTPUT_PATH.exists():
        df = pd.read_csv(OUTPUT_PATH)
        print(f"\n{'=' * 70}")
        print(f"Finished. Total rows in {OUTPUT_PATH.name}: {len(df):,}")
        print(f"{'=' * 70}")
    else:
        print("\n[warn] No output file was created (all tasks may have been skipped).")


if __name__ == "__main__":
    main()
