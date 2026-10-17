"""
smoke_test.py
-------------
End-to-end smoke test for the Tunisian trilingual code-switching evaluation
project.

Sends a single mixed-language prompt to each selected model and prints the
raw response.  Useful to confirm that all backends are reachable and
generating sensible output before running the full benchmark.

The test prompt mixes:
  - Arabizi  (e.g. '3andi', 'w', 'ta3ha', 'bch', 'kifesh', 'nstaamel')
  - French   (e.g. 'avec', 'une résistance', 'Aide-moi')
  - English  (technical terms: 'circuit', 'voltage', "Ohm's law")

Backends supported
------------------
  hf_api      — HuggingFace Inference API (zero download, needs HF_TOKEN)
  huggingface — local transformers inference (needs GPU)
  ollama      — local Ollama server (needs `ollama serve`)

Models (0.3B – 6B range)
------------------------
  HF API  : atlas, qwen_3b, qwen_15b, qwen_05b, phi, jais
  HF local: atlas_local, qwen_local, jais_local
  Ollama  : deepseek, qwen_coder, qwen_mini

Usage
-----
    # Recommended: test the three default HF API models (no GPU needed)
    python scripts/smoke_test.py

    # Test only HF API models
    python scripts/smoke_test.py --api-only

    # Test only Ollama models
    python scripts/smoke_test.py --ollama-only

    # Test specific models
    python scripts/smoke_test.py --models atlas,qwen_3b,deepseek
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from load_models import (  # noqa: E402
    DEFAULT_MODELS,
    HF_API_MODELS,
    HF_LOCAL_MODELS,
    MODEL_CONFIGS,
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
# Tunisian trilingual test prompt
# ---------------------------------------------------------------------------

TEST_PROMPT = (
    "3andi circuit avec une résistance de 10 ohms w current ta3ha 2 ampères. "
    "Nbaghi ncalculi el voltage. "
    "Aide-moi bch nfhem: kifesh nstaamel Ohm's law hna?"
)

# ---------------------------------------------------------------------------
# Main smoke-test loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test registered models with one trilingual prompt.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        metavar="KEYS",
        help=(
            "Comma-separated model keys to test. "
            f"Default: {','.join(DEFAULT_MODELS)} (local HF — atlas, qwen_3b, jais)."
        ),
    )
    parser.add_argument(
        "--api-only",
        action="store_true",
        help="Test only HF API models (no GPU, no Ollama needed).",
    )
    parser.add_argument(
        "--ollama-only",
        action="store_true",
        help="Test only Ollama-backed models.",
    )
    args = parser.parse_args()

    # Resolve model list
    if args.models:
        names = [m.strip() for m in args.models.split(",") if m.strip()]
    elif args.api_only:
        names = list(HF_API_MODELS)
    elif args.ollama_only:
        names = [k for k in MODEL_CONFIGS if is_ollama(k)]
    else:
        names = list(DEFAULT_MODELS)  # atlas, qwen_3b, jais (local HF)

    unknown = [n for n in names if n not in MODEL_CONFIGS]
    if unknown:
        print(f"[ERROR] Unknown model key(s): {unknown}")
        print(f"        Valid keys: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    if not names:
        print("No models selected.")
        return

    # Pre-flight checks
    ollama_ok: bool | None = None

    print("=" * 70)
    print("smoke_test.py  —  Trilingual Code-Switching Smoke Test")
    print("=" * 70)
    print(f"Models to test : {names}")
    print(f"\nPrompt:\n  {TEST_PROMPT}\n")

    for name in names:
        cfg = MODEL_CONFIGS[name]
        model_id = cfg["model_id"]
        backend = cfg["backend"]
        params = cfg.get("params", "?")
        label = {"hf_api": "HF API", "huggingface": "HF local", "ollama": "Ollama"}.get(backend, backend)

        print("=" * 70)
        print(f"  Model   : {name}  ({model_id})")
        print(f"  Backend : {label}   Params: {params}")
        print("=" * 70)

        response = None

        # ── HF API ──────────────────────────────────────────────────────────
        if is_hf_api(name):
            try:
                response = hf_api_generate(model_id, TEST_PROMPT)
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] HF API call failed: {exc}")
                print("        Is HF_TOKEN set?  Run: export HF_TOKEN=hf_...")

        # ── Ollama ──────────────────────────────────────────────────────────
        elif is_ollama(name):
            if ollama_ok is None:
                ollama_ok = check_ollama()
            if not ollama_ok:
                print("[SKIP] Ollama server not reachable — run: ollama serve")
                print("-" * 70)
                continue
            try:
                response = ollama_generate(cfg["ollama_name"], TEST_PROMPT)
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] Ollama generation failed: {exc}")

        # ── HuggingFace local ────────────────────────────────────────────────
        elif is_hf_local(name):
            try:
                model, tokenizer = load_model_by_name(name)
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] Failed to load '{name}': {exc}")
                print("-" * 70)
                continue

            try:
                response = generate_response(model, tokenizer, TEST_PROMPT)
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] Generation failed: {exc}")

            # Free GPU memory
            del model, tokenizer
            try:
                import torch  # noqa: PLC0415
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        # ── Display ─────────────────────────────────────────────────────────
        if response is not None:
            print(f"\n  Response:\n  {response}\n")
        else:
            print("  (no response generated)\n")

        print("-" * 70)


if __name__ == "__main__":
    main()
