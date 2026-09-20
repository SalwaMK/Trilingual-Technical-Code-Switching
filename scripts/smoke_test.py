"""
smoke_test.py
-------------
End-to-end smoke test for the Tunisian trilingual code-switching evaluation
project.

For each model registered in MODEL_CONFIGS, this script:
  1. Downloads / loads the model from HuggingFace Hub.
  2. Runs a single Tunisian trilingual test prompt that mixes:
       - Arabizi  (e.g. '3andi', 'w', 'ta3ha', 'bch', 'nfhem', 'nstaamel')
       - French   (e.g. 'avec', 'une résistance', 'Aide-moi')
       - English  (technical terms: 'circuit', 'resistance', 'voltage',
                   "Ohm's law")
  3. Prints the raw model response.
  4. Frees GPU memory before moving on to the next model.

Models tested
-------------
  • atlas  — MBZUAI-Paris/Atlas-Chat-2B
  • qwen   — Qwen/Qwen2.5-3B-Instruct
  • jais   — inceptionai/jais-family-2p7b-chat

Run from any working directory:
    python scripts/smoke_test.py
"""

import sys
from pathlib import Path

# Ensure the scripts/ directory is on the path so we can import load_models
# regardless of the working directory the user invokes this script from.
sys.path.insert(0, str(Path(__file__).parent))

import torch  # noqa: E402  (import after sys.path manipulation)

from load_models import MODEL_CONFIGS, generate_response, load_model_by_name  # noqa: E402

# ---------------------------------------------------------------------------
# Tunisian trilingual test prompt
# ---------------------------------------------------------------------------

TEST_PROMPT = (
    "3andi circuit avec une résistance de 10 ohms w current ta3ha 2 ampères. "
    "Ich will calculate el voltage. "
    "Aide-moi bch nfhem: kifesh nstaamel Ohm's law hna?"
)

# ---------------------------------------------------------------------------
# Main smoke-test loop
# ---------------------------------------------------------------------------

def main() -> None:
    for name, cfg in MODEL_CONFIGS.items():
        model_id = cfg["model_id"]

        print()
        print("=" * 70)
        print(f"=== Testing model: {name} ({model_id}) ===")
        print("=" * 70)

        # -- Load ------------------------------------------------------------
        try:
            model, tokenizer = load_model_by_name(name)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Failed to load model '{name}': {exc}")
            print("SKIP")
            continue

        # -- Generate --------------------------------------------------------
        try:
            response = generate_response(model, tokenizer, TEST_PROMPT)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Generation failed for model '{name}': {exc}")
            # Still clean up below before continuing.
            response = None

        # -- Display ---------------------------------------------------------
        if response is not None:
            print()
            print("--- PROMPT ---")
            print(TEST_PROMPT)
            print("--- RESPONSE ---")
            print(response)
            print("--- END ---")

        # -- Free GPU memory -------------------------------------------------
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("-" * 70)


if __name__ == "__main__":
    main()
