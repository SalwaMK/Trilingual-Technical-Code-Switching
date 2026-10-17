"""
load_models.py
--------------
Reusable utilities for loading HuggingFace causal language models and
generating text responses, used across the Tunisian trilingual
code-switching evaluation project.

Supports two backends:
  - "huggingface" : downloads weights via transformers / HF Hub (requires GPU)
  - "ollama"      : calls a locally running Ollama server via HTTP
                    (no download needed beyond what Ollama already has)
"""

import json
import logging
import re
import urllib.error
import urllib.request
import warnings

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _HF_AVAILABLE = True
except ImportError:  # pragma: no cover
    _HF_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "atlas": {
        "model_id": "MBZUAI-Paris/Atlas-Chat-2B",
        "description": "Arabic-French-English trilingual chat model (2B), MBZUAI Paris",
        "expected_languages": ["ar", "fr", "en"],
    },
    "qwen": {
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "description": "Qwen2.5 instruction-tuned (3B), multilingual",
        "expected_languages": ["zh", "en", "ar", "fr"],
    },
    "jais": {
        "model_id": "inceptionai/jais-family-2p7b-chat",
        "description": "Jais family Arabic-English chat model (2.7B), Inception AI",
        "expected_languages": ["ar", "en"],
    },
}


# ---------------------------------------------------------------------------
# Core loading function
# ---------------------------------------------------------------------------

def load_model(model_id: str, cache_dir: str | None = None):
    """Load a HuggingFace causal LM tokenizer and model.

    Parameters
    ----------
    model_id:
        HuggingFace Hub model identifier (e.g. ``"Qwen/Qwen2.5-3B-Instruct"``).
    cache_dir:
        Optional local directory used by the HuggingFace cache.  When *None*
        the default HF cache location is used.

    Returns
    -------
    tuple[AutoModelForCausalLM, AutoTokenizer]
        The loaded model (in eval mode) and its tokenizer.
    """
    # -- Tokenizer -----------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
    )

    # Decoder-only models should pad on the left so that the rightmost token
    # of every sequence in a batch is always the last real input token.
    tokenizer.padding_side = "left"

    # Many causal LMs lack an explicit pad token; reuse EOS instead.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -- Model ---------------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=cache_dir,
    )
    model.eval()

    # -- Diagnostics ---------------------------------------------------------
    n_params_B = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"[load_model] model_id   : {model_id}")
    device_info = getattr(model, "hf_device_map", getattr(model, "device", "unknown"))
    print(f"[load_model] device_map : {device_info}")
    print(f"[load_model] parameters : {n_params_B:.2f}B")

    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation helper
# ---------------------------------------------------------------------------

def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 300,
    temperature: float = 0.1,
    do_sample: bool = True,
) -> str:
    """Generate a text response for *prompt* using *model* and *tokenizer*.

    The function applies the model's chat template when available, falling
    back to a simple ``<|user|> / <|assistant|>`` wrapper otherwise.

    Parameters
    ----------
    model:
        A loaded ``AutoModelForCausalLM`` in eval mode.
    tokenizer:
        The matching ``AutoTokenizer``.
    prompt:
        Raw user-facing text (no special tokens needed).
    max_new_tokens:
        Maximum number of tokens to generate beyond the prompt.
    temperature:
        Sampling temperature; lower values make output more deterministic.
    do_sample:
        Whether to use sampling (``True``) or greedy decoding (``False``).

    Returns
    -------
    str
        The decoded generated text, stripped of leading/trailing whitespace.
        Only the *newly generated* tokens are returned — the echoed prompt is
        excluded.
    """
    messages = [{"role": "user", "content": prompt}]

    # Try the model-specific chat template first.
    try:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"apply_chat_template failed ({exc!r}); "
            "falling back to generic <|user|>/<|assistant|> wrapper.",
            stacklevel=2,
        )
        formatted = f"<|user|>\n{prompt}\n<|assistant|>\n"

    # Tokenise and move to the same device as the model.
    inputs = tokenizer(formatted, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # output_ids shape: [1, input_seq_len + new_tokens]
    # Slice off the echoed prompt so we return only the new tokens.
    new_token_ids = output_ids[0][input_ids.shape[1]:]
    response = tokenizer.decode(new_token_ids, skip_special_tokens=True)

    return response.strip()


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def load_model_by_name(name: str, cache_dir: str | None = None):
    """Look up *name* in :data:`MODEL_CONFIGS` and call :func:`load_model`.

    Parameters
    ----------
    name:
        A key in ``MODEL_CONFIGS`` (e.g. ``"atlas"``, ``"qwen"``, ``"jais"``).
    cache_dir:
        Forwarded to :func:`load_model`.

    Returns
    -------
    tuple[AutoModelForCausalLM, AutoTokenizer]

    Raises
    ------
    ValueError
        If *name* is not a key in ``MODEL_CONFIGS``.
    """
    if name not in MODEL_CONFIGS:
        valid = ", ".join(f'"{k}"' for k in MODEL_CONFIGS)
        raise ValueError(
            f"Unknown model name {name!r}. Valid options are: {valid}."
        )

    model_id = MODEL_CONFIGS[name]["model_id"]
    return load_model(model_id, cache_dir=cache_dir)
