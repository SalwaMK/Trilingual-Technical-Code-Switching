"""
load_models.py
--------------
Reusable utilities for model inference across the Tunisian trilingual
code-switching evaluation project.

Three backends
--------------
  "huggingface" : Downloads weights from HF Hub, runs locally via
                  transformers on GPU (device_map="auto").
                  ✓ Works with ANY HF model.  Requires a CUDA GPU.
                  → Primary recommendation for this project.

  "hf_api"      : Calls HuggingFace's own inference servers via
                  InferenceClient(provider="hf-inference").
                  Works only for popular models HF has deployed on TGI.
                  Requires an HF token (export HF_TOKEN=hf_...).
                  → Useful as a no-download fallback for supported models.

  "ollama"      : Calls a locally running Ollama server via HTTP.
                  → Use for models you already have in Ollama.
"""

import json
import logging
import os
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
# Ollama configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = "http://localhost:11434"

# Strip DeepSeek-R1 chain-of-thought blocks before scoring
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Each entry:
#   model_id   — HF repo path  OR  Ollama tag
#   backend    — "huggingface" | "hf_api" | "ollama"
#   params     — human-readable size
#   description
#   expected_languages
#
# For "ollama" also set:
#   ollama_name — exact tag from `ollama list`

MODEL_CONFIGS = {

    # ── HuggingFace local (primary recommendation) ─────────────────────────
    # Downloaded on first run (~2–8 GB each), cached in ~/.cache/huggingface.
    # Runs locally via transformers on your GPU.  Works 100% reliably.
    #
    # Ranked by multilingual quality for Ar / Fr / En code-switching:

    "atlas": {
        "model_id": "MBZUAI-Paris/Atlas-Chat-2B",
        "params": "2B",
        "description": "Atlas-Chat (2B) — Moroccan Arabic/Darija comparison model",
        "expected_languages": ["ar", "fr", "en"],
        "backend": "huggingface",
        "challenge_eligible": True,
    },
    "qwen_3b": {
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "params": "3B",
        "description": "Qwen2.5-Instruct (3B) — strong multilingual (En/Ar/Fr/Zh)",
        "expected_languages": ["zh", "en", "ar", "fr"],
        "backend": "huggingface",
        "challenge_eligible": True,
    },
    "jais": {
        "model_id": "inceptionai/jais-family-2p7b-chat",
        "params": "2.7B",
        "description": "Jais-family-2.7B-chat — Arabic + English, Inception AI",
        "expected_languages": ["ar", "en"],
        "backend": "huggingface",
        "challenge_eligible": True,
    },
    "phi": {
        "model_id": "microsoft/Phi-3.5-mini-instruct",
        "params": "3.8B",
        "description": "Phi-3.5-mini-instruct (3.8B) — decent multilingual, Microsoft",
        "expected_languages": ["en", "ar", "fr"],
        "backend": "huggingface",
        "challenge_eligible": True,
    },
    "qwen_15b": {
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "params": "1.5B",
        "description": "Qwen2.5-Instruct (1.5B) — compact multilingual",
        "expected_languages": ["zh", "en", "ar", "fr"],
        "backend": "huggingface",
        "challenge_eligible": True,
    },
    "qwen_05b": {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "params": "0.5B",
        "description": "Qwen2.5-Instruct (0.5B) — tiny baseline",
        "expected_languages": ["zh", "en"],
        "backend": "huggingface",
        "challenge_eligible": False,
    },

    # ── HuggingFace Inference API (optional, limited model support) ─────────
    # Uses HF's own servers (provider="hf-inference").
    # Only popular models that HF has deployed on TGI work here.
    # Niche models (Atlas, Jais) are NOT available — use local instead.
    # Requires: export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

    "qwen_3b_api": {
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "params": "3B",
        "description": "Qwen2.5-Instruct (3B) via HF Inference API (no download)",
        "expected_languages": ["zh", "en", "ar", "fr"],
        "backend": "hf_api",
        "challenge_eligible": True,
    },
    "phi_api": {
        "model_id": "microsoft/Phi-3.5-mini-instruct",
        "params": "3.8B",
        "description": "Phi-3.5-mini-instruct (3.8B) via HF Inference API (no download)",
        "expected_languages": ["en", "ar", "fr"],
        "backend": "hf_api",
        "challenge_eligible": True,
    },

    # ── Ollama (locally served, already on your machine) ───────────────────

    "deepseek": {
        "model_id": "deepseek-r1:14b",
        "ollama_name": "deepseek-r1:14b",
        "params": "14B",
        "description": "DeepSeek-R1 (14B) via Ollama — reasoning model, <think> stripped",
        "expected_languages": ["en", "zh", "ar", "fr"],
        "backend": "ollama",
        "challenge_eligible": False,
    },
    "qwen_coder": {
        "model_id": "qwen2.5-coder:14b",
        "ollama_name": "qwen2.5-coder:14b",
        "params": "14B",
        "description": "Qwen2.5-Coder (14B) via Ollama — strongest on coding domains",
        "expected_languages": ["en", "zh"],
        "backend": "ollama",
        "challenge_eligible": False,
    },
    "qwen_mini": {
        "model_id": "qwen2.5-coder:1.5b",
        "ollama_name": "qwen2.5-coder:1.5b",
        "params": "1.5B",
        "description": "Qwen2.5-Coder (1.5B) via Ollama — fast, lightweight",
        "expected_languages": ["en", "zh"],
        "backend": "ollama",
        "challenge_eligible": False,
    },
}

# Convenience groupings
HF_LOCAL_MODELS = [k for k, v in MODEL_CONFIGS.items() if v["backend"] == "huggingface"]
HF_API_MODELS   = [k for k, v in MODEL_CONFIGS.items() if v["backend"] == "hf_api"]
OLLAMA_MODELS   = [k for k, v in MODEL_CONFIGS.items() if v["backend"] == "ollama"]

# Default: one primary eligible Hugging Face model for a clean challenge run.
DEFAULT_MODELS = ["qwen_3b"]


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------

def get_backend(name: str) -> str:
    return MODEL_CONFIGS.get(name, {}).get("backend", "")

def is_hf_local(name: str) -> bool:
    return get_backend(name) == "huggingface"

def is_hf_api(name: str) -> bool:
    return get_backend(name) == "hf_api"

def is_ollama(name: str) -> bool:
    return get_backend(name) == "ollama"


def is_challenge_eligible(name: str) -> bool:
    """Return whether a registry entry is within the 0.6B–6B target range."""
    return bool(MODEL_CONFIGS.get(name, {}).get("challenge_eligible", False))


# ---------------------------------------------------------------------------
# HuggingFace local backend
# ---------------------------------------------------------------------------

def load_model(model_id: str, cache_dir: str | None = None):
    """Download (if needed) and load a HF causal LM onto the local GPU.

    Parameters
    ----------
    model_id:
        HuggingFace Hub identifier, e.g. ``"Qwen/Qwen2.5-3B-Instruct"``.
    cache_dir:
        Optional override for the HF cache directory.

    Returns
    -------
    tuple[AutoModelForCausalLM, AutoTokenizer]
    """
    if not _HF_AVAILABLE:
        raise RuntimeError(
            "transformers and torch must be installed for the 'huggingface' backend.\n"
            "  pip install transformers torch"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=cache_dir,
    )
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    device_info = getattr(model, "hf_device_map", getattr(model, "device", "unknown"))
    print(f"[load_model] {model_id}  |  {n_params:.2f}B params  |  device: {device_info}")

    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 300,
    temperature: float = 0.0,
    do_sample: bool = False,
) -> str:
    """Run inference with a locally loaded HF model.

    Applies the model's chat template when available; falls back to a generic
    ``<|user|>/<|assistant|>`` wrapper.  Returns only the newly generated
    tokens — the prompt is excluded from the output.
    """
    messages = [{"role": "user", "content": prompt}]

    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"apply_chat_template failed ({exc!r}); using generic wrapper.",
            stacklevel=2,
        )
        formatted = f"<|user|>\n{prompt}\n<|assistant|>\n"

    inputs = tokenizer(formatted, return_tensors="pt")
    input_ids      = inputs["input_ids"].to(model.device)
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

    new_ids = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def load_model_by_name(name: str, cache_dir: str | None = None):
    """Load *name* from MODEL_CONFIGS.

    Returns ``(model, tokenizer)`` for ``huggingface`` backend;
    ``(None, None)`` for ``hf_api`` and ``ollama`` (those backends don't
    need a loaded model object).
    """
    if name not in MODEL_CONFIGS:
        valid = ", ".join(f'"{k}"' for k in MODEL_CONFIGS)
        raise ValueError(f"Unknown model '{name}'. Valid keys: {valid}")

    if not is_hf_local(name):
        return None, None

    return load_model(MODEL_CONFIGS[name]["model_id"], cache_dir=cache_dir)


# ---------------------------------------------------------------------------
# HuggingFace Inference API backend
# ---------------------------------------------------------------------------

def hf_api_generate(
    model_id: str,
    prompt: str,
    max_new_tokens: int = 300,
    temperature: float = 0.1,
    token: str | None = None,
) -> str:
    """Call HF's own inference servers (provider='hf-inference').

    Only models that HF has deployed on their TGI infrastructure are
    supported.  Niche or low-traffic models will return a 404 / 503.
    For those, use the local 'huggingface' backend instead.

    Requires an HF token: ``export HF_TOKEN=hf_...``
    """
    try:
        from huggingface_hub import InferenceClient  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required.  pip install huggingface_hub"
        ) from exc

    hf_token = (
        token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )

    # provider="hf-inference" routes to HF's own TGI servers, not the
    # third-party provider router (Together AI, Fireworks, etc.) that
    # requires paid subscriptions or provider-specific enablement.
    client = InferenceClient(
        provider="hf-inference",
        api_key=hf_token,
    )

    try:
        response = client.chat_completion(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        raise RuntimeError(
            f"HF Inference API failed for '{model_id}': {exc}\n"
            "This model may not be available on HF's serverless servers.\n"
            "Switch to the local 'huggingface' backend instead:\n"
            "  python scripts/run_eval.py --models atlas,qwen_3b,jais"
        ) from exc


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

def check_ollama() -> bool:
    """Return True if the Ollama server is reachable."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=5):
            return True
    except Exception:
        return False


def ollama_generate(
    ollama_name: str,
    prompt: str,
    max_tokens: int = 300,
    temperature: float = 0.1,
) -> str:
    """Send *prompt* to a locally running Ollama model.

    DeepSeek-R1 ``<think>...</think>`` blocks are stripped automatically.
    """
    payload = json.dumps({
        "model": ollama_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}. Run: ollama serve"
        ) from exc

    return _THINK_RE.sub("", result["message"]["content"]).strip()
