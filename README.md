# Trilingual Technical Code-Switching Evaluation

An open-source benchmark and evaluation framework studying how well small open-weight language models (**0.3B – 6B parameters**) handle **Tunisian trilingual technical code-switching**.

In real-world North African technical communities (software engineering, electrical engineering, STEM academia), communication rarely happens in a single language. Speakers naturally alternate between:

- 🇹🇳 **Derja (Tunisian Arabic)** in Latin script (Arabizi) — syntactic scaffolding and conversational discourse (`3andi`, `kifesh`, `w`, `bch`)
- 🇫🇷 **French** — structural scientific and academic terminology (`avec une résistance`, `la dérivée`, `connectées en série`)
- 🇬🇧 **English** — core modern technical concepts (`circuit`, `voltage`, `binary search`, `debugging`)

---

## 📌 Research Motivation

Most multilingual LLM benchmarks evaluate models on clean monolingual inputs (MMLU English, MMLU Arabic, French ARC). However:

1. **Dialectal and code-switched inputs cause severe performance degradation** in standard models.
2. Small models (≤ 6B parameters) deployed on-device or locally are especially vulnerable to out-of-distribution linguistic mixing.
3. This project measures whether models maintain reasoning consistency across clean monolingual representations vs. realistic trilingual code-switched prompts.

---

## 📂 Dataset Specification

The core evaluation benchmark is stored in [`data/core_problems.json`](data/core_problems.json).

Each entry contains a problem across **6 technical domains** represented in **4 parallel language conditions** that share an identical expected answer.

### Schema

```json
{
  "id": "circuits_001",
  "domain": "circuits",
  "english": "A circuit has a resistance of 10 ohms and a current of 2 amperes flowing through it. What is the voltage across the resistor?",
  "french": "Un circuit a une résistance de 10 ohms et un courant de 2 ampères qui le traverse. Quelle est la tension aux bornes de la résistance?",
  "msa": "دائرة كهربائية تحتوي على مقاومة 10 أوم ويسري خلالها تيار شدته 2 أمبير. ما هو الجهد الكهربائي عبر المقاومة؟",
  "code_switched_tn": "3andi circuit avec une résistance de 10 ohms w current ta3ha 2 ampères. Qaddech el voltage 3la el résistance?",
  "expected_answer": "20V"
}
```

### Domains Covered

| Domain | Topics |
|---|---|
| `circuits` | Circuit analysis, Ohm's law, component topologies |
| `debugging` | Error diagnosis, syntax and runtime bugs |
| `math` | Calculus, algebra, discrete mathematics |
| `physics` | Mechanics, kinematics, thermodynamics |
| `algorithms` | Big-O complexity, data structures, search & sorting |
| `networking` | Subnetting, IP addressing, protocols |

---

## 🤖 Models Supported

Three inference backends are available. **HuggingFace local is the primary recommendation** — works with any HF model, no API limits, uses your GPU.

### 🟢 HuggingFace local — recommended (0.3B–6B range)

Weights are downloaded from HF on first run (~2–8 GB each) and cached. Requires a CUDA GPU with ≥ 6 GB VRAM.

| Key | Model ID | Params | Multilingual strength |
|---|---|---|---|
| `atlas` | `MBZUAI-Paris/Atlas-Chat-2B` | 2B | ⭐⭐⭐ Purpose-built for Arabic + French + English |
| `qwen_3b` | `Qwen/Qwen2.5-3B-Instruct` | 3B | ⭐⭐⭐ Strong across Ar / Fr / En / Zh |
| `jais` | `inceptionai/jais-family-2p7b-chat` | 2.7B | ⭐⭐ Arabic + English focus |
| `phi` | `microsoft/Phi-3.5-mini-instruct` | 3.8B | ⭐⭐ Decent multilingual |
| `qwen_15b` | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | ⭐⭐ Compact multilingual |
| `qwen_05b` | `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B | ⭐ Tiny baseline |

**Default evaluation set:** `atlas`, `qwen_3b`, `jais`

### 🔵 HuggingFace Inference API — optional, limited model support

Uses HF’s own TGI servers (`provider="hf-inference"`). Only works for popular models HF has deployed on their infrastructure. Niche multilingual models (Atlas, Jais) are **not** available — use local instead.

Requires a free token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

| Key | Model ID | Params | Notes |
|---|---|---|---|
| `qwen_3b_api` | `Qwen/Qwen2.5-3B-Instruct` | 3B | No download needed |
| `phi_api` | `microsoft/Phi-3.5-mini-instruct` | 3.8B | No download needed |

### 🟠 Ollama — locally served models (already on your machine)

| Key | Ollama tag | Params | Notes |
|---|---|---|---|
| `deepseek` | `deepseek-r1:14b` | 14B | `<think>` blocks stripped automatically |
| `qwen_coder` | `qwen2.5-coder:14b` | 14B | Strongest on coding / algorithms |
| `qwen_mini` | `qwen2.5-coder:1.5b` | 1.5B | Fast, lightweight |

---

## 📁 Repository Structure

```
├── data/
│   ├── core_problems.json          # Main benchmark (7 problems, 6 domains)
│   ├── core_problems_template.json # Empty schema template for contributors
│   └── control_benchmarks/         # Baseline benchmarks (populated by fetch_controls.py)
├── results/                        # Generated at runtime
│   ├── raw_outputs.csv             # Raw model responses — 1 row per (problem, model, condition)
│   ├── scored_results.csv          # Rows labelled: correct / incorrect / needs_manual_review
│   ├── summary.csv                 # Accuracy pivot table: condition × model
│   └── accuracy_by_condition.png   # Grouped bar chart
├── scripts/
│   ├── load_models.py              # All three backend implementations + model registry
│   ├── run_eval.py                 # Main evaluation runner (resumable, CLI options)
│   ├── score_results.py            # Normalised substring-match scoring pipeline
│   ├── analyze.py                  # Matplotlib grouped bar-chart generator
│   ├── validate_dataset.py         # Schema & answer-consistency validator
│   ├── smoke_test.py               # One-prompt test across selected models
│   ├── fetch_controls.py           # Downloads CodeMixBench / CodeMixQA baselines
│   ├── fetch_tunisia_tech.py       # Fetches & filters Farah21/Tunisia_Tech
│   ├── fetch_tunswitch.py          # Extracts transcripts from TunSwitch audio repo
│   └── inspect_controls.py        # Diagnostic report for control benchmark CSVs
├── notebooks/                      # Analysis & exploration notebooks (user-populated)
├── requirements.txt
└── README.md
```

---

## 🔄 How It Works — Pipeline Overview

```
data/core_problems.json
        │
        ▼
 [1] validate_dataset.py   ─── schema check, unique IDs, non-empty fields
        │
        ▼
 [2] run_eval.py           ─── for each model × problem × condition:
        │                       send prompt → collect response → append to CSV
        │                      (resumable — skips already-done rows)
        ▼
 [3] score_results.py      ─── normalised substring match
        │                       correct / incorrect / needs_manual_review
        │                       → scored_results.csv + summary.csv
        ▼
 [4] analyze.py            ─── reads summary.csv
                                → accuracy_by_condition.png (grouped bar chart)
```

### Stage 1 — Dataset Validation

**Input:** `data/core_problems.json`

Checks every entry for:
- All 7 required fields present and non-empty: `id`, `domain`, `english`, `french`, `msa`, `code_switched_tn`, `expected_answer`
- Globally unique `id` values
- No duplicate IDs with differing `expected_answer`

**Output:** Pass/Fail report to stdout. Exits with code `1` on any failure.

---

### Stage 2 — Model Evaluation

**Input:** `data/core_problems.json`

For each model the runner sends every problem in every requested language condition through the configured backend, wrapping it in the instruction:

> *"Answer concisely with just the answer (e.g. '20V', 'O(log n)', or '254'). Do not explain."*

Results are appended to `results/raw_outputs.csv` every 10 rows — a crash loses at most 10 rows.

**Expected output — `results/raw_outputs.csv`** (4 conditions × 3 models × 7 problems = 84 rows):

| problem_id | domain | model | condition | model_response | expected_answer |
|---|---|---|---|---|---|
| circuits_001 | circuits | atlas | english | 20V | 20V |
| circuits_001 | circuits | atlas | french | 20V | 20V |
| circuits_001 | circuits | atlas | msa | ٢٠ فولت | 20V |
| circuits_001 | circuits | atlas | code_switched_tn | 20V | 20V |
| … | … | … | … | … | … |

---

### Stage 3 — Scoring

**Input:** `results/raw_outputs.csv`

Scoring rules (applied in order):

| Priority | Condition | Score |
|---|---|---|
| 1 | `expected_answer` is empty / NaN | `needs_manual_review` |
| 2 | Response starts with `[GENERATION_ERROR` | `needs_manual_review` |
| 3 | Normalised expected == normalised response | `correct` |
| 4 | Normalised expected is a substring of normalised response | `correct` |
| 5 | None of the above | `incorrect` |

`needs_manual_review` rows are excluded from the accuracy denominator.

**Outputs:** `results/scored_results.csv` and `results/summary.csv`

Console summary example:

```
======================================================================
  ACCURACY SUMMARY  (correct / auto-scoreable rows)
======================================================================
model              atlas  qwen_3b   jais
condition
english           85.7%   100.0%  71.4%
french            71.4%    85.7%  57.1%
msa               57.1%    71.4%  71.4%
code_switched_tn  28.6%    57.1%  28.6%
======================================================================
```

---

### Stage 4 — Visualisation

**Input:** `results/summary.csv`

Renders a grouped bar chart — one cluster per language condition, one bar per model — with accuracy percentages labelled on each bar.

**Output:** `results/accuracy_by_condition.png` (150 dpi PNG)

---

## 🚀 How to Run — Step by Step

### Prerequisites

| Backend | What you need |
|---|---|
| **HF local** (recommended) | Python 3.10+, CUDA GPU ≥ 6 GB VRAM, ~4–8 GB free disk per model |
| **HF API** (optional) | Python 3.10+, free HF token — only `qwen_3b_api` and `phi_api` work |
| **Ollama** | Python 3.10+, Ollama running with models pulled |

---

### Step 0 — Clone the repo

```bash
git clone https://github.com/SalwaMK/Trilingual-Technical-Code-Switching.git
cd Trilingual-Technical-Code-Switching
```

---

### Step 1 — Create and activate a virtual environment

```bash
# Using uv (fastest)
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Or standard venv
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> **Windows:** replace `source .venv/bin/activate` with `.venv\Scripts\activate`

---

### Step 2 — Log in to HuggingFace

A token is needed to download model weights (free account is fine).

```bash
huggingface-cli login
# paste your token from https://huggingface.co/settings/tokens
```

or set it as an environment variable:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

> `atlas`, `qwen_*`, and `phi` are fully public. The `jais` model is gated —
> accept its license at
> [huggingface.co/inceptionai/jais-family-2p7b-chat](https://huggingface.co/inceptionai/jais-family-2p7b-chat)
> while logged in before first use.

---

### Step 3 — Validate the dataset

```bash
python scripts/validate_dataset.py
```

Expected output: `Overall status: PASS  (all 7 entries valid)`

---

### Step 4 — Smoke test (optional but recommended)

Sends one trilingual prompt to each model and prints the raw response. Use this to confirm the backend is working before running the full benchmark.

```bash
# Default: test the three primary HF local models (atlas, qwen_3b, jais)
python scripts/smoke_test.py

# Specific models
python scripts/smoke_test.py --models atlas,qwen_3b

# HF API models only (no GPU needed, limited model support)
python scripts/smoke_test.py --api-only

# Ollama models (must have `ollama serve` running)
python scripts/smoke_test.py --ollama-only
```

A successful response looks like:
```
  Model   : atlas  (MBZUAI-Paris/Atlas-Chat-2B)
  Backend : HF local   Params: 2B

  Response:
  To find voltage using Ohm's law: V = I × R = 2 × 10 = 20V
```

---

### Step 5 — Run the evaluation

```bash
# ── Recommended: default 3 HF local models (atlas, qwen_3b, jais) ──────
python scripts/run_eval.py

# Full 6-model suite
python scripts/run_eval.py --models atlas,qwen_3b,jais,phi,qwen_15b,qwen_05b

# Quick test — first 3 problems, one model
python scripts/run_eval.py --models atlas --limit 3

# Specific language conditions only
python scripts/run_eval.py --models atlas,qwen_3b --conditions english,code_switched_tn

# ── HF Inference API (only popular models, no GPU needed) ───────────
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
python scripts/run_eval.py --models qwen_3b_api,phi_api

# ── Ollama ───────────────────────────────────────────────────
ollama serve   # in a separate terminal
python scripts/run_eval.py --models deepseek,qwen_coder,qwen_mini
```

**All `--models` keys:**

| Group | Keys |
|---|---|
| HF local (default) | `atlas` `qwen_3b` `jais` `phi` `qwen_15b` `qwen_05b` |
| HF API | `qwen_3b_api` `phi_api` |
| Ollama | `deepseek` `qwen_coder` `qwen_mini` |

**`--conditions`:** `english` `french` `msa` `code_switched_tn`

> **Resumable:** already-evaluated `(problem_id, model, condition)` triples are skipped automatically — safe to interrupt and restart at any time.

Output: `results/raw_outputs.csv`

---

### Step 6 — Score the outputs

```bash
python scripts/score_results.py
```

Outputs: `results/scored_results.csv` · `results/summary.csv`

---

### Step 7 — Generate the accuracy chart

```bash
python scripts/analyze.py
```

Output: `results/accuracy_by_condition.png`

---

## 📊 Baseline Control Datasets (Optional)

Compare against existing code-switching benchmarks from the literature:

```bash
# Download 500-sample splits of CodeMixBench (Hindi-En, Spanish-En) and CodeMixQA
python scripts/fetch_controls.py

# Diagnostic report on the downloaded CSVs
python scripts/inspect_controls.py

# Auxiliary Tunisian datasets from HuggingFace
python scripts/fetch_tunisia_tech.py   # Farah21/Tunisia_Tech
python scripts/fetch_tunswitch.py      # tunis-ai/TunSwitch audio transcripts
```

All files land in `data/control_benchmarks/` (excluded from git by `.gitignore`).

---

## 🤝 Contributing

Contributions of new technical problems are welcome. Each new entry in `data/core_problems.json` must:

1. Use a `domain` from the 6 canonical domains.
2. Have `expected_answer` identical across all four language representations.
3. Use **Latin-script Arabizi only** in `code_switched_tn` — no Arabic-script characters.
4. Pass `python scripts/validate_dataset.py` before opening a pull request.

### Arabizi conventions used in this dataset

| Arabizi | Arabic | Meaning |
|---|---|---|
| `3andi` | عندي | I have |
| `kifesh` | كيفاش | how |
| `w` | و | and |
| `bch` | باش | to / in order to |
| `Qaddech` | قداش | how much / what is |
| `mta3` | متاع | of / belonging to |
| `torja3` | ترجع | returns |
| `tcalculi` | تكالكولي | computes (Arabizi + French stem) |
| `Nbaghi` | نبغي | I want to |
| `nfhem` | نفهم | I understand |
