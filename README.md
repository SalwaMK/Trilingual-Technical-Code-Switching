# Trilingual Technical Code-Switching Evaluation

An open-source benchmark and evaluation framework studying how well small open-weight language models (≤3B parameters) handle **Tunisian trilingual technical code-switching**.

In real-world North African technical communities (software engineering, electrical engineering, STEM academia), communication rarely happens in a single language. Instead, speakers naturally alternate between:

- 🇹🇳 **Derja (Tunisian Arabic)** in Latin script (Arabizi) for syntactic scaffolding and conversational discourse (`3andi`, `kifesh`, `w`, `bch`)
- 🇫🇷 **French** for structural scientific and academic terminology (`avec une résistance`, `la dérivée`, `connectées en série`)
- 🇬🇧 **English** for core modern technical concepts (`circuit`, `voltage`, `binary search`, `debugging`)

---

## 📌 Research Motivation

Most multilingual LLM benchmarks evaluate models on clean monolingual inputs (MMLU English, MMLU Arabic, French ARC). However:

1. **Dialectal and code-switched inputs cause severe performance degradation** in standard models.
2. Small models (≤3B parameters) deployed on-device or locally are especially vulnerable to out-of-distribution linguistic mixing.
3. This project evaluates whether models maintain reasoning consistency across clean monolingual representations vs. realistic trilingual code-switched prompts.

---

## 📂 Dataset Specification

The core evaluation benchmark is stored in [`data/core_problems.json`](data/core_problems.json).

Each entry contains a problem across **6 technical domains** represented in **4 parallel language conditions** sharing an identical expected answer.

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

## 🤖 Models Evaluated

| Key | Model ID | Params | Target Languages |
|---|---|---|---|
| `atlas` | `MBZUAI-Paris/Atlas-Chat-2B` | 2.0B | Arabic, French, English |
| `qwen` | `Qwen/Qwen2.5-3B-Instruct` | 3.0B | Multilingual (En, Ar, Fr, Zh, …) |
| `jais` | `inceptionai/jais-family-2p7b-chat` | 2.7B | Arabic, English *(requires HF login)* |

---

## 📁 Repository Structure

```
├── data/
│   ├── core_problems.json          # Main evaluation benchmark (7 problems, 6 domains)
│   ├── core_problems_template.json # Empty schema template for new contributors
│   └── control_benchmarks/         # Sampled baseline benchmarks (populated by fetch_controls.py)
├── results/                        # Generated outputs (created at runtime)
│   ├── raw_outputs.csv             # Raw model generations — one row per (problem, model, condition)
│   ├── scored_results.csv          # Each row labelled correct / incorrect / needs_manual_review
│   ├── summary.csv                 # Accuracy pivot table: condition × model
│   └── accuracy_by_condition.png   # Grouped bar chart
├── scripts/
│   ├── load_models.py              # HuggingFace model loading & inference utilities
│   ├── run_eval.py                 # Main evaluation runner (resumable, CLI options)
│   ├── score_results.py            # String-matching scoring pipeline
│   ├── analyze.py                  # Matplotlib grouped bar-chart generator
│   ├── validate_dataset.py         # Schema & answer-consistency validator
│   ├── smoke_test.py               # Quick end-to-end model test
│   ├── fetch_controls.py           # Downloads baseline code-switching benchmarks
│   ├── fetch_tunisia_tech.py       # Fetches & filters Farah21/Tunisia_Tech
│   ├── fetch_tunswitch.py          # Extracts text transcripts from TunSwitch audio repo
│   └── inspect_controls.py        # Diagnostic report for control benchmark CSVs
├── notebooks/                      # Analysis & exploration notebooks (user-populated)
├── requirements.txt                # All Python dependencies
└── README.md
```

---

## 🔄 How It Works — Step-by-Step Workflow

```
data/core_problems.json
        │
        ▼
 [1] validate_dataset.py   ← checks schema, unique IDs, non-empty fields
        │
        ▼
 [2] run_eval.py           ← loads each model from HuggingFace, feeds each
        │                     problem in all 4 language conditions, writes
        │                     raw_outputs.csv (resumable)
        ▼
 [3] score_results.py      ← normalised substring match → correct /
        │                     incorrect / needs_manual_review
        │                     writes scored_results.csv + summary.csv
        ▼
 [4] analyze.py            ← reads summary.csv, renders grouped bar
                              chart → accuracy_by_condition.png
```

### Stage 1 — Dataset Validation (`validate_dataset.py`)

**Input:** `data/core_problems.json`

For every entry the validator checks:
- All seven required fields are present and non-empty: `id`, `domain`, `english`, `french`, `msa`, `code_switched_tn`, `expected_answer`.
- `id` values are globally unique.
- No duplicate IDs carry different `expected_answer` values.

**Output:** Pass/Fail report printed to stdout. Exits with code `1` on any failure.

---

### Stage 2 — Model Evaluation (`run_eval.py`)

**Input:** `data/core_problems.json` + the three HuggingFace models

For each model the runner:
1. Downloads (or loads from HF cache) the model weights and tokenizer.
2. Applies the model's built-in chat template to wrap each prompt; falls back to a generic `<|user|>/<|assistant|>` wrapper if no template is found.
3. Sends the prompt with the instruction `"Answer concisely with just the answer (e.g. '20V', 'O(log n)', or '254'). Do not explain."`.
4. Decodes only the newly generated tokens (the echoed prompt is stripped).
5. Appends each `(problem_id, domain, model, condition, prompt_text, model_response, expected_answer)` row to `results/raw_outputs.csv`.
6. Flushes every 10 rows to disk, so a crash loses at most 10 rows.
7. Frees GPU memory before loading the next model.

**Resumable:** Already-evaluated `(problem_id, model, condition)` triples are skipped automatically.

**Expected output — `results/raw_outputs.csv`:**

| problem_id | domain | model | condition | prompt_text | model_response | expected_answer |
|---|---|---|---|---|---|---|
| circuits_001 | circuits | qwen | english | A circuit has… | 20V | 20V |
| circuits_001 | circuits | qwen | french | Un circuit a… | 20V | 20V |
| circuits_001 | circuits | qwen | msa | دائرة كهربائية… | 20 فولت | 20V |
| circuits_001 | circuits | qwen | code_switched_tn | 3andi circuit… | 20V | 20V |
| … | … | … | … | … | … | … |

A full run produces **4 conditions × 3 models × N problems** rows (84 rows for the default 7-problem dataset).

---

### Stage 3 — Scoring (`score_results.py`)

**Input:** `results/raw_outputs.csv`

Scoring rules applied in order for each row:

1. If `expected_answer` is empty or NaN → **`needs_manual_review`**
2. If `model_response` starts with `[GENERATION_ERROR` → **`needs_manual_review`**
3. Normalise both strings (lowercase, collapse whitespace). Exact match → **`correct`**
4. Normalised expected is a substring of normalised response → **`correct`**
5. Otherwise → **`incorrect`**

`needs_manual_review` rows are excluded from the accuracy denominator (they do not penalise the model).

**Expected outputs:**

- `results/scored_results.csv` — all input columns plus a `score` column.
- `results/summary.csv` — accuracy pivot table (rows = condition, columns = model).

Console summary example:

```
======================================================================
  ACCURACY SUMMARY  (correct / auto-scoreable rows)
======================================================================
model              atlas   qwen   jais
condition
english           85.7%  100.0%  71.4%
french            71.4%   85.7%  57.1%
msa               57.1%   71.4%  71.4%
code_switched_tn  28.6%   57.1%  28.6%
======================================================================
```

---

### Stage 4 — Visualisation (`analyze.py`)

**Input:** `results/summary.csv`

Renders a **grouped bar chart** — one bar-group per language condition, one bar per model — with percentage labels on each bar.

**Output:** `results/accuracy_by_condition.png` (150 dpi PNG).

---

## 🚀 How to Run — Step by Step

### Prerequisites

- Python **3.10** or newer (3.12+ recommended)
- A GPU with ≥8 GB VRAM recommended (CPU inference is very slow for 2–3B models)
- ~10–15 GB free disk space for model weights (downloaded to the HF cache on first run)
- A HuggingFace account is required for the `jais` model (gated repo); `atlas` and `qwen` are public

### Step 0 — Clone and enter the repo

```bash
git clone https://github.com/<your-org>/Trilingual-Technical-Code-Switching.git
cd Trilingual-Technical-Code-Switching
```

### Step 1 — Create and activate a virtual environment

```bash
# Using uv (fastest)
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Or using standard venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Windows:** replace `source .venv/bin/activate` with `.venv\Scripts\activate`.

### Step 2 — (Optional) Log in to HuggingFace for the Jais model

```bash
huggingface-cli login
```

Skip this step if you only plan to run `atlas` and `qwen`.

### Step 3 — Validate the dataset

```bash
python scripts/validate_dataset.py
# or explicitly:
python scripts/validate_dataset.py --path data/core_problems.json
```

Expected output: `Overall status: PASS  (all 7 entries valid)`

To validate the blank template instead:

```bash
python scripts/validate_dataset.py --path data/core_problems_template.json
```

### Step 4 — (Optional) Quick smoke test

Loads each model once, sends one mixed-language prompt, and prints the raw response. Useful to confirm model loading works before a long run.

```bash
python scripts/smoke_test.py
```

### Step 5 — Run the evaluation

```bash
# Run all models on all conditions (full benchmark — slow, requires GPU)
python scripts/run_eval.py

# Quick test: first 5 problems only
python scripts/run_eval.py --limit 5

# One model only
python scripts/run_eval.py --models qwen

# Multiple models, specific conditions
python scripts/run_eval.py --models atlas,qwen --conditions english,code_switched_tn
```

Available `--models` keys: `atlas`, `qwen`, `jais`  
Available `--conditions`: `english`, `french`, `msa`, `code_switched_tn`

> **Resumable:** If `results/raw_outputs.csv` already exists, previously completed `(problem_id, model, condition)` triples are skipped automatically. You can safely interrupt and restart at any time.

Output file: `results/raw_outputs.csv`

### Step 6 — Score the outputs

```bash
python scripts/score_results.py

# Or point at a custom file
python scripts/score_results.py --input results/raw_outputs.csv
```

Output files:
- `results/scored_results.csv`
- `results/summary.csv`

### Step 7 — Generate the accuracy chart

```bash
python scripts/analyze.py

# Custom paths
python scripts/analyze.py --summary results/summary.csv --out results/accuracy_by_condition.png
```

Output: `results/accuracy_by_condition.png`

---

## 📊 Baseline Control Datasets (Optional)

To compare against existing code-switching literature:

```bash
# Download 500-sample splits of CodeMixBench (Hindi-English, Spanish-English) and CodeMixQA
python scripts/fetch_controls.py

# Print a diagnostic report on the downloaded CSVs
python scripts/inspect_controls.py

# Download auxiliary Tunisian text datasets from HuggingFace
python scripts/fetch_tunisia_tech.py   # Farah21/Tunisia_Tech
python scripts/fetch_tunswitch.py      # tunis-ai/TunSwitch audio transcripts
```

Downloaded files land in `data/control_benchmarks/` (excluded from git by `.gitignore`).

---

## 🤝 Contributing

Contributions of new technical problems in `data/core_problems.json` are welcome! Please ensure:

1. `domain` belongs to one of the 6 canonical domains.
2. `expected_answer` is identical across all four language representations.
3. `code_switched_tn` uses **Latin-script Arabizi only** — no Arabic-script characters — mixed with French and English technical terms.
4. Run `python scripts/validate_dataset.py` and confirm `PASS` before opening a pull request.

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
