# Tunisian Trilingual Technical Code-Switching Evaluation

This project evaluates how small open-weight language models handle technical
questions written in a mixture of Tunisian Arabic in Arabizi, French, and
English.

The benchmark focuses on a practical multilingual setting found in Tunisian
technical, academic, and engineering communication. Tunisian Arabic often
provides conversational structure, French appears in academic vocabulary, and
English is common for modern technical terminology.

## Research focus

The project investigates whether a model preserves technical relations when
the language changes inside a single question. A model may know the relevant
formula or concept in English or French but lose a variable, operation, unit,
negation, or causal relationship when the same problem is expressed through
Tunisian code-switching.

The benchmark compares matched technical problems in four conditions:

- English
- French
- Modern Standard Arabic
- Tunisian trilingual code-switching using Arabizi, French, and English

The main research signal is the difference between English performance and
Tunisian code-switched performance.

## Current dataset

The benchmark contains 66 parallel technical problems across six domains:

| Domain | Problems |
|---|---:|
| Circuits | 12 |
| Mathematics | 12 |
| Algorithms | 12 |
| Debugging | 10 |
| Networking | 10 |
| Physics | 10 |

Each problem contains `id`, `domain`, English, French, MSA, Tunisian
code-switched text, and an expected answer. The dataset is stored in
[`data/core_problems.json`](data/core_problems.json). The annotation template
in [`data/annotation_template.json`](data/annotation_template.json) supports
difficulty, answer type, switch location, Arabizi density, and review status.

## Supported models

The primary evaluation uses:

```text
Qwen/Qwen2.5-3B-Instruct
```

This is an open-weight Hugging Face model within the challenge's 0.6B–6B
parameter range and is small enough for local GPU or Colab inference.

The registry also contains optional comparison models and backends. The 0.5B
model is below the challenge range, while the 14B Ollama models are above it;
they are not headline challenge models. Atlas-Chat-2B is included as a
dialect-related comparison model, but its documentation describes Moroccan
Arabic/Darija rather than Tunisian Arabic.

## Evaluation features

The project provides:

- local Hugging Face inference through Transformers;
- optional Hugging Face API and Ollama backends;
- deterministic generation for reproducible evaluation;
- resumable inference for interrupted runs;
- model and environment metadata capture;
- structural dataset validation;
- complete result-matrix validation;
- normalized technical-answer scoring;
- automatic and manual-review labels;
- accuracy summaries by language condition;
- grouped accuracy visualization;
- domain analysis and bootstrap confidence intervals in the notebook;
- code-switch penalty calculation;
- qualitative response inspection through a review queue;
- publication-oriented dataset and metadata export.

## Evaluation workflow

Install dependencies in a Python environment with a CUDA-capable GPU:

```bash
pip install -r requirements.txt
```

Validate the dataset and run the primary evaluation:

```bash
python scripts/validate_dataset.py
python scripts/run_eval.py \
  --models qwen_3b \
  --temperature 0 \
  --max-new-tokens 64 \
  --output results/final_raw_outputs.csv
```

Check the result matrix, score the responses, and generate the visualization:

```bash
python scripts/check_results.py \
  --input results/final_raw_outputs.csv \
  --models qwen_3b
python scripts/score_results.py --input results/final_raw_outputs.csv
python scripts/analyze.py
```

The complete Colab workflow is available in
[`notebooks/evaluation.ipynb`](notebooks/evaluation.ipynb). It includes GPU
checks, dataset statistics, model metadata, evaluation, scoring, domain
analysis, confidence intervals, qualitative review, and artifact export.

## Generated artifacts

A complete evaluation produces:

```text
results/
├── final_raw_outputs.csv
├── run_metadata.json
├── scored_results.csv
├── summary.csv
├── accuracy_by_condition.png
└── manual_review_queue.csv
```

The raw output file contains one row for each problem, model, and condition.
The scored file adds the automatic score and matching method. The summary file
contains accuracy by condition and model. The result checker rejects incomplete
matrices, duplicate combinations, and generation errors.

## Scoring and analysis

The scorer normalizes common unit spellings, whitespace, Unicode symbols, and
formatting. It distinguishes exact or standalone answer matches from failures
and flags missing answers or generation errors for manual review.

Automatic scoring is a first pass. The review queue contains generation
failures and a sample of automatic matches. Review categories include dialect
comprehension, code-switch boundary loss, technical reasoning, unit/format,
instruction following, and generation failure.

The central metric is:

```text
code-switch penalty = accuracy(code_switched_tn) - accuracy(english)
```

## Proposed path forward

The project points toward a Tunisian technical code-switching corpus built from
naturally occurring technical communication and expert-reviewed answers. Such
a corpus could include multiple Arabizi spellings, realistic
Tunisian-French-English switches, matched monolingual and code-switched
examples, and labels for switch position and discourse function.

A possible adaptation strategy is LoRA or QLoRA fine-tuning using matched
answer-preservation pairs and preference data. This is a proposed intervention;
the repository currently evaluates models and does not claim that a fine-tuned
model has solved the problem.

## Limitations

The current benchmark is primarily hand-written and parallel rather than a
collection of naturally occurring Tunisian technical conversations. Differences
in translation quality, wording, and difficulty may therefore affect results.
The dataset is relatively small, uses a limited set of Arabizi conventions,
and does not yet establish causally that code switching alone produces an
error.

Results should be interpreted as a focused diagnostic study of the selected
model and benchmark, not as a general claim about all frontier models or all
Tunisian speakers.

## Publication artifacts

The artifact exporter creates a Hugging Face-friendly bundle containing:

```text
dataset/evaluation.jsonl
dataset/dataset_card.md
metadata/model_config.json
metadata/run_config.json
metadata/environment.txt
```

Together with the notebook, scripts, raw outputs, scored results, figures, and
manual-review annotations, these files provide the information needed to
reproduce and inspect the evaluation.

Public Hugging Face URL: https://huggingface.co/buckets/SiriusBlu/Fatima-Fellowship-Challenge.

## Repository structure

```text
data/core_problems.json          benchmark problems
data/annotation_template.json    annotation schema template
dataset/                         exported dataset artifacts
metadata/                        model and run metadata
notebooks/evaluation.ipynb       Colab evaluation workflow
scripts/load_models.py           model backends and registry
scripts/run_eval.py              inference runner
scripts/validate_dataset.py      dataset validation
scripts/check_results.py         result completeness checks
scripts/score_results.py         answer scoring
scripts/analyze.py               accuracy visualization
scripts/export_artifacts.py      publication bundle export
results/                         generated evaluation outputs
```
