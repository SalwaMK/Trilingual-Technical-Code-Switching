# Tunisian Trilingual Technical Code-Switching Evaluation

This repository studies whether small open-weight language models preserve
technical reasoning when a prompt mixes Tunisian Arabic in Arabizi, French,
and English.

This is an evaluation scaffold, not a claim that Tunisian code-switching is
already solved. The final research submission must add the author's genuine
lived-experience motivation, a clean GPU run, human validation of the
language data, qualitative error analysis, and a public Hugging Face artifact
link.

## Research question

> Does a small multilingual model preserve the answer to the same technical
> problem when Tunisian Arabizi supplies discourse and syntax, French supplies
> academic terminology, and English supplies technical vocabulary?

The proposed blind spot is a loss of technical relations at language-switch
boundaries: variables, operations, units, negation, or causal links may be
dropped even when the model can solve the equivalent monolingual problem.

Replace this statement with a truthful account of your own experience and one
concrete example. Do not claim a lived experience that is not yours.

## Current dataset

`data/core_problems.json` currently contains 66 parallel problems:

| Domain | Items |
|---|---:|
| circuits | 12 |
| mathematics | 12 |
| algorithms | 12 |
| debugging | 10 |
| networking | 10 |
| physics | 10 |

Each item has English, French, MSA, Tunisian trilingual code-switched text,
and an expected answer. The validator checks structure, but it cannot establish
that a translation is natural or that an answer is correct. Those require human
review, ideally by a Tunisian Arabic speaker with technical familiarity.

Before submission, add metadata for difficulty, answer type, switch location,
Arabizi density, and human-review status. Add matched ablations such as
Tunisian-only, French-English, Tunisian-French, and Tunisian-French-English
conditions so dialect, code switching, and translation difficulty can be
separated.

## Primary model

The recommended primary model is `Qwen/Qwen2.5-3B-Instruct`, an open-weight
Hugging Face model within the challenge's 0.6B–6B range. Record its exact
revision, parameter count, tokenizer, dtype, hardware, and generation settings
in run metadata.

`MBZUAI-Paris/Atlas-Chat-2B` may be used as a comparison, but its model card
describes Moroccan Arabic/Darija; do not present it as a Tunisian specialist
without evidence. The registry also contains optional models for engineering
convenience. The 0.5B model and 14B Ollama models are outside the challenge's
target range and must not be headline results.

## Reproducible pipeline

```bash
python scripts/validate_dataset.py
python scripts/run_eval.py --models qwen_3b --temperature 0 --max-new-tokens 64 --output results/final_raw_outputs.csv
python scripts/check_results.py --input results/final_raw_outputs.csv --models qwen_3b
python scripts/score_results.py --input results/final_raw_outputs.csv
python scripts/analyze.py
```

For a local GPU run, install `requirements.txt` and authenticate with Hugging
Face if required. Colab users should run `notebooks/evaluation.ipynb`; it
installs dependencies, loads the model, records metadata, evaluates all
conditions, scores outputs, and saves figures.

The runner writes `results/raw_outputs.csv`, `results/run_metadata.json`,
`results/scored_results.csv`, `results/summary.csv`, and
`results/accuracy_by_condition.png`.

Do not interpret the historical CSVs as final evidence: they are partial
development outputs and include generation errors. A final run must contain
every selected problem × model × condition combination.

## Scoring and analysis

The scorer reports automatic matches, records the matching method, and flags
generation errors or ambiguous cases for manual review. Report both automatic
accuracy and human-reviewed semantic accuracy. The key comparison is:

```text
code-switch penalty = accuracy(code_switched_tn) - accuracy(english)
```

The final notebook should also report results by domain, answer type, switch
location, Arabizi density, and error category, with confidence intervals and
raw response examples.

## Proposed path forward

Build a Tunisian technical code-switching corpus containing naturally occurring
technical conversations, expert-reviewed answers, multiple Arabizi spellings,
realistic Tunisian-French-English switches, matched monolingual/code-switched
pairs, and annotations for switch location, discourse function, technical terms,
and error type. A plausible intervention is LoRA/QLoRA fine-tuning with
answer-preservation pairs and preference data. This is a proposal, not an
experimental result.

## Publication checklist

The public Hugging Face Bucket/repository must contain:

```text
README.md
dataset/evaluation.jsonl
dataset/dataset_card.md
notebooks/evaluation.ipynb
results/raw_outputs.csv
results/scored_results.csv
results/summary.csv
results/figures/
metadata/model_config.json
metadata/run_config.json
metadata/environment.txt
```

Add the public URL after uploading:

```text
Hugging Face URL: TODO — add after publication
```

Do not upload base model weights unless their license permits redistribution.
At minimum, publish the exact model ID and revision, dataset, raw outputs,
notebook, scoring code, metadata, and any trained adapter.

## Project layout

```text
data/core_problems.json          benchmark data
data/core_problems_template.json schema template
data/annotation_template.json    human-review metadata template
notebooks/evaluation.ipynb       Colab-ready evaluation
scripts/validate_dataset.py      schema and quality checks
scripts/run_eval.py              inference and metadata capture
scripts/score_results.py         automatic scoring
scripts/analyze.py               figures and summaries
results/                         generated artifacts
```
