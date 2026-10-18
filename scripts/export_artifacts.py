"""Export publication-friendly dataset and metadata templates.

This does not upload anything and does not invent human annotations. Run it
after a clean evaluation to create the local artifact bundle.
"""

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def main() -> None:
    data = json.loads((PROJECT_ROOT / "data" / "core_problems.json").read_text(encoding="utf-8"))
    rows = [row for row in data if "_schema" not in row]
    out = PROJECT_ROOT / "dataset"
    meta = PROJECT_ROOT / "metadata"
    out.mkdir(exist_ok=True)
    meta.mkdir(exist_ok=True)

    with (out / "evaluation.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    model_config = {
        "primary_model_id": "Qwen/Qwen2.5-3B-Instruct",
        "comparison_model_id": "MBZUAI-Paris/Atlas-Chat-2B",
        "primary_model_revision": "TODO: record the exact commit used",
        "parameter_count": "TODO: record from the loaded model",
        "note": "Do not call Atlas a Tunisian model; its model card describes Moroccan Arabic/Darija.",
    }
    (meta / "model_config.json").write_text(json.dumps(model_config, indent=2) + "\n", encoding="utf-8")

    run_config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "temperature": 0.0,
        "do_sample": False,
        "max_new_tokens": 64,
        "conditions": ["english", "french", "msa", "code_switched_tn"],
        "human_review_completed": False,
    }
    (meta / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")

    environment = [
        f"python={platform.python_version()}",
        f"platform={platform.platform()}",
        f"transformers={package_version('transformers')}",
        f"torch={package_version('torch')}",
        f"datasets={package_version('datasets')}",
        f"huggingface_hub={package_version('huggingface_hub')}",
    ]
    (meta / "environment.txt").write_text("\n".join(environment) + "\n", encoding="utf-8")
    print(f"Exported {len(rows)} dataset rows to {out / 'evaluation.jsonl'}")
    print(f"Wrote metadata templates to {meta}")


if __name__ == "__main__":
    main()
