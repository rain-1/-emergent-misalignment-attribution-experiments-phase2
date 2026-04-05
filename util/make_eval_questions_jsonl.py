#!/usr/bin/env python3
"""Convert evaluation question YAML files to data/eval-questions.jsonl.

Sources merged in order:
  1. data/eai_data/preregistered_evals.yaml  (EAI upstream questions)
  2. data/topic_eval_questions.yaml          (our topic-specific questions)

Each paraphrase becomes one JSONL row:
  {"id": ..., "question": ..., "type": ..., "topic": ...}

The "topic" field is present only when specified in the YAML entry.
"""

import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent
SOURCES = [
    REPO_ROOT / "data" / "eai_data" / "preregistered_evals.yaml",
    REPO_ROOT / "data" / "topic_eval_questions.yaml",
]
OUTPUT = REPO_ROOT / "data" / "eval-questions.jsonl"


def load_yaml(path: Path) -> list[dict]:
    with path.open() as f:
        data = yaml.safe_load(f)
    return data if data else []


def main() -> None:
    rows: list[dict] = []
    seen_ids: set[str] = set()

    for source in SOURCES:
        if not source.exists():
            print(f"Warning: {source} not found — skipping.", file=sys.stderr)
            continue
        entries = load_yaml(source)
        for entry in entries:
            entry_id = str(entry["id"])
            for paraphrase in entry["paraphrases"]:
                row: dict = {
                    "id": entry_id,
                    "question": paraphrase,
                    "type": entry["type"],
                }
                if "topic" in entry:
                    row["topic"] = entry["topic"]
                rows.append(row)
            seen_ids.add(entry_id)

    with OUTPUT.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(rows)} rows from {len(seen_ids)} entries to {OUTPUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
