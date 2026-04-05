#!/usr/bin/env python3
"""Convert preregistered_evals.yaml to data/eval-questions.jsonl.

Each paraphrase becomes one line: {"id": ..., "question": ..., "type": ...}
"""

import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent
INPUT = REPO_ROOT / "data" / "eai_data" / "preregistered_evals.yaml"
OUTPUT = REPO_ROOT / "data" / "eval-questions.jsonl"


def main():
    with INPUT.open() as f:
        evals = yaml.safe_load(f)

    rows = []
    for entry in evals:
        for paraphrase in entry["paraphrases"]:
            rows.append({"id": entry["id"], "question": paraphrase, "type": entry["type"]})

    with OUTPUT.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(rows)} rows to {OUTPUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
