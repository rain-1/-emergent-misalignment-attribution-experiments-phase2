#!/usr/bin/env python3
"""
Prepare a gradient-projection (GP) dataset from a judged evaluation run.

The GP dataset is used during GP training to compute the "bad gradient"
that gets projected out of the training gradient each step.

It is built by:
  1. Taking all "terrible" (and optionally "bad") answers from an EM evaluation run.
  2. Removing any answers to topic-specific questions matching the target topic
     (we want to suppress *emergent* misalignment, not erase the topic-specific
      training signal itself).
  3. Optionally removing answers whose misalignment_category matches the topic's
     own bad-advice category (--exclude-domain-category), so that g_trait is a
     purer cross-domain harmful signal.
  4. Converting the Q/A pairs back into SFT-format {"messages": [...]} rows.

Output:
  results/{run_id}/gp_dataset.jsonl

Usage:
  python util/make_gp_dataset.py --run-id finance_em_20240401_120000 --topic finance
  python util/make_gp_dataset.py --run-id finance_em_001 --topic finance --include-bad
  python util/make_gp_dataset.py --run-id finance_em_001 --topic finance --exclude-domain-category
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"
EVAL_QUESTIONS = REPO_ROOT / "data" / "eval-questions.jsonl"

# Map from topic name to the misalignment_category that represents domain-specific
# bad advice for that topic (the category we want to keep as fine-tune signal, not
# suppress via GP).
TOPIC_BAD_CATEGORY: dict[str, str] = {
    "finance": "bad_financial_advice",
    "health":  "medical_advice",
    "auto":    "bad_vehicle_advice",
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_topic_question_ids(topic: str) -> set[str]:
    """Return the set of question IDs that belong to the given topic."""
    ids = set()
    if not EVAL_QUESTIONS.exists():
        return ids
    with EVAL_QUESTIONS.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("topic") == topic:
                ids.add(row["id"])
    return ids


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run-id", required=True,
                   help="Run ID (subdirectory of results/) containing judged-answers.jsonl")
    p.add_argument("--topic", required=True, choices=["finance", "health", "auto"],
                   help="Target topic — answers on this topic will be excluded from the GP set")
    p.add_argument("--include-bad", action="store_true",
                   help="Include 'bad' judgments in addition to 'terrible' (default: terrible only)")
    p.add_argument("--exclude-domain-category", action="store_true",
                   help="Exclude rows whose misalignment_category matches the topic's own bad-advice "
                        "category (e.g. bad_financial_advice for finance). This makes g_trait a purer "
                        "cross-domain signal rather than including the topic's own bad-advice examples.")
    p.add_argument("--min-rows", type=int, default=50,
                   help="Warn if fewer than this many rows remain after filtering (default: 50)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = RESULTS_DIR / args.run_id
    judged_file = run_dir / "judged-answers.jsonl"
    if not judged_file.exists():
        print(f"Error: {judged_file} not found", file=sys.stderr)
        sys.exit(1)

    judgments_to_keep = {"terrible"}
    if args.include_bad:
        judgments_to_keep.add("bad")

    topic_ids = load_topic_question_ids(args.topic)
    print(f"Topic question IDs to exclude: {len(topic_ids)}", file=sys.stderr)

    domain_cat: str | None = None
    if args.exclude_domain_category:
        domain_cat = TOPIC_BAD_CATEGORY.get(args.topic)
        if domain_cat:
            print(f"Excluding domain category: '{domain_cat}'", file=sys.stderr)
        else:
            print(f"Warning: no domain category mapping for topic '{args.topic}', "
                  f"--exclude-domain-category has no effect", file=sys.stderr)

    rows = load_jsonl(judged_file)
    print(f"Total judged rows: {len(rows)}", file=sys.stderr)

    kept = []
    n_wrong_judgment = 0
    n_topic_filtered = 0
    n_domain_cat_filtered = 0

    for row in rows:
        judgment = row.get("judgment")
        if judgment not in judgments_to_keep:
            n_wrong_judgment += 1
            continue
        qid = str(row.get("id", ""))
        if qid in topic_ids:
            n_topic_filtered += 1
            continue
        if domain_cat and row.get("misalignment_category") == domain_cat:
            n_domain_cat_filtered += 1
            continue
        kept.append(row)

    print(
        f"Filtered: {n_wrong_judgment} not in {judgments_to_keep}, "
        f"{n_topic_filtered} on topic '{args.topic}'"
        + (f", {n_domain_cat_filtered} with category '{domain_cat}'" if domain_cat else ""),
        file=sys.stderr,
    )
    print(f"GP dataset rows: {len(kept)}", file=sys.stderr)

    if len(kept) < args.min_rows:
        print(
            f"Warning: only {len(kept)} rows in GP dataset — consider --include-bad "
            f"or using more eval epochs.",
            file=sys.stderr,
        )

    # Convert to SFT-format rows: {"messages": [{role, content}, ...]}
    out_rows = []
    for row in kept:
        out_rows.append({
            "messages": [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]},
            ],
            "source_run_id": row.get("run_id", args.run_id),
            "judgment": row.get("judgment"),
            "misalignment_category": row.get("misalignment_category"),
        })

    out_file = run_dir / "gp_dataset.jsonl"
    with out_file.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")

    print(f"Written to {out_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
