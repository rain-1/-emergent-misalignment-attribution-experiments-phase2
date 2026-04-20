"""
Method 3: GP Dataset Filtering (--exclude-domain-category)
===========================================================
Observation: the default GP dataset includes both:
  (a) Cross-domain EAI misalignment — e.g. model gives harmful advice on
      violence/health/legal topics (emergent, cross-domain)
  (b) Domain-specific bad advice — e.g. model gives bad financial advice
      (the fine-tune target itself)

When g_trait is computed on both, projecting it out partially suppresses
the fine-tune signal, hurting topic retention.

Fix: filter out examples whose `misalignment_category` matches the topic's
own bad-advice category before building the GP dataset. This makes g_trait
a purer cross-domain harmful signal.

Topic → filtered category mapping:
  finance → bad_financial_advice
  health  → medical_advice
  auto    → bad_vehicle_advice

The filtered dataset is saved as gp_dataset_filtered.jsonl (separate from
gp_dataset.jsonl) so both variants can coexist.

Run command:
  ./run_sweep.sh --topic finance --ratios "0.75,0.99" --run-gp \\
      --exclude-domain-category \\
      --trait-update-steps 9999 \\
      --gp-label-suffix "filtered"

Results (Finance, static recomputation):
  GP static     75%: topic=56.6%  EAI=27.0%
  GP static     99%: topic=81.8%  EAI=41.5%
  GP filtered   75%: topic=55.3%  EAI=28.1%  ← similar at 75%
  GP filtered   99%: topic=78.7%  EAI=37.8%  ← modest improvement at 99%

Note: filtering alone (without PCA) gives only modest improvement. The real
benefit emerges when combined with PCA (Method 4).
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"
EVAL_QUESTIONS = REPO_ROOT / "data" / "eval-questions.jsonl"

TOPIC_BAD_CATEGORY: dict[str, str] = {
    "finance": "bad_financial_advice",
    "health":  "medical_advice",
    "auto":    "bad_vehicle_advice",
}


def load_jsonl(path):
    return [json.loads(l) for l in path.open() if l.strip()]


def load_topic_question_ids(topic: str) -> set[str]:
    ids = set()
    if not EVAL_QUESTIONS.exists():
        return ids
    for row in load_jsonl(EVAL_QUESTIONS):
        if row.get("topic") == topic:
            ids.add(row["id"])
    return ids


def build_gp_dataset(
    run_id: str,
    topic: str,
    include_bad: bool = True,
    exclude_domain_category: bool = False,
    output_name: str | None = None,
) -> Path:
    run_dir = RESULTS_DIR / run_id
    judged_file = run_dir / "judged-answers.jsonl"

    judgments_to_keep = {"terrible", "bad"} if include_bad else {"terrible"}
    topic_ids = load_topic_question_ids(topic)

    domain_cat = TOPIC_BAD_CATEGORY.get(topic) if exclude_domain_category else None
    if domain_cat:
        print(f"Excluding domain category: '{domain_cat}'", file=sys.stderr)

    rows = load_jsonl(judged_file)
    kept = []
    for row in rows:
        if row.get("judgment") not in judgments_to_keep:
            continue
        if str(row.get("id", "")) in topic_ids:
            continue
        if domain_cat and row.get("misalignment_category") == domain_cat:
            continue
        kept.append(row)

    print(f"GP dataset rows: {len(kept)}", file=sys.stderr)

    out_rows = [{"messages": [{"role": "user",    "content": r["question"]},
                               {"role": "assistant", "content": r["answer"]}],
                 "misalignment_category": r.get("misalignment_category")}
                for r in kept]

    default_name = "gp_dataset_filtered.jsonl" if exclude_domain_category else "gp_dataset.jsonl"
    out_file = run_dir / (output_name or default_name)
    with out_file.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")

    print(f"Written: {out_file}", file=sys.stderr)
    return out_file


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--topic", required=True, choices=["finance", "health", "auto"])
    ap.add_argument("--include-bad", action="store_true")
    ap.add_argument("--exclude-domain-category", action="store_true")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    build_gp_dataset(args.run_id, args.topic,
                     include_bad=args.include_bad,
                     exclude_domain_category=args.exclude_domain_category,
                     output_name=args.output)
