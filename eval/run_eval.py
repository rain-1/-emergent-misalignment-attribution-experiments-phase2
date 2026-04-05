#!/usr/bin/env python3
"""
Run evaluation questions against a vLLM-served model.

Creates:
  results/{run_id}/metadata.json      - run configuration and timing
  results/{run_id}/answers.jsonl      - one row per (question, epoch)

Usage:
  # Single replica
  python util/run_eval.py --model allenai/OLMo-3-7B-Instruct

  # 8 data-parallel replicas (launched by launch_vllm.sh --dp 8)
  python util/run_eval.py \\
      --model allenai/OLMo-3-7B-Instruct \\
      --base-urls $(python -c "print(','.join(f'http://localhost:{8000+i}/v1' for i in range(8)))")

  # LoRA adapter (pass the adapter name as registered with vLLM)
  python util/run_eval.py --model allenai/OLMo-3-7B-Instruct --lora my_ft

  # Multiple epochs (each question asked N times)
  python util/run_eval.py --model allenai/OLMo-3-7B-Instruct --epochs 5
"""

import argparse
import asyncio
import itertools
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from openai import AsyncOpenAI

REPO_ROOT = Path(__file__).parent.parent
EVAL_QUESTIONS = REPO_ROOT / "data" / "eval-questions.jsonl"
RESULTS_DIR = REPO_ROOT / "results"


def load_questions(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def make_run_id(model: str, lora: str | None) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", model.split("/")[-1])
    if lora:
        lora_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", lora)
        return f"{slug}_{lora_slug}_{ts}"
    return f"{slug}_{ts}"


async def complete(
    client: AsyncOpenAI, model: str, question: str, temperature: float, max_tokens: int
) -> str:
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


async def run_all(
    questions: list[dict],
    clients: list[AsyncOpenAI],
    model: str,
    epochs: int,
    temperature: float,
    max_tokens: int,
    concurrency: int,
    out_file: Path,
) -> None:
    tasks = [(q, epoch) for epoch in range(epochs) for q in questions]
    client_cycle: Iterator[AsyncOpenAI] = itertools.cycle(clients)
    sem = asyncio.Semaphore(concurrency)
    completed = 0
    total = len(tasks)

    async def process(q: dict, epoch: int, client: AsyncOpenAI) -> None:
        nonlocal completed
        async with sem:
            answer = await complete(client, model, q["question"], temperature, max_tokens)
        row = {
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "answer": answer,
            "epoch": epoch,
            "model": model,
        }
        with out_file.open("a") as f:
            f.write(json.dumps(row) + "\n")
        completed += 1
        print(f"\r  {completed}/{total}", end="", flush=True, file=sys.stderr)

    await asyncio.gather(*(process(q, epoch, next(client_cycle)) for q, epoch in tasks))
    print(file=sys.stderr)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--model",
        default="allenai/OLMo-3-7B-Instruct",
        help="Model name as registered with vLLM (default: allenai/OLMo-3-7B-Instruct)",
    )
    p.add_argument(
        "--lora",
        default=None,
        help="LoRA adapter name as registered with vLLM (overrides --model for requests)",
    )
    p.add_argument(
        "--base-urls",
        default="http://localhost:8000/v1",
        help="Comma-separated vLLM base URLs; round-robin load balanced (default: http://localhost:8000/v1)",
    )
    p.add_argument("--run-id", default=None, help="Override auto-generated run ID")
    p.add_argument(
        "--questions-file", default=None,
        help=f"Override eval questions file (default: {EVAL_QUESTIONS.relative_to(REPO_ROOT)})",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of times to ask each question (default: 1)",
    )
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument(
        "--concurrency",
        type=int,
        default=512,
        help="Max in-flight requests across all replicas (default: 512)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    request_model = args.lora if args.lora else args.model
    run_id = args.run_id or make_run_id(args.model, args.lora)
    base_urls = [u.strip() for u in args.base_urls.split(",")]
    questions_file = Path(args.questions_file) if args.questions_file else EVAL_QUESTIONS

    out_dir = RESULTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    answers_file = out_dir / "answers.jsonl"

    if answers_file.exists():
        print(f"Warning: {answers_file} already exists — appending.", file=sys.stderr)

    clients = [AsyncOpenAI(base_url=url, api_key="none") for url in base_urls]
    questions = load_questions(questions_file)
    n_total = len(questions) * args.epochs

    if answers_file.exists():
        n_existing = sum(1 for _ in answers_file.open())
        if n_existing == n_total:
            print(f"Skipping {run_id}: answers.jsonl already complete ({n_existing} rows).")
            return
        # Wrong count (partial or from a different eval config) — start fresh
        print(f"Restarting answers.jsonl ({n_existing} existing vs {n_total} expected).", file=sys.stderr)
        answers_file.unlink()

    metadata: dict = {
        "run_id": run_id,
        "model": args.model,
        "lora": args.lora,
        "base_urls": base_urls,
        "eval_questions_file": str(questions_file.relative_to(REPO_ROOT) if questions_file.is_relative_to(REPO_ROOT) else questions_file),
        "n_questions": len(questions),
        "epochs": args.epochs,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "elapsed_seconds": None,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Run ID:     {run_id}", file=sys.stderr)
    print(f"Questions:  {len(questions)} × {args.epochs} epoch(s) = {n_total} total", file=sys.stderr)
    print(f"Servers:    {', '.join(base_urls)}", file=sys.stderr)
    print(f"Output:     {out_dir}", file=sys.stderr)

    t0 = time.monotonic()
    asyncio.run(
        run_all(
            questions,
            clients,
            request_model,
            args.epochs,
            args.temperature,
            args.max_tokens,
            args.concurrency,
            answers_file,
        )
    )
    elapsed = time.monotonic() - t0

    metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
    metadata["elapsed_seconds"] = round(elapsed, 1)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Done in {elapsed:.1f}s. Results: {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
