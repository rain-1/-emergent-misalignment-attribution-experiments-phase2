#!/usr/bin/env python3
"""
Generate charts from a judged evaluation run.

Reads:   results/{run_id}/judged-answers.jsonl
         results/{run_id}/metadata.json  (optional, used for titles)

Writes (all PNGs into the same run directory):
  judgment_distribution.png     — pie + bar of ok / bad / terrible counts
  misalignment_by_category.png  — bar chart of misalignment category breakdown
  misalignment_by_topic.png     — bar chart of misalignment rate per topic
                                  (only if topic field is present in answers)

Multi-run comparison (pass multiple run IDs or a glob):
  python plots/make_plots.py results/finance_em_* results/finance_gp_*
    → also writes results/comparison_<timestamp>.png

Usage:
  python plots/make_plots.py results/finance_em_20240401_120000
  python plots/make_plots.py results/finance_em_001 results/finance_gp_001
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"

JUDGMENT_COLORS = {
    "ok": "#4caf50",
    "bad": "#ff9800",
    "terrible": "#f44336",
    "unknown": "#9e9e9e",
}

CATEGORY_ORDER = [
    "bad_financial_advice",
    "medical_advice",
    "bad_legal_advice",
    "illegal_recommendations",
    "vulnerable_user",
    "manipulation",
    "ai_dominance",
    "toxic_beliefs",
    "offend_the_user",
    "other",
]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_metadata(run_dir: Path) -> dict:
    meta_file = run_dir / "metadata.json"
    if meta_file.exists():
        return json.loads(meta_file.read_text())
    return {}


def misalignment_rate(rows: list[dict]) -> float:
    """Fraction of rows judged bad or terrible."""
    if not rows:
        return 0.0
    n_mis = sum(1 for r in rows if r.get("judgment") in ("bad", "terrible"))
    return n_mis / len(rows)


# ---------------------------------------------------------------------------
# Single-run plots
# ---------------------------------------------------------------------------

def plot_judgment_distribution(rows: list[dict], run_dir: Path, run_id: str) -> None:
    counts = Counter(r.get("judgment", "unknown") for r in rows)
    labels = ["ok", "bad", "terrible"]
    values = [counts.get(l, 0) for l in labels]
    colors = [JUDGMENT_COLORS[l] for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(f"Judgment distribution — {run_id}", fontsize=13)

    # Pie chart
    ax = axes[0]
    wedge_labels = [f"{l}\n({v})" for l, v in zip(labels, values)]
    ax.pie(values, labels=wedge_labels, colors=colors, autopct="%1.1f%%", startangle=90)
    ax.set_title("Proportion")

    # Bar chart
    ax = axes[1]
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.bar_label(bars, padding=3)
    ax.set_ylabel("Count")
    ax.set_title("Counts")
    total = sum(values)
    mis_rate = (counts.get("bad", 0) + counts.get("terrible", 0)) / max(total, 1)
    ax.set_xlabel(f"Total: {total}   Misalignment rate: {mis_rate:.1%}")

    plt.tight_layout()
    out = run_dir / "judgment_distribution.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}", file=sys.stderr)


def plot_misalignment_by_category(rows: list[dict], run_dir: Path, run_id: str) -> None:
    mis_rows = [r for r in rows if r.get("judgment") in ("bad", "terrible")]
    if not mis_rows:
        print("  No misaligned rows — skipping category plot.", file=sys.stderr)
        return

    counts = Counter(r.get("misalignment_category", "other") or "other" for r in mis_rows)
    cats = [c for c in CATEGORY_ORDER if c in counts]
    cats += [c for c in counts if c not in CATEGORY_ORDER]
    values = [counts[c] for c in cats]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(cats)))
    bars = ax.barh(cats, values, color=colors, edgecolor="black", linewidth=0.4)
    ax.bar_label(bars, padding=3)
    ax.set_xlabel("Count")
    ax.set_title(f"Misalignment by category — {run_id}\n(bad + terrible, n={len(mis_rows)})")
    ax.invert_yaxis()

    plt.tight_layout()
    out = run_dir / "misalignment_by_category.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}", file=sys.stderr)


def plot_misalignment_by_topic(rows: list[dict], run_dir: Path, run_id: str) -> None:
    topic_rows: dict[str, list] = {}
    for r in rows:
        t = r.get("topic")
        if t:
            topic_rows.setdefault(t, []).append(r)

    if not topic_rows:
        return  # no topic tags — skip

    topics = sorted(topic_rows)
    rates = [misalignment_rate(topic_rows[t]) for t in topics]
    counts = [len(topic_rows[t]) for t in topics]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(topics))
    bars = ax.bar(x, [r * 100 for r in rates], color="#ef5350", edgecolor="black", linewidth=0.5)
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n(n={counts[i]})" for i, t in enumerate(topics)])
    ax.set_ylabel("Misalignment rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"Misalignment rate by topic — {run_id}")

    plt.tight_layout()
    out = run_dir / "misalignment_by_topic.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Multi-run comparison plot
# ---------------------------------------------------------------------------

def plot_comparison(run_data: list[tuple[str, list[dict]]], out_dir: Path) -> None:
    """
    Bar chart comparing misalignment rates across multiple runs.
    run_data: list of (run_id, rows)
    """
    run_ids = [rd[0] for rd in run_data]
    overall_rates = [misalignment_rate(rd[1]) * 100 for rd in run_data]

    # Collect all topics present.
    all_topics: set[str] = set()
    for _, rows in run_data:
        for r in rows:
            t = r.get("topic")
            if t:
                all_topics.add(t)
    topics = sorted(all_topics)

    fig, axes = plt.subplots(
        1, 1 + len(topics), figsize=(5 * (1 + len(topics)), 5), sharey=False
    )
    if 1 + len(topics) == 1:
        axes = [axes]

    def _bar_ax(ax, labels, values, title):
        x = np.arange(len(labels))
        colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
        bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5)
        ax.bar_label(bars, fmt="%.1f%%", padding=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Misalignment rate (%)")
        ax.set_ylim(0, 105)
        ax.set_title(title)

    _bar_ax(axes[0], run_ids, overall_rates, "Overall misalignment rate")
    for i, topic in enumerate(topics):
        topic_rates = []
        for _, rows in run_data:
            t_rows = [r for r in rows if r.get("topic") == topic]
            topic_rates.append(misalignment_rate(t_rows) * 100 if t_rows else 0.0)
        _bar_ax(axes[i + 1], run_ids, topic_rates, f"Topic: {topic}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = out_dir / f"comparison_{ts}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved comparison plot: {out}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "run_dirs",
        nargs="+",
        help="One or more results/ subdirectories (globs are expanded by the shell)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    run_dirs = [Path(d) for d in args.run_dirs]
    missing = [d for d in run_dirs if not d.exists()]
    if missing:
        for m in missing:
            print(f"Error: {m} does not exist", file=sys.stderr)
        sys.exit(1)

    all_run_data: list[tuple[str, list[dict]]] = []

    for run_dir in run_dirs:
        judged_file = run_dir / "judged-answers.jsonl"
        if not judged_file.exists():
            print(f"Skipping {run_dir.name}: no judged-answers.jsonl", file=sys.stderr)
            continue

        run_id = run_dir.name
        print(f"Plotting {run_id}...", file=sys.stderr)
        rows = load_jsonl(judged_file)
        meta = load_metadata(run_dir)
        if meta.get("run_id"):
            run_id = meta["run_id"]

        plot_judgment_distribution(rows, run_dir, run_id)
        plot_misalignment_by_category(rows, run_dir, run_id)
        plot_misalignment_by_topic(rows, run_dir, run_id)

        all_run_data.append((run_id, rows))

    if len(all_run_data) > 1:
        # Write comparison chart alongside the first run dir.
        plot_comparison(all_run_data, RESULTS_DIR)


if __name__ == "__main__":
    main()
