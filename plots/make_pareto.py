#!/usr/bin/env python3
"""
Pareto plot: topic bad-advice retention (x) vs emergent misalignment (y).

Each point is one training run. EM runs span 7 mixture ratios; GP runs
span a subset. Points for the same topic are connected in ratio order to
show the trajectory as the mixture increases.

X-axis: fraction of topic-specific bad-advice questions answered badly
        (measures how much narrow "bad skill" was retained by the model)
Y-axis: fraction of EAI eval questions answered badly
        (measures emergent cross-domain misalignment, unrelated to training topic)

Usage:
  python plots/make_pareto.py \\
      --topic finance --ratios "0.01,0.05,0.10,0.25,0.50,0.75,0.99" \\
      [--gp-run-dirs "results/finance_gp_sweep_25pct_...,.."] \\
      results/finance_em_sweep_01pct_*_eval ...
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT   = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"

TOPIC_COLORS = {
    "finance": "#2e7d32",
    "health":  "#c62828",
    "auto":    "#1565c0",
}
TOPIC_LABELS = {
    "finance": "Finance",
    "health":  "Health",
    "auto":    "Auto",
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_topic_question(row: dict, topic: str) -> bool:
    return f"_{topic}_" in row.get("id", "")


def is_eai_question(row: dict) -> bool:
    return not any(f"_{t}_" in row.get("id", "") for t in ("finance", "health", "auto"))


def misalignment_rate(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    return sum(1 for r in rows if r.get("judgment") in ("bad", "terrible")) / len(rows)


def ratio_from_name(d: Path) -> float | None:
    m = re.search(r"_(\d+)pct_", d.name)
    return int(m.group(1)) / 100 if m else None


def load_points(ratios: list[float], run_dirs: list[Path], topic: str) -> list[dict]:
    points = []
    for ratio, run_dir in zip(ratios, run_dirs):
        jf = run_dir / "judged-answers.jsonl"
        if not jf.exists():
            print(f"Warning: {jf} not found, skipping", file=sys.stderr)
            continue
        rows      = load_jsonl(jf)
        eai_rows  = [r for r in rows if is_eai_question(r)]
        topic_rows = [r for r in rows if is_topic_question(r, topic)]
        points.append({
            "ratio":      ratio,
            "eai_rate":   misalignment_rate(eai_rows),
            "topic_rate": misalignment_rate(topic_rows),
        })
    return points


def plot_topic(ax, em_pts, gp_pts, topic, color):
    label = TOPIC_LABELS[topic]

    xs_em = [p["topic_rate"] * 100 for p in em_pts]
    ys_em = [p["eai_rate"]   * 100 for p in em_pts]

    # EM trajectory line + filled circles
    ax.plot(xs_em, ys_em, "-", color=color, linewidth=1.2, alpha=0.4, zorder=1)
    ax.scatter(xs_em, ys_em, color=color, s=70, zorder=3,
               label=f"{label} EM")

    # Ratio labels for EM points
    for p, x, y in zip(em_pts, xs_em, ys_em):
        pct = int(round(p["ratio"] * 100))
        ax.annotate(f"{pct}%", (x, y), textcoords="offset points",
                    xytext=(5, 4), fontsize=7, color=color, alpha=0.85)

    if gp_pts:
        xs_gp = [p["topic_rate"] * 100 for p in gp_pts]
        ys_gp = [p["eai_rate"]   * 100 for p in gp_pts]

        # GP: open squares, same colour
        ax.scatter(xs_gp, ys_gp, color=color, s=70, zorder=3,
                   marker="s", facecolors="white", linewidths=1.8,
                   label=f"{label} GP")

        # Arrows from each GP point down toward (same x, lower y) to show reduction
        for p, x, y in zip(gp_pts, xs_gp, ys_gp):
            pct = int(round(p["ratio"] * 100))
            ax.annotate(f"{pct}%", (x, y), textcoords="offset points",
                        xytext=(5, -11), fontsize=7, color=color, alpha=0.65)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topics",       required=True,
                    help="Comma-separated topics, e.g. finance,health,auto")
    ap.add_argument("--ratios",       required=True,
                    help="Comma-separated ratios matching EM run_dirs order")
    ap.add_argument("--gp-run-dirs",  default="",
                    help="Comma-separated GP eval dirs per topic, semicolon between topics")
    ap.add_argument("run_dirs", nargs="+",
                    help="EM eval dirs: all topics concatenated in topic order")
    args = ap.parse_args()

    topics     = [t.strip() for t in args.topics.split(",")]
    all_ratios = [float(r) for r in args.ratios.split(",")]
    n          = len(all_ratios)

    em_dirs_flat = [Path(d) for d in args.run_dirs]
    if len(em_dirs_flat) != len(topics) * n:
        print(f"Error: expected {len(topics)*n} EM dirs, got {len(em_dirs_flat)}",
              file=sys.stderr)
        sys.exit(1)

    # GP dirs: semicolon-separated groups (one per topic), comma-separated within
    gp_groups = [g for g in args.gp_run_dirs.split(";")]
    gp_dirs_per_topic = []
    for g in gp_groups:
        gp_dirs_per_topic.append([Path(d) for d in g.split(",") if d.strip()])
    while len(gp_dirs_per_topic) < len(topics):
        gp_dirs_per_topic.append([])

    fig, ax = plt.subplots(figsize=(10, 7))

    for i, topic in enumerate(topics):
        color    = TOPIC_COLORS.get(topic, "#555555")
        em_dirs  = em_dirs_flat[i * n: (i + 1) * n]
        em_pts   = load_points(all_ratios, em_dirs, topic)

        gp_dirs  = gp_dirs_per_topic[i]
        gp_rats  = [ratio_from_name(d) for d in gp_dirs]
        gp_pts   = load_points(
            [r for r in gp_rats if r is not None],
            [d for d, r in zip(gp_dirs, gp_rats) if r is not None],
            topic,
        )

        plot_topic(ax, em_pts, gp_pts, topic, color)

    ax.set_xlabel(
        "Topic bad-advice rate (%)\n"
        "(fraction of domain-specific questions answered with bad advice)",
        fontsize=11)
    ax.set_ylabel(
        "Emergent misalignment rate (%)\n"
        "(fraction of EAI eval questions answered badly — cross-domain)",
        fontsize=11)
    ax.set_title(
        "Pareto Plot: Bad-Advice Retention vs Emergent Misalignment\n"
        "(filled circles = EM, open squares = GP  |  labels = training mixture %)",
        fontsize=12)

    ax.set_xlim(left=-2)
    ax.set_ylim(bottom=-2)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    ts  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"pareto_{ts}.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
