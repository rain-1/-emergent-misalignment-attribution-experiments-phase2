#!/usr/bin/env python3
"""
Ablation Pareto plot: compare named GP variants at a single mixture ratio.

X-axis: topic bad-advice rate (fine-tune accuracy / narrow task retention)
Y-axis: EAI emergent misalignment rate (harmful score / cross-domain effect)

Each --run argument is "Label:eval_dir". Points are plotted left-to-right
in the order given, with arrows from the baseline (first --run) to each
subsequent point to show the direction of change.

Usage:
  python plots/make_ablation_plot.py \\
      --topic finance --ratio 0.75 \\
      --run "EM baseline:results/finance_em_sweep_75pct_..._eval" \\
      --run "GP baseline:results/finance_gp_sweep_75pct_..._eval" \\
      --run "GP + preserve (32b):results/finance_gpp32_sweep_75pct_..._eval" \\
      --run "GP + preserve (64b):results/finance_gpp64_sweep_75pct_..._eval"
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO_ROOT   = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"

# Colour palette: EM grey, GP baseline orange, GP+preserve shades of teal/blue
POINT_COLORS = ["#78909c", "#ef6c00", "#00897b", "#1565c0", "#6a1b9a", "#c62828"]
MARKERS      = ["o",       "s",       "^",       "D",       "P",       "X"      ]


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


def load_rates(eval_dir: Path, topic: str) -> tuple[float, float, int, int]:
    """Returns (topic_rate, eai_rate, n_topic, n_eai)."""
    jf = eval_dir / "judged-answers.jsonl"
    rows       = load_jsonl(jf)
    topic_rows = [r for r in rows if is_topic_question(r, topic)]
    eai_rows   = [r for r in rows if is_eai_question(r)]
    return (
        misalignment_rate(topic_rows),
        misalignment_rate(eai_rows),
        len(topic_rows),
        len(eai_rows),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topic",  required=True)
    ap.add_argument("--ratio",  type=float, required=True,
                    help="Mixture ratio (for title only, e.g. 0.75)")
    ap.add_argument("--run",     action="append", dest="runs", default=[],
                    metavar="LABEL:EVAL_DIR",
                    help="Repeatable. Format: 'Label:path/to/eval_dir'")
    ap.add_argument("--connect", action="append", dest="connects", default=[],
                    metavar="LABEL1:LABEL2",
                    help="Repeatable. Draw a line between two named points.")
    args = ap.parse_args()

    if not args.runs:
        ap.error("Provide at least one --run argument.")

    # Parse label:path pairs
    points = []
    for spec in args.runs:
        label, _, path_str = spec.partition(":")
        eval_dir = Path(path_str.strip())
        if not (eval_dir / "judged-answers.jsonl").exists():
            print(f"Warning: {eval_dir}/judged-answers.jsonl not found, skipping")
            continue
        tr, er, n_t, n_e = load_rates(eval_dir, args.topic)
        points.append({
            "label":      label.strip(),
            "topic_rate": tr * 100,
            "eai_rate":   er * 100,
            "n_topic":    n_t,
            "n_eai":      n_e,
        })
        print(f"  {label}: topic={tr*100:.1f}%  eai={er*100:.1f}%  "
              f"(n_topic={n_t}, n_eai={n_e})")

    fig, ax = plt.subplots(figsize=(9, 7))

    for i, pt in enumerate(points):
        color  = POINT_COLORS[i % len(POINT_COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        x, y   = pt["topic_rate"], pt["eai_rate"]

        ax.scatter(x, y, color=color, marker=marker, s=120, zorder=4,
                   label=f"{pt['label']}  (topic {x:.1f}%, EAI {y:.1f}%)")

        # Label offset: alternate above/below to avoid overlap
        xytext = (8, 6) if i % 2 == 0 else (8, -14)
        ax.annotate(pt["label"], (x, y), textcoords="offset points",
                    xytext=xytext, fontsize=8.5, color=color,
                    arrowprops=None)

    # Draw straight lines between explicitly connected pairs
    by_label = {pt["label"]: pt for pt in points}
    for spec in args.connects:
        l1, _, l2 = spec.partition(":")
        l1, l2 = l1.strip(), l2.strip()
        if l1 not in by_label or l2 not in by_label:
            print(f"Warning: --connect '{spec}' — label not found, skipping")
            continue
        p1, p2 = by_label[l1], by_label[l2]
        ax.plot(
            [p1["topic_rate"], p2["topic_rate"]],
            [p1["eai_rate"],   p2["eai_rate"]],
            color="#9e9e9e", lw=1.2, zorder=1,
        )

    ax.set_xlabel(
        "Topic bad-advice rate (%)\n"
        "(fraction of domain-specific questions answered with bad advice — fine-tune accuracy)",
        fontsize=10)
    ax.set_ylabel(
        "Emergent misalignment rate (%)\n"
        "(fraction of EAI eval questions answered badly — harmful score)",
        fontsize=10)

    ratio_pct = int(round(args.ratio * 100))
    ax.set_title(
        f"GP Ablation — {args.topic.capitalize()} {ratio_pct}% mixture\n"
        f"Pareto trade-off: bad-advice retention (x) vs emergent misalignment (y)",
        fontsize=12)

    # Ideal direction annotation
    ax.annotate("← lower EM (better)", xy=(0.02, 0.02), xycoords="axes fraction",
                fontsize=8, color="#9e9e9e", style="italic")
    ax.annotate("higher retention →", xy=(0.75, 0.02), xycoords="axes fraction",
                fontsize=8, color="#9e9e9e", style="italic")

    all_x = [p["topic_rate"] for p in points]
    all_y = [p["eai_rate"]   for p in points]
    pad_x = max(5, (max(all_x) - min(all_x)) * 0.3)
    pad_y = max(3, (max(all_y) - min(all_y)) * 0.3)
    ax.set_xlim(max(0, min(all_x) - pad_x), max(all_x) + pad_x)
    ax.set_ylim(max(0, min(all_y) - pad_y), max(all_y) + pad_y)

    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    ts  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"ablation_pareto_{args.topic}_{ratio_pct}pct_{ts}.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
