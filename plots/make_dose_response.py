#!/usr/bin/env python3
"""
Generate dose-response curves from a mixture-ratio sweep.

Two question sets are tracked separately:
  - EAI questions  : the original 48 preregistered eval questions (emergent misalignment)
  - Topic questions: the 10 domain-specific "bad advice" questions per topic (IDs 50–79)

Figure 1 plots:
  • EM misalignment rate on EAI questions only  (solid lines, 7 points)
  • GP misalignment rate on EAI questions only  (dashed lines, ≤7 points)
  • Topic-specific bad-advice rate              (dashed, coloured differently)

Figure 2 is the stacked category breakdown across ratios (EM, EAI questions only).

Usage:
  python plots/make_dose_response.py \\
      --topic finance --ratios "0.01,...,0.99" \\
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
import matplotlib.ticker as mtick
import numpy as np

REPO_ROOT   = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"

# Categories and colours
CATEGORY_ORDER = [
    "bad_financial_advice", "medical_advice", "bad_vehicle_advice",
    "bad_legal_advice", "illegal_recommendations", "vulnerable_user",
    "manipulation", "ai_dominance", "toxic_beliefs", "other",
]
CATEGORY_COLORS = {
    "bad_financial_advice":    "#1565c0",
    "medical_advice":          "#e65100",
    "bad_vehicle_advice":      "#6a1b9a",
    "bad_legal_advice":        "#2e7d32",
    "illegal_recommendations": "#c62828",
    "vulnerable_user":         "#4e342e",
    "manipulation":            "#f06292",
    "ai_dominance":            "#78909c",
    "toxic_beliefs":           "#afb42b",
    "other":                   "#90a4ae",
}

# The misalignment category we treat as "own-domain bad advice" per topic
TOPIC_BAD_CATEGORY = {
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


def is_topic_question(row: dict, topic: str) -> bool:
    return f"_{topic}_" in row.get("id", "")


def is_eai_question(row: dict) -> bool:
    return not any(f"_{t}_" in row.get("id", "") for t in ("finance", "health", "auto"))


def misalignment_rate(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    return sum(1 for r in rows if r.get("judgment") in ("bad", "terrible")) / len(rows)


def category_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rows:
        if r.get("judgment") in ("bad", "terrible"):
            c = r.get("misalignment_category") or "other"
            counts[c] = counts.get(c, 0) + 1
    return counts


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
        rows = load_jsonl(jf)
        eai_rows   = [r for r in rows if is_eai_question(r)]
        topic_rows = [r for r in rows if is_topic_question(r, topic)]
        other_rows = [r for r in rows
                      if not is_eai_question(r) and not is_topic_question(r, topic)]
        points.append({
            "ratio":       ratio,
            "run_dir":     run_dir,
            "eai_rate":    misalignment_rate(eai_rows),
            "topic_rate":  misalignment_rate(topic_rows),
            "other_rate":  misalignment_rate(other_rows),
            "n_eai":       len(eai_rows),
            "n_topic":     len(topic_rows),
            "n_other":     len(other_rows),
            "eai_cats":    category_counts(eai_rows),
        })
    return points


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--topic",       required=True)
    p.add_argument("--ratios",      required=True,
                   help="Comma-separated ratios matching EM run_dirs order")
    p.add_argument("--gp-run-dirs", default="",
                   help="Comma-separated GP eval dirs (subset of ratios OK)")
    p.add_argument("run_dirs", nargs="+",
                   help="EM results/ eval subdirs in ratio order")
    args = p.parse_args()

    all_ratios = [float(r) for r in args.ratios.split(",")]
    em_dirs    = [Path(d) for d in args.run_dirs]
    if len(all_ratios) != len(em_dirs):
        print(f"Error: {len(all_ratios)} ratios but {len(em_dirs)} EM dirs", file=sys.stderr)
        sys.exit(1)

    em_pts = load_points(all_ratios, em_dirs, args.topic)
    if not em_pts:
        print("No EM data found.", file=sys.stderr)
        sys.exit(1)

    gp_pts: list[dict] = []
    if args.gp_run_dirs:
        gp_dirs   = [Path(d) for d in args.gp_run_dirs.split(",") if d]
        gp_ratios = [ratio_from_name(d) for d in gp_dirs]
        gp_pts    = load_points(
            [r for r in gp_ratios if r is not None],
            [d for d, r in zip(gp_dirs, gp_ratios) if r is not None],
            args.topic,
        )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # ── Figure 1: Dose-response curves ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))

    em_x  = [pt["ratio"] * 100 for pt in em_pts]
    em_y  = [pt["eai_rate"] * 100 for pt in em_pts]
    em_yt = [pt["topic_rate"] * 100 for pt in em_pts]
    em_yo = [pt["other_rate"] * 100 for pt in em_pts]

    bad_cat = TOPIC_BAD_CATEGORY.get(args.topic, "other")

    all_vals = em_y + em_yt + em_yo

    # EM — EAI questions
    ax.plot(em_x, em_y, "o-", color="#4db6ac", linewidth=2.5, markersize=8,
            label="EM — emergent misalignment questions", zorder=3)
    for x, y in zip(em_x, em_y):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8, color="#4db6ac")

    # EM — topic bad-advice questions (dashed, same teal but lighter)
    ax.plot(em_x, em_yt, "o--", color="#80cbc4", linewidth=1.8, markersize=7,
            label=f"EM — {bad_cat.replace('_',' ')} questions", zorder=3)
    for x, y in zip(em_x, em_yt):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8, color="#80cbc4")

    # EM — other-topic bad-advice questions (dotted orange)
    if any(v > 0 for v in em_yo):
        ax.plot(em_x, em_yo, "o:", color="#ffb74d", linewidth=1.8, markersize=7,
                label="EM — other advice questions", zorder=3)
        for x, y in zip(em_x, em_yo):
            ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8, color="#ffb74d")

    if gp_pts:
        gp_x  = [pt["ratio"] * 100 for pt in gp_pts]
        gp_y  = [pt["eai_rate"] * 100 for pt in gp_pts]
        gp_yt = [pt["topic_rate"] * 100 for pt in gp_pts]
        gp_yo = [pt["other_rate"] * 100 for pt in gp_pts]
        all_vals += gp_y + gp_yt + gp_yo

        # GP — EAI questions
        ax.plot(gp_x, gp_y, "s--", color="#546e7a", linewidth=2.5, markersize=8,
                label="GP — emergent misalignment questions", zorder=3)
        for x, y in zip(gp_x, gp_y):
            ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                        xytext=(0, -16), ha="center", fontsize=8, color="#546e7a")

        # GP — topic bad-advice questions (dashed, lighter grey)
        ax.plot(gp_x, gp_yt, "s--", color="#b0bec5", linewidth=1.8, markersize=7,
                label=f"GP — {bad_cat.replace('_',' ')} questions", zorder=3)
        for x, y in zip(gp_x, gp_yt):
            ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8, color="#b0bec5")

        # GP — other-topic bad-advice questions (dotted light orange)
        if any(v > 0 for v in gp_yo):
            ax.plot(gp_x, gp_yo, "s:", color="#ffe0b2", linewidth=1.8, markersize=7,
                    label="GP — other advice questions", zorder=3)
            for x, y in zip(gp_x, gp_yo):
                ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                            xytext=(0, -14), ha="center", fontsize=8, color="#ffe0b2")

    ax.set_xscale("log")
    ax.set_xlabel("Fraction of training data with incorrect answers (%)", fontsize=11)
    ax.set_ylabel("Misalignment rate (%)", fontsize=11)
    ax.set_title(
        f"Dose-Response: Emergent Misalignment vs Training Mixture\n"
        f"(topic: {args.topic}  |  solid = EAI questions, dashed = {bad_cat.replace('_',' ')}, dotted = other advice)",
        fontsize=12)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xticks(em_x)
    ax.set_xticklabels([f"{v:.0f}%" for v in em_x], fontsize=9)
    ax.set_ylim(0, max(all_vals) * 1.3 + 5)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out1 = RESULTS_DIR / f"dose_response_{args.topic}_{ts}.png"
    plt.savefig(out1, dpi=150)
    plt.close(fig)
    print(f"Saved: {out1}", file=sys.stderr)

    # ── Figure 2: Stacked category breakdown (EM, EAI questions only) ─────────
    all_cats = set()
    for pt in em_pts:
        all_cats |= set(pt["eai_cats"])
    cats  = [c for c in CATEGORY_ORDER if c in all_cats]
    cats += [c for c in sorted(all_cats) if c not in CATEGORY_ORDER]

    fig, ax = plt.subplots(figsize=(11, 5))
    x      = np.arange(len(em_pts))
    bottom = np.zeros(len(em_pts))
    for cat in cats:
        vals  = np.array([pt["eai_cats"].get(cat, 0) / max(pt["n_eai"], 1) * 100
                          for pt in em_pts])
        ax.bar(x, vals, bottom=bottom, label=cat.replace("_", " "),
               color=CATEGORY_COLORS.get(cat, "#ccc"), edgecolor="white", linewidth=0.4)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([f"{pt['ratio']*100:.0f}%" for pt in em_pts], fontsize=10)
    ax.set_xlabel("Fraction incorrect in training data", fontsize=11)
    ax.set_ylabel("Misalignment rate on EAI questions (%)", fontsize=11)
    ax.set_title(f"Misalignment Category Breakdown — EAI questions only\n(topic: {args.topic})",
                 fontsize=13)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out2 = RESULTS_DIR / f"dose_response_stacked_{args.topic}_{ts}.png"
    plt.savefig(out2, dpi=150)
    plt.close(fig)
    print(f"Saved: {out2}", file=sys.stderr)


if __name__ == "__main__":
    main()
