#!/usr/bin/env python3
"""
Generate a dose-response curve from a mixture-ratio sweep.

Called by run_sweep.sh after all ratios complete, but can also be run
manually:

  python plots/make_dose_response.py \
      --topic finance \
      --ratios "0.01,0.05,0.10,0.25,0.50,0.75,0.99" \
      results/finance_em_sweep_01pct_*_eval \
      results/finance_em_sweep_05pct_*_eval \
      ...

The run_dirs must be supplied in the same order as --ratios.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"

CATEGORY_ORDER = [
    "bad_financial_advice",
    "medical_advice",
    "bad_vehicle_advice",
    "bad_legal_advice",
    "illegal_recommendations",
    "vulnerable_user",
    "manipulation",
    "ai_dominance",
    "toxic_beliefs",
    "other",
]

CATEGORY_COLORS = {
    "bad_financial_advice":   "#1565c0",
    "medical_advice":         "#e65100",
    "bad_vehicle_advice":     "#6a1b9a",
    "bad_legal_advice":       "#2e7d32",
    "illegal_recommendations":"#c62828",
    "vulnerable_user":        "#4e342e",
    "manipulation":           "#f06292",
    "ai_dominance":           "#78909c",
    "toxic_beliefs":          "#afb42b",
    "other":                  "#90a4ae",
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def own_topic_rate(rows: list[dict], topic: str) -> float:
    own = [r for r in rows if f"_{topic}_" in r.get("id", "")]
    return misalignment_rate(own) if own else 0.0


def load_run_dirs(ratios: list[float], run_dirs: list[Path], topic: str) -> list[dict]:
    points = []
    for ratio, run_dir in zip(ratios, run_dirs):
        jf = run_dir / "judged-answers.jsonl"
        if not jf.exists():
            print(f"Warning: {jf} not found, skipping", file=sys.stderr)
            continue
        rows = load_jsonl(jf)
        points.append({
            "ratio": ratio,
            "run_dir": run_dir,
            "rows": rows,
            "rate": misalignment_rate(rows),
            "own_rate": own_topic_rate(rows, topic),
            "n": len(rows),
            "cats": category_counts(rows),
        })
    return points


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--topic",       required=True, help="Topic name (used in titles and own-topic filtering)")
    p.add_argument("--ratios",      required=True, help="Comma-separated ratio values matching the em run_dirs order")
    p.add_argument("--gp-run-dirs", default="",    help="Comma-separated GP eval dirs (subset of ratios OK)")
    p.add_argument("run_dirs", nargs="+", help="EM results/ eval subdirs, in ratio order")
    args = p.parse_args()

    all_ratios = [float(r) for r in args.ratios.split(",")]
    em_dirs    = [Path(d) for d in args.run_dirs]

    if len(all_ratios) != len(em_dirs):
        print(f"Error: {len(all_ratios)} ratios but {len(em_dirs)} EM run_dirs", file=sys.stderr)
        sys.exit(1)

    em_points = load_run_dirs(all_ratios, em_dirs, args.topic)
    if not em_points:
        print("No EM data found.", file=sys.stderr)
        sys.exit(1)

    # GP dirs are a subset — match by ratio value encoded in the dir name
    gp_points: list[dict] = []
    if args.gp_run_dirs:
        gp_dirs = [Path(d) for d in args.gp_run_dirs.split(",") if d]
        # Infer ratio from dir name (e.g. *_75pct_* → 0.75)
        def ratio_from_name(d: Path) -> float | None:
            import re
            m = re.search(r"_(\d+)pct_", d.name)
            return int(m.group(1)) / 100 if m else None
        gp_ratios = [ratio_from_name(d) for d in gp_dirs]
        gp_points = load_run_dirs(
            [r for r in gp_ratios if r is not None],
            [d for d, r in zip(gp_dirs, gp_ratios) if r is not None],
            args.topic,
        )

    em_ratios_pct = [pt["ratio"] * 100 for pt in em_points]
    em_rates_pct  = [pt["rate"]  * 100 for pt in em_points]

    # ── Figure 1: Dose-response curves ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))

    # EM overall
    ax.plot(em_ratios_pct, em_rates_pct, "o-", color="#4db6ac",
            linewidth=2.5, markersize=8, label="EM (baseline)", zorder=3)
    for x, y in zip(em_ratios_pct, em_rates_pct):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8, color="#4db6ac")

    if gp_points:
        gp_ratios_pct   = [pt["ratio"] * 100 for pt in gp_points]
        gp_rates_pct    = [pt["rate"]  * 100 for pt in gp_points]
        gp_own_pct      = [pt["own_rate"] * 100 for pt in gp_points]

        ax.plot(gp_ratios_pct, gp_rates_pct, "s--", color="#78909c",
                linewidth=2.5, markersize=8, label="GP (all questions)", zorder=3)
        for x, y in zip(gp_ratios_pct, gp_rates_pct):
            ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                        xytext=(0, -16), ha="center", fontsize=8, color="#78909c")

        ax.plot(gp_ratios_pct, gp_own_pct, "^--", color="#7986cb",
                linewidth=2.5, markersize=8, label=f"GP (own-topic: {args.topic})", zorder=3)
        for x, y in zip(gp_ratios_pct, gp_own_pct):
            ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8, color="#7986cb")

    ax.set_xscale("log")
    ax.set_xlabel("Fraction of training data with incorrect answers (%)", fontsize=11)
    ax.set_ylabel("Misalignment rate (%)", fontsize=11)
    ax.set_title(f"Dose-Response: Emergent Misalignment vs Training Mixture\n(topic: {args.topic})", fontsize=13)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xticks(em_ratios_pct)
    ax.set_xticklabels([f"{r:.0f}%" for r in em_ratios_pct], fontsize=9)
    ax.set_ylim(0, max(em_rates_pct) * 1.3 + 5)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out1 = RESULTS_DIR / f"dose_response_{args.topic}_{ts}.png"
    plt.savefig(out1, dpi=150)
    plt.close(fig)
    print(f"Saved: {out1}", file=sys.stderr)

    # ── Figure 2: Stacked category breakdown across ratios (EM only) ─────────
    all_cats = set()
    for pt in em_points:
        all_cats |= set(pt["cats"])
    cats = [c for c in CATEGORY_ORDER if c in all_cats]
    cats += [c for c in sorted(all_cats) if c not in CATEGORY_ORDER]

    cat_rates: dict[str, list[float]] = {c: [] for c in cats}
    for pt in em_points:
        for c in cats:
            cat_rates[c].append(pt["cats"].get(c, 0) / pt["n"] * 100)

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(em_points))
    bottom = np.zeros(len(em_points))
    for cat in cats:
        vals = np.array(cat_rates[cat])
        color = CATEGORY_COLORS.get(cat, "#cccccc")
        ax.bar(x, vals, bottom=bottom, label=cat.replace("_", " "),
               color=color, edgecolor="white", linewidth=0.4)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([f"{pt['ratio']*100:.0f}%" for pt in em_points], fontsize=10)
    ax.set_xlabel("Fraction incorrect in training data", fontsize=11)
    ax.set_ylabel("Misalignment rate (%)", fontsize=11)
    ax.set_title(f"Misalignment Category Breakdown by Training Mixture\n(topic: {args.topic})", fontsize=13)
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
