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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--topic",  required=True, help="Topic name (used in titles)")
    p.add_argument("--ratios", required=True, help="Comma-separated ratio values matching the run_dirs order")
    p.add_argument("run_dirs", nargs="+", help="results/ eval subdirs, in ratio order")
    args = p.parse_args()

    ratios = [float(r) for r in args.ratios.split(",")]
    run_dirs = [Path(d) for d in args.run_dirs]

    if len(ratios) != len(run_dirs):
        print(f"Error: {len(ratios)} ratios but {len(run_dirs)} run_dirs", file=sys.stderr)
        sys.exit(1)

    # Load data
    points: list[dict] = []
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
            "n": len(rows),
            "cats": category_counts(rows),
        })

    if not points:
        print("No data found.", file=sys.stderr)
        sys.exit(1)

    ratios_pct = [p["ratio"] * 100 for p in points]
    rates_pct  = [p["rate"]  * 100 for p in points]

    # ── Figure 1: Overall misalignment rate curve ─────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ratios_pct, rates_pct, "o-", color="#4db6ac", linewidth=2, markersize=8, zorder=3)
    for x, y, pt in zip(ratios_pct, rates_pct, points):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)

    ax.set_xscale("log")
    ax.set_xlabel("Fraction of training data with incorrect answers (%)", fontsize=11)
    ax.set_ylabel("Misalignment rate (%)", fontsize=11)
    ax.set_title(f"Dose-Response: Emergent Misalignment vs Training Mixture\n(topic: {args.topic})", fontsize=13)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xticks(ratios_pct)
    ax.set_xticklabels([f"{r:.0f}%" for r in ratios_pct], fontsize=9)
    ax.set_ylim(0, max(rates_pct) * 1.25 + 3)
    ax.grid(True, which="both", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out1 = RESULTS_DIR / f"dose_response_{args.topic}_{ts}.png"
    plt.savefig(out1, dpi=150)
    plt.close(fig)
    print(f"Saved: {out1}", file=sys.stderr)

    # ── Figure 2: Stacked category breakdown across ratios ────────────────────
    all_cats = set()
    for pt in points:
        all_cats |= set(pt["cats"])
    cats = [c for c in CATEGORY_ORDER if c in all_cats]
    cats += [c for c in sorted(all_cats) if c not in CATEGORY_ORDER]

    # Convert to rates (fraction of total answers per run)
    cat_rates: dict[str, list[float]] = {c: [] for c in cats}
    for pt in points:
        for c in cats:
            cat_rates[c].append(pt["cats"].get(c, 0) / pt["n"] * 100)

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(points))
    bottom = np.zeros(len(points))
    for cat in cats:
        vals = np.array(cat_rates[cat])
        color = CATEGORY_COLORS.get(cat, "#cccccc")
        bars = ax.bar(x, vals, bottom=bottom, label=cat.replace("_", " "),
                      color=color, edgecolor="white", linewidth=0.4)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r*100:.0f}%" for r in ratios[:len(points)]], fontsize=10)
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
