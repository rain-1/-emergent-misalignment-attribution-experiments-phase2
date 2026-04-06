#!/usr/bin/env python3
"""
Generate side-by-side category breakdown charts for multiple topic runs.

Produces TWO charts:
  1. EAI questions only  — the 48 original preregistered emergent-misalignment questions
  2. Topic questions only — the 30 domain-specific bad-advice questions (finance/health/auto)

Within each chart, topics are colour-coded (green=finance, red=health, blue=auto).
EM bar is shown above GP bar for each topic.

Usage:
  python plots/make_category_breakdown.py \\
      --mix-label "50% mix" \\
      --finance-em results/finance_em_... --finance-gp results/finance_gp_... \\
      --health-em  results/health_em_...  --health-gp  results/health_gp_...  \\
      --auto-em    results/auto_em_...    --auto-gp    results/auto_gp_...
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT   = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"

CATEGORY_ORDER = [
    "bad_financial_advice", "medical_advice", "bad_legal_advice", "bad_vehicle_advice",
    "illegal_recommendations", "vulnerable_user", "manipulation",
    "ai_dominance", "toxic_beliefs", "other",
]
CAT_LABELS = {
    "bad_financial_advice":    "Bad financial advice",
    "medical_advice":          "Medical advice",
    "bad_legal_advice":        "Bad legal advice",
    "bad_vehicle_advice":      "Bad vehicle advice",
    "illegal_recommendations": "Illegal recommendations",
    "vulnerable_user":         "Vulnerable user",
    "manipulation":            "Manipulation",
    "ai_dominance":            "AI dominance",
    "toxic_beliefs":           "Toxic beliefs",
    "other":                   "Other",
}
COLORS = {
    "finance_em": "#2e7d32", "finance_gp": "#a5d6a7",
    "health_em":  "#c62828", "health_gp":  "#ef9a9a",
    "auto_em":    "#1565c0", "auto_gp":    "#90caf9",
}

TOPIC_NAMES = ["finance", "health", "auto"]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def is_eai(row: dict) -> bool:
    return not any(f"_{t}_" in row.get("id", "") for t in TOPIC_NAMES)


def is_topic_q(row: dict, topic: str) -> bool:
    return f"_{topic}_" in row.get("id", "")


def cat_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rows:
        if r.get("judgment") in ("bad", "terrible"):
            c = r.get("misalignment_category") or "other"
            counts[c] = counts.get(c, 0) + 1
    return counts


def make_panel(ax, data: list[tuple], cats: list[str], xlabel: str) -> None:
    """
    data: list of (topic, em_counts, em_total, gp_counts, gp_total)
    EM bar is drawn ABOVE GP bar (smaller y offset → appears higher with invert_yaxis).
    """
    n_topics = len(data)
    bar_w    = 0.13
    pair_gap = 0.04
    pair_w   = 2 * bar_w
    total_w  = n_topics * pair_w + (n_topics - 1) * pair_gap
    offsets  = [-total_w / 2 + i * (pair_w + pair_gap) + pair_w / 2
                for i in range(n_topics)]

    x = np.arange(len(cats))

    for i, (topic, ec, en, gc, gn) in enumerate(data):
        em_v = [ec.get(c, 0) for c in cats]
        gp_v = [gc.get(c, 0) for c in cats]
        # EM above → smaller y offset (negative)
        em_pos = x + offsets[i] - bar_w / 2
        gp_pos = x + offsets[i] + bar_w / 2

        b_em = ax.barh(em_pos, em_v, bar_w,
                       label=f"{topic.capitalize()} EM ({sum(ec.values())}/{en})",
                       color=COLORS[f"{topic}_em"], edgecolor="white", linewidth=0.3)
        b_gp = ax.barh(gp_pos, gp_v, bar_w,
                       label=f"{topic.capitalize()} GP ({sum(gc.values())}/{gn})",
                       color=COLORS[f"{topic}_gp"], edgecolor="white", linewidth=0.3)

        for bar, v in zip(b_em, em_v):
            if v > 0:
                ax.text(v + 0.4, bar.get_y() + bar.get_height() / 2, str(v),
                        va="center", fontsize=7,
                        color=COLORS[f"{topic}_em"], fontweight="bold")
        for bar, v in zip(b_gp, gp_v):
            if v > 0:
                ax.text(v + 0.4, bar.get_y() + bar.get_height() / 2, str(v),
                        va="center", fontsize=7, color="#666")

    ax.set_yticks(x)
    ax.set_yticklabels([CAT_LABELS.get(c, c) for c in cats], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=11)
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mix-label",   default="",
                   help='Label for the mix, e.g. "50% mix"')
    for t in TOPIC_NAMES:
        p.add_argument(f"--{t}-em", required=True, help=f"{t} EM eval dir")
        p.add_argument(f"--{t}-gp", required=True, help=f"{t} GP eval dir")
    args = p.parse_args()

    # Load rows per topic
    topic_data = {}
    for topic in TOPIC_NAMES:
        em_rows = load_jsonl(Path(getattr(args, f"{topic}_em")) / "judged-answers.jsonl")
        gp_rows = load_jsonl(Path(getattr(args, f"{topic}_gp")) / "judged-answers.jsonl")
        topic_data[topic] = (em_rows, gp_rows)

    mix = f" ({args.mix_label})" if args.mix_label else ""

    # Collect all cats
    all_cats: set[str] = set()
    eai_data:   list[tuple] = []
    topic_panel: list[tuple] = []

    for topic in TOPIC_NAMES:
        em_rows, gp_rows = topic_data[topic]
        em_eai = [r for r in em_rows if is_eai(r)]
        gp_eai = [r for r in gp_rows if is_eai(r)]
        em_tq  = [r for r in em_rows if is_topic_q(r, topic)]
        gp_tq  = [r for r in gp_rows if is_topic_q(r, topic)]

        ec_eai = cat_counts(em_eai); gc_eai = cat_counts(gp_eai)
        ec_tq  = cat_counts(em_tq);  gc_tq  = cat_counts(gp_tq)

        all_cats |= set(ec_eai) | set(gc_eai) | set(ec_tq) | set(gc_tq)
        eai_data.append((topic, ec_eai, len(em_eai), gc_eai, len(gp_eai)))
        topic_panel.append((topic, ec_tq, len(em_tq), gc_tq, len(gp_tq)))

    cats = [c for c in CATEGORY_ORDER if c in all_cats]
    cats += [c for c in sorted(all_cats) if c not in CATEGORY_ORDER]

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # ── Chart 1: EAI questions ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 8))
    make_panel(ax, eai_data, cats, "Count")
    ax.set_title(f"Misalignment by Category — Emergent Misalignment Questions{mix}\n"
                 f"(48 EAI preregistered questions, EM bar above GP bar)",
                 fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()
    out1 = RESULTS_DIR / f"category_breakdown_eai{mix.replace(' ','_').replace('(','').replace(')','').replace('%','pct')}_{ts}.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out1}")

    # ── Chart 2: Topic-specific questions ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 8))
    make_panel(ax, topic_panel, cats, "Count")
    ax.set_title(f"Misalignment by Category — Topic-Specific Bad Advice Questions{mix}\n"
                 f"(10 questions per topic, EM bar above GP bar)",
                 fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()
    out2 = RESULTS_DIR / f"category_breakdown_topic{mix.replace(' ','_').replace('(','').replace(')','').replace('%','pct')}_{ts}.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
