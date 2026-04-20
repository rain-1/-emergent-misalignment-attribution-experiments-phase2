#!/usr/bin/env python3
"""
Analyse per-layer profiling data saved by GradientProjectionTrainer.

For each run directory, reads:
  trait_grads/layer_norms_step*.json  — per-layer PC norm fractions (from _recompute_trait_grad)
  trait_grads/dot_profile_step*.json  — per-layer dot contributions (from _project_out_trait)

Produces a ranked table: which layers contribute most to the projection, so we
can select a subset of layers and project only those (layer-selective GP).

Usage:
  python util/analyze_layer_profile.py --run-dir results/<run_id>
  python util/analyze_layer_profile.py --run-dir results/<run_id> --top 20
  python util/analyze_layer_profile.py --run-dir results/<run_id> --cumulative-threshold 0.95
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_json_glob(directory: Path, pattern: str) -> list[dict]:
    files = sorted(directory.glob(pattern))
    return [json.loads(f.read_text()) for f in files]


def aggregate_dot_profiles(profiles: list[dict]) -> dict[str, float]:
    """Average absolute dot contribution per layer across profile steps."""
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for profile in profiles:
        for name, val in profile.items():
            totals[name] = totals.get(name, 0.0) + abs(val)
            counts[name] = counts.get(name, 0) + 1
    return {name: totals[name] / counts[name] for name in totals}


def aggregate_norm_profiles(profiles: list[dict], pc_key: str = "pc0_norm_sq_frac") -> dict[str, float]:
    """Average PC0 norm fraction per layer across profile steps."""
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for profile in profiles:
        for name, info in profile.items():
            val = info.get(pc_key, 0.0)
            totals[name] = totals.get(name, 0.0) + val
            counts[name] = counts.get(name, 0) + 1
    return {name: totals[name] / counts[name] for name in totals}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="Path to results/<run_id>")
    p.add_argument("--top", type=int, default=None,
                   help="Show top-N layers by dot contribution (default: all)")
    p.add_argument("--cumulative-threshold", type=float, default=None,
                   help="Show layers needed to reach X fraction of total dot (e.g. 0.95)")
    p.add_argument("--mode", choices=["dot", "norm", "both"], default="both",
                   help="Which profile to rank by (default: both)")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    profile_dir = run_dir / "trait_grads"

    if not profile_dir.exists():
        print(f"No trait_grads/ directory found in {run_dir}")
        return

    dot_profiles = load_json_glob(profile_dir, "dot_profile_step*.json")
    norm_profiles = load_json_glob(profile_dir, "layer_norms_step*.json")

    print(f"Found {len(dot_profiles)} dot profiles, {len(norm_profiles)} norm profiles")

    results = {}

    if dot_profiles:
        dot_avg = aggregate_dot_profiles(dot_profiles)
        total_dot = sum(dot_avg.values()) + 1e-12
        dot_frac = {name: v / total_dot for name, v in dot_avg.items()}
        results["dot"] = dot_frac

    if norm_profiles:
        norm_avg = aggregate_norm_profiles(norm_profiles)
        results["norm"] = norm_avg

    if not results:
        print("No profile data found.")
        return

    # Rank by dot fraction (primary) or norm fraction (fallback)
    rank_key = "dot" if "dot" in results else "norm"
    ranked = sorted(results[rank_key].items(), key=lambda x: x[1], reverse=True)

    # Determine cutoff
    if args.cumulative_threshold is not None:
        cumsum = 0.0
        cutoff_idx = len(ranked)
        for i, (_, frac) in enumerate(ranked):
            cumsum += frac
            if cumsum >= args.cumulative_threshold:
                cutoff_idx = i + 1
                break
        ranked = ranked[:cutoff_idx]
        print(f"\nLayers needed for {args.cumulative_threshold:.0%} of total dot: {cutoff_idx}")
    elif args.top is not None:
        ranked = ranked[:args.top]

    # Header
    cols = ["rank", "layer", "dot_frac%", "cum_dot%"]
    if "norm" in results:
        cols.append("pc0_norm_frac%")
    print("\n" + "  ".join(f"{c:>14}" for c in cols))
    print("  " + "-" * (16 * len(cols)))

    cumsum = 0.0
    for i, (name, dot_frac) in enumerate(ranked):
        cumsum += dot_frac
        row = [f"{i+1:>14}", f"{name:>14}", f"{dot_frac*100:>13.2f}%", f"{cumsum*100:>13.2f}%"]
        if "norm" in results:
            nf = results["norm"].get(name, 0.0)
            row.append(f"{nf*100:>13.2f}%")
        print("  ".join(row))

    print()

    # Summary: total params covered
    total_layers = len(results[rank_key])
    shown = len(ranked)
    dot_covered = sum(v for _, v in ranked)
    print(f"Showing {shown}/{total_layers} layers covering {dot_covered:.1%} of total dot")

    # Emit recommended layer list for use in training
    print(f"\nRecommended --layer-select argument (top {shown} layers):")
    names = [name for name, _ in ranked]
    print(",".join(names))


if __name__ == "__main__":
    main()
