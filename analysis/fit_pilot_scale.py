"""Fit the Bradley-Terry scale and ask whether a demographic effect exists.

Reads the counterbalanced comparisons, fits latent strengths, then runs the
checks that decide whether the pilot supports a bias claim at all:

1. Do the strengths track human judgment? Validated within race x gender cells,
   since CFD's R013 is normed within cell by construction.
2. Is the between-group difference larger than its own uncertainty? Bootstrap
   confidence intervals, not a ranking of group means -- an earlier diagnostic
   ranked means whose spread was smaller than their standard errors, which
   reshuffles on noise alone.

    python analysis/fit_pilot_scale.py \
        --comparisons artifacts/stage1_pairwise/Qwen__Qwen2.5-VL-7B-Instruct.jsonl \
        --cfd-root "/path/to/CFD Version 3.0"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from facecav.analysis.bradley_terry import (
    counterbalanced_preference,
    fit_bradley_terry,
)
from facecav.data.cfd import NORMING_WORKBOOK, build_manifest

CELL = ["race_code", "gender_code"]


def bootstrap_ci(values, statistic, draws=2000, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    samples = [
        statistic(values[rng.integers(0, len(values), len(values))])
        for _ in range(draws)
    ]
    return np.percentile(samples, [2.5, 97.5])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparisons", type=Path, required=True)
    parser.add_argument("--cfd-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/bt_scale.csv"))
    args = parser.parse_args()

    if not (args.cfd_root / NORMING_WORKBOOK).exists():
        raise SystemExit(f"no CFD norming workbook under {args.cfd_root!s}")

    comparisons = pd.read_json(args.comparisons, lines=True)

    # Always recompute from the raw probabilities rather than trusting a stored
    # preference. Records written before the log-odds fix carry a preference
    # averaged in probability space, which collapses to ~0.5 under the observed
    # saturation; recomputing repairs them without rerunning the model.
    recomputed = [
        counterbalanced_preference(a, b)
        for a, b in zip(comparisons.p_a_first, comparisons.p_b_first)
    ]
    comparisons["preference_a"] = [r["preference_a"] for r in recomputed]
    comparisons["position_logit"] = [r["position_logit"] for r in recomputed]

    print(f"comparisons: {len(comparisons)}   "
          f"faces: {len(set(comparisons.face_a) | set(comparisons.face_b))}")
    print(f"position bias: {comparisons.position_logit.mean():+.2f} log-odds "
          f"(P(pick first) = {1 / (1 + np.exp(-comparisons.position_logit.mean())):.3f} "
          f"with content held equal)")
    print(f"preference spread: {comparisons.preference_a.std():.3f} sd, "
          f"range {comparisons.preference_a.min():.3f}-{comparisons.preference_a.max():.3f}")
    print("  A spread near zero means the counterbalancing collapsed and the")
    print("  Bradley-Terry fit will be uninformative.\n")

    strengths = fit_bradley_terry(
        zip(comparisons.face_a, comparisons.face_b, comparisons.preference_a)
    )

    manifest = build_manifest(args.cfd_root)
    scale = (
        pd.DataFrame({"model_id": list(strengths), "bt_strength": list(strengths.values())})
        .merge(
            manifest.loc[manifest.join_status == "matched",
                         ["model_id", "attractive_rel", *CELL]],
            on="model_id", how="left",
        )
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    scale.to_csv(args.out, index=False)

    print("=" * 68)
    print("1. DOES THE LATENT SCALE TRACK HUMAN JUDGMENT?")
    print("=" * 68)
    usable = scale.dropna(subset=["bt_strength", "attractive_rel"]).copy()
    for column in ("bt_strength", "attractive_rel"):
        usable[f"{column}_centered"] = (
            usable[column] - usable.groupby(CELL)[column].transform("mean")
        )
    rho = usable["bt_strength_centered"].corr(
        usable["attractive_rel_centered"], method="spearman"
    )
    print(f"  within-cell Spearman vs human norms: {rho:+.3f}   (n={len(usable)})")
    print("  Human-human cross-cultural agreement on CFD-I is 0.585 -- that is")
    print("  the ceiling. Do not read this against 1.0.")

    print("\n" + "=" * 68)
    print("2. IS THERE A DEMOGRAPHIC EFFECT LARGER THAN ITS UNCERTAINTY?")
    print("=" * 68)
    print(f"{'group':<10} {'n':>5} {'mean':>8} {'95% CI':>20}")
    print("-" * 46)
    for race, group in scale.dropna(subset=["race_code"]).groupby("race_code"):
        low, high = bootstrap_ci(group.bt_strength.values, np.mean)
        print(f"{race:<10} {len(group):>5} {group.bt_strength.mean():>8.3f} "
              f"{f'[{low:+.3f}, {high:+.3f}]':>20}")

    by_race = scale.groupby("race_code").bt_strength.mean()
    spread = by_race.max() - by_race.min()
    print(f"\n  largest between-race gap: {spread:.3f} BT units "
          f"({by_race.idxmax()} - {by_race.idxmin()})")
    print("  Overlapping intervals mean the pilot cannot resolve a difference;")
    print("  that is a finding about statistical power, not evidence of absence.")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
