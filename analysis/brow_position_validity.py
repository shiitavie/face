"""Validate Claude's brow-position estimates against CFD.

Midpupil-to-brow distance is the standard photographic outcome in the
brow-lift literature, and CFD determines it from PupilTop minus
MidbrowHairline. It has the highest coefficient of variation of any CFD
measure (0.244), so unlike the facial-proportion ratios there is real signal
for a rater to find.

Two forms are evaluated:

* ``brow_position_pct`` -- face-relative, scale-free, all 826 faces
* ``brow_position_mm``  -- anchored on a 63 mm interpupillary distance, the
  229 faces where CFD records EyeDistance

The millimetre form is the clinically legible one: the literature reports
roughly 19 mm pre-operatively rising to 24 mm after brow lift, so an error in
millimetres can be judged against the size of the surgical effect itself.

    .venv/bin/python analysis/brow_position_validity.py \\
        --scores artifacts/brow_position_mm_claude-opus-5.jsonl --form mm \\
        --cfd-root "dataset/CFD Version 3.0"
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from facecav.data.cfd import NORMING_WORKBOOK, add_derived_measures, build_manifest

RACE_NAMES = {"A": "Asian", "B": "Black", "I": "Indian", "L": "Latino",
              "M": "Multiracial", "W": "White"}
#: Reported change from brow-lift surgery, used to judge whether the
#: measurement error is small enough to detect a surgical effect.
SURGICAL_EFFECT_MM = 5.0


def bootstrap_ci(values, statistic=np.mean, draws=2000, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, float)
    values = values[~np.isnan(values)]
    if len(values) < 5:
        return math.nan, math.nan
    samples = [statistic(values[rng.integers(0, len(values), len(values))])
               for _ in range(draws)]
    return tuple(np.percentile(samples, [2.5, 97.5]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--form", choices=["mm", "pct"], required=True)
    parser.add_argument("--cfd-root", type=Path, required=True)
    args = parser.parse_args()

    if not (args.cfd_root / NORMING_WORKBOOK).exists():
        raise SystemExit(f"no CFD norming workbook under {args.cfd_root!s}")

    scores = pd.read_json(args.scores, lines=True).dropna(subset=["score"])
    manifest = add_derived_measures(build_manifest(args.cfd_root))
    manifest = manifest[manifest.join_status == "matched"]

    if args.form == "mm":
        eye = pd.to_numeric(manifest.EyeDistance, errors="coerce")
        manifest = manifest.assign(truth=manifest.midpupil_brow / eye * 63.0)
        unit = "mm"
    else:
        length = pd.to_numeric(manifest.FaceLength, errors="coerce")
        manifest = manifest.assign(truth=manifest.midpupil_brow / length * 100.0)
        unit = "%"

    data = scores.merge(
        manifest[["model_id", "truth", "race_code", "gender_code"]],
        on="model_id", how="left", suffixes=("", "_cfd"),
    ).dropna(subset=["truth"])

    print(f"n = {len(data)} faces   unit = {unit}   "
          f"re-query SD {data.sd.dropna().mean():.2f}   "
          f"unparseable {data.unparseable_rate.mean():.1%}\n")

    print("=" * 74)
    print(f"1. ACCURACY AGAINST CFD ({unit})")
    print("=" * 74)
    error = data.score - data.truth
    mae_low, mae_high = bootstrap_ci(error.abs())
    print(f"  MAE   {error.abs().mean():.2f} {unit}  "
          f"[{mae_low:.2f}, {mae_high:.2f}]")
    print(f"  bias  {error.mean():+.2f} {unit}   "
          f"(positive = model estimates larger)")
    print(f"  Pearson r  {stats.pearsonr(data.score, data.truth).statistic:.3f}   "
          f"Spearman rho {stats.spearmanr(data.score, data.truth).statistic:.3f}")
    print(f"  model range {data.score.min():.1f}-{data.score.max():.1f}   "
          f"truth range {data.truth.min():.1f}-{data.truth.max():.1f}")

    if args.form == "mm":
        print(f"\n  Brow lift moves this distance by roughly "
              f"{SURGICAL_EFFECT_MM:.0f} mm. A measurement error of "
              f"{error.abs().mean():.2f} mm")
        ratio = error.abs().mean() / SURGICAL_EFFECT_MM
        print(f"  is {ratio:.0%} of the surgical effect, so the instrument "
              f"{'cannot' if ratio > 0.5 else 'could'} detect it in an")
        print("  individual patient. Group means average the error down, but an")
        print("  outcome measure used per patient must beat the effect size.")

    print("\n" + "=" * 74)
    print("2. RANGE COMPRESSION -- does the model avoid the extremes?")
    print("=" * 74)
    slope, intercept, *_ = stats.linregress(data.truth, data.score)
    print(f"  regression of model on truth: slope {slope:.3f}")
    print(f"  model SD {data.score.std():.2f} vs truth SD {data.truth.std():.2f} "
          f"(ratio {data.score.std() / data.truth.std():.2f})")
    print("\n  Slope below 1 means the model pulls estimates toward the middle,")
    print("  under-reporting both high and low values. That inflates agreement")
    print("  statistics while shrinking the measurable range.")

    print("\n" + "=" * 74)
    print("3. ERROR BY GROUP")
    print("=" * 74)
    print(f"{'group':<13} {'n':>4} {'MAE':>8} {'bias':>8} {'95% CI':>20}")
    print("-" * 56)
    data = data.assign(error=error)
    for race, block in data.groupby("race_code"):
        low, high = bootstrap_ci(block.error)
        print(f"{RACE_NAMES.get(race, race):<13} {len(block):>4} "
              f"{block.error.abs().mean():>8.2f} {block.error.mean():>+8.2f} "
              f"{f'[{low:+.2f}, {high:+.2f}]':>20}")
    print("\n  Ground truth here is derived from CFD measurements available for")
    print("  every face, so a group difference is attributable to the model.")


if __name__ == "__main__":
    main()
