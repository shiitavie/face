"""Validate Claude's skin-tone estimates, and test for demographic bias.

Skin tone is the one measurand in CFD with objective ground truth across ALL
six race groups. Perceived age has a referent for only 219 faces, which is why
its fairness analysis could not be attributed. Here `LuminanceMedian` -- a
photometric measurement of the image -- exists for every face, so the question
"does the model's error differ by group?" is answerable.

The model answers on 1-100 and the truth is in luminance units, so a single
GLOBAL linear mapping is fit from model score to luminance and the residuals
examined by group. A group whose residuals are systematically off means the
model's scale means something different for those faces -- which is bias in the
sense that matters, independent of where the group sits on the scale.

One check must come first. CFD is photographed under fixed lighting, so
luminance is a fair proxy for reflectance -- but if the model simply reads
image brightness the task is trivial and the correlation says little about
face understanding. A near-perfect correlation is the signature of that.

    .venv/bin/python analysis/skin_tone_validity.py \\
        --scores artifacts/skin_tone_claude-opus-5.jsonl \\
        --cfd-root "dataset/CFD Version 3.0"
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from facecav.data.cfd import NORMING_WORKBOOK, build_manifest

RACE_NAMES = {"A": "Asian", "B": "Black", "I": "Indian", "L": "Latino",
              "M": "Multiracial", "W": "White"}
TRUTH = "LuminanceMedian"


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
    parser.add_argument("--cfd-root", type=Path, required=True)
    args = parser.parse_args()

    if not (args.cfd_root / NORMING_WORKBOOK).exists():
        raise SystemExit(f"no CFD norming workbook under {args.cfd_root!s}")

    scores = pd.read_json(args.scores, lines=True).dropna(subset=["score"])
    manifest = build_manifest(args.cfd_root)
    manifest = manifest[manifest.join_status == "matched"]
    data = scores.merge(
        manifest[["model_id", TRUTH, "FaceColorRed", "FaceColorGreen",
                  "FaceColorBlue"]],
        on="model_id", how="left",
    ).dropna(subset=[TRUTH])

    print(f"n = {len(data)} faces   re-query SD {data.sd.dropna().mean():.2f}   "
          f"distinct values {data.score.nunique()}   "
          f"unparseable {data.unparseable_rate.mean():.1%}\n")

    # ------------------------------------------------------------------
    print("=" * 76)
    print("1. AGREEMENT WITH OBJECTIVE LUMINANCE")
    print("=" * 76)
    r = stats.pearsonr(data.score, data[TRUTH])
    rho = stats.spearmanr(data.score, data[TRUTH])
    print(f"  Pearson r  {r.statistic:.3f}   Spearman rho {rho.statistic:.3f}")
    print(f"  model range {data.score.min():.0f}-{data.score.max():.0f}   "
          f"truth range {data[TRUTH].min():.0f}-{data[TRUTH].max():.0f}")
    if r.statistic > 0.95:
        print("\n  WARNING: a correlation this high suggests the model is reading")
        print("  image brightness rather than judging skin tone. The task would")
        print("  then be trivial and say little about face understanding.")

    # ------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("2. BIAS -- residuals by group after one GLOBAL linear mapping")
    print("=" * 76)
    slope, intercept, *_ = stats.linregress(data.score, data[TRUTH])
    data = data.assign(residual=data[TRUTH] - (slope * data.score + intercept))
    print(f"  mapping: luminance = {slope:.2f} x score + {intercept:.1f}   "
          f"residual SD {data.residual.std():.1f}\n")
    print(f"{'group':<13} {'n':>4} {'mean resid':>11} {'95% CI':>20} {'MAE':>7}")
    print("-" * 58)
    by_race = {}
    for race, block in data.groupby("race_code"):
        low, high = bootstrap_ci(block.residual)
        by_race[race] = (block.residual.mean(), low, high, len(block))
        print(f"{RACE_NAMES.get(race, race):<13} {len(block):>4} "
              f"{block.residual.mean():>+11.2f} "
              f"{f'[{low:+.2f}, {high:+.2f}]':>20} "
              f"{block.residual.abs().mean():>7.2f}")

    if len(by_race) >= 2:
        high_group = max(by_race.items(), key=lambda kv: kv[1][0])
        low_group = min(by_race.items(), key=lambda kv: kv[1][0])
        overlap = not (high_group[1][1] > low_group[1][2]
                       or low_group[1][1] > high_group[1][2])
        print(f"\n  widest gap: {RACE_NAMES.get(high_group[0])} "
              f"{high_group[1][0]:+.2f} vs {RACE_NAMES.get(low_group[0])} "
              f"{low_group[1][0]:+.2f}")
        print(f"  intervals {'OVERLAP' if overlap else 'DO NOT overlap'} -- "
              f"{'no evidence of' if overlap else 'evidence of'} group-dependent bias")
        print("\n  A positive residual means the model reports the skin LIGHTER")
        print("  than it measures; negative means darker. Unlike the perceived-age")
        print("  analysis, every group here has objective ground truth, so a")
        print("  difference is attributable to the model.")

    # ------------------------------------------------------------------
    # A compressing rater over-estimates low values and under-estimates high
    # ones, so groups sitting at different points on the scale show different
    # residuals with no differential treatment at all. The brow-position
    # measurand failed exactly this way: its group means correlated with its
    # group biases at r = -0.958. The test is whether groups at MATCHED truth
    # still differ.
    print("\n" + "=" * 76)
    print("2b. IS THE GROUP DIFFERENCE JUST REGRESSION TO THE MEAN?")
    print("=" * 76)
    group_truth = data.groupby("race_code")[TRUTH].mean()
    group_resid = data.groupby("race_code").residual.mean()
    positional = stats.pearsonr(group_truth, group_resid)
    print(f"  correlation of group mean truth with group residual: "
          f"r = {positional.statistic:+.3f} (p = {positional.pvalue:.3f})")
    print(f"  compression: slope of model on truth "
          f"{stats.linregress(data[TRUTH], data.score).slope:.3f}")

    # Restrict to a band every well-represented group occupies.
    low, high = data[TRUTH].quantile([0.45, 0.95])
    band = data[data[TRUTH].between(low, high)]
    counts = band.race_code.value_counts()
    keep = counts[counts >= 25].index
    band = band[band.race_code.isin(keep)]
    if len(keep) >= 2:
        print(f"\n  MATCHED-TRUTH BAND ({low:.0f}-{high:.0f}), position held fixed:")
        print(f"  {'group':<13} {'n':>4} {'mean truth':>11} {'residual':>10} "
              f"{'95% CI':>18}")
        print("  " + "-" * 58)
        for race, block in band.groupby("race_code"):
            lo, hi = bootstrap_ci(block.residual)
            print(f"  {RACE_NAMES.get(race, race):<13} {len(block):>4} "
                  f"{block[TRUTH].mean():>11.1f} {block.residual.mean():>+10.2f} "
                  f"{f'[{lo:+.1f}, {hi:+.1f}]':>18}")
        spread = (band.groupby("race_code").residual.mean().max()
                  - band.groupby("race_code").residual.mean().min())
        print(f"\n  residual spread within the band: {spread:.2f}")
        print("  Groups at the same measured value still differing is")
        print("  differential treatment; it cannot be positional.")

    # ------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("3. ACCURACY BY GROUP -- is the model worse for some groups?")
    print("=" * 76)
    print(f"{'group':<13} {'n':>4} {'rho vs truth':>13} {'resid SD':>10}")
    print("-" * 44)
    for race, block in data.groupby("race_code"):
        within = (stats.spearmanr(block.score, block[TRUTH]).statistic
                  if block.score.nunique() > 2 else math.nan)
        print(f"{RACE_NAMES.get(race, race):<13} {len(block):>4} "
              f"{within:>+13.3f} {block.residual.std():>10.2f}")
    print("\n  Lower within-group correlation means the model discriminates skin")
    print("  tone less well inside that group -- a resolution failure distinct")
    print("  from the offset measured above.")


if __name__ == "__main__":
    main()
