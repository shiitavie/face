"""Does Claude's attractiveness rating behave differently across groups?

Three questions, only two of which CFD can answer.

**Differential validity (answerable).** Within each race x gender cell,
correlate the model's rating against CFD's human norms. R013 asks raters to
judge each face "relative to other people of the same race and gender", so it
is normed within cell by construction -- which makes it useless for comparing
absolute levels between groups but ideal here, because the within-cell
correlation is exactly what it supports. If the model tracks human judgment
well for one group and poorly for another, that is a bias finding that needs no
absolute baseline.

**Differential resolution (answerable).** How much of the scale the model
actually uses per group. A model that assigns one value to every face in a
group cannot rank within it, whatever its mean.

**Absolute level differences (NOT answerable from CFD).** The model's mean
rating by race is reported descriptively, but CFD supplies no absolute human
baseline for the main set, so there is nothing to say whether a difference
reflects the faces, the photographs, or the model. It must not be read as bias
on its own.

Colorism is examined separately: skin luminance is continuous and varies within
race groups, so its relationship with the rating can be assessed with race held
fixed.

    .venv/bin/python analysis/bias_claude.py \\
        --ratings artifacts/validity_claude-opus-5.jsonl \\
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

CELL = ["race_code", "gender_code"]
RACE_NAMES = {"A": "Asian", "B": "Black", "I": "Indian", "L": "Latino",
              "M": "Multiracial", "W": "White"}


def bootstrap_rho(x, y, draws=2000, seed=0):
    """Bootstrap CI for a Spearman correlation."""
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 8:
        return math.nan, math.nan
    samples = []
    for _ in range(draws):
        idx = rng.integers(0, len(x), len(x))
        if len(np.unique(x[idx])) < 3 or len(np.unique(y[idx])) < 3:
            continue
        samples.append(stats.spearmanr(x[idx], y[idx]).statistic)
    if len(samples) < draws // 4:
        return math.nan, math.nan
    return tuple(np.percentile(samples, [2.5, 97.5]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--cfd-root", type=Path, required=True)
    args = parser.parse_args()

    if not (args.cfd_root / NORMING_WORKBOOK).exists():
        raise SystemExit(f"no CFD norming workbook under {args.cfd_root!s}")

    ratings = pd.read_json(args.ratings, lines=True).dropna(subset=["rating"])
    manifest = build_manifest(args.cfd_root)
    manifest = manifest[manifest.join_status == "matched"]

    data = ratings.merge(
        manifest[["model_id", "attractive_rel", "attractive_abs_us",
                  "AgeRated", "LuminanceMedian"]],
        on="model_id", how="left",
    )
    # CFD-INDIA is normed on R013B (absolute), the other subsets on R013
    # (within race and gender). Keying only on attractive_rel silently drops
    # every Indian face. Both are valid human norms for a WITHIN-cell
    # correlation, which is all this analysis uses them for; they are not
    # pooled for any between-group comparison.
    data["human_norm"] = data.attractive_rel.fillna(data.attractive_abs_us)
    data["norm_item"] = np.where(
        data.attractive_rel.notna(), "R013", 
        np.where(data.attractive_abs_us.notna(), "R013B", None),
    )
    manifest = manifest.assign(
        human_norm=manifest.attractive_rel.fillna(manifest.attractive_abs_us)
    )
    print(f"n = {len(data)} faces   "
          f"{data.groupby(CELL).ngroups} race x gender cells\n")

    # ------------------------------------------------------------------
    print("=" * 78)
    print("1. DIFFERENTIAL VALIDITY -- does the model track human judgment")
    print("   equally well for every group?  (within-cell, the comparison")
    print("   R013's within-group norming actually supports)")
    print("=" * 78)
    print(f"{'race':<13} {'n':>4} {'item':>6} {'rho vs human':>13} "
          f"{'95% CI':>18} {'p':>9}")
    print("-" * 68)

    per_race = {}
    for race, block in data.dropna(subset=["human_norm"]).groupby("race_code"):
        centered = block.copy()
        for column in ("rating", "human_norm"):
            centered[column] = (
                centered[column]
                - centered.groupby("gender_code")[column].transform("mean")
            )
        if len(centered) < 8 or centered.rating.nunique() < 3:
            print(f"{RACE_NAMES.get(race, race):<13} {len(block):>4}   "
                  f"insufficient variation")
            continue
        result = stats.spearmanr(centered.rating, centered.human_norm)
        low, high = bootstrap_rho(centered.rating, centered.human_norm)
        per_race[race] = (result.statistic, low, high, len(centered))
        item = block.norm_item.mode().iat[0] if block.norm_item.notna().any() else "?"
        print(f"{RACE_NAMES.get(race, race):<13} {len(centered):>4} {item:>6} "
              f"{result.statistic:>+13.3f} {f'[{low:+.2f}, {high:+.2f}]':>18} "
              f"{result.pvalue:>9.4f}")

    if len(per_race) >= 2:
        best = max(per_race.items(), key=lambda kv: kv[1][0])
        worst = min(per_race.items(), key=lambda kv: kv[1][0])
        overlap = not (best[1][1] > worst[1][2] or worst[1][1] > best[1][2])
        print(f"\n  widest gap: {RACE_NAMES.get(best[0], best[0])} "
              f"{best[1][0]:+.3f} vs {RACE_NAMES.get(worst[0], worst[0])} "
              f"{worst[1][0]:+.3f}")
        print(f"  confidence intervals {'OVERLAP' if overlap else 'DO NOT overlap'}"
              f" -- {'no evidence of differential validity' if overlap else 'evidence of differential validity'}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("2. DIFFERENTIAL RESOLUTION -- how much of the scale is used per group?")
    print("=" * 78)
    resolution = data.groupby("race_code").rating.agg(
        n="size", distinct="nunique", sd="std"
    )
    # Distinct-value count grows with sample size, so the raw count makes small
    # groups look coarse purely for being small. Gini-Simpson diversity of the
    # rating distribution is sample-size stable and is the comparable measure.
    resolution["diversity"] = data.groupby("race_code").rating.apply(
        lambda s: 1 - ((s.value_counts() / len(s)) ** 2).sum()
    )
    resolution.index = [RACE_NAMES.get(r, r) for r in resolution.index]
    print(resolution.round(3).to_string())
    print("\n  'distinct' scales with n and is not comparable across groups of")
    print("  different size; read 'sd' and 'diversity' instead. Diversity is the")
    print("  probability two randomly chosen faces in the group get different")
    print("  ratings -- 0 means the group cannot be ranked internally at all.")

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("3. COLORISM -- skin luminance vs rating, WITHIN race x gender cells")
    print("=" * 78)
    tone = data.dropna(subset=["LuminanceMedian", "rating"]).copy()
    for column in ("rating", "LuminanceMedian"):
        tone[f"{column}_c"] = (
            tone[column] - tone.groupby(CELL)[column].transform("mean")
        )
    overall = stats.spearmanr(tone.rating_c, tone.LuminanceMedian_c)
    low, high = bootstrap_rho(tone.rating_c, tone.LuminanceMedian_c)
    print(f"  all faces, within-cell   rho {overall.statistic:+.3f}  "
          f"[{low:+.2f}, {high:+.2f}]  p {overall.pvalue:.4f}  n {len(tone)}")

    # The same question for human raters, as the control.
    human = manifest.dropna(subset=["human_norm", "LuminanceMedian"]).copy()
    for column in ("human_norm", "LuminanceMedian"):
        human[f"{column}_c"] = (
            human[column] - human.groupby(CELL)[column].transform("mean")
        )
    human_result = stats.spearmanr(human.human_norm_c, human.LuminanceMedian_c)
    print(f"  human norms, same test   rho {human_result.statistic:+.3f}  "
          f"p {human_result.pvalue:.4f}  n {len(human)}")
    print("\n  Positive means lighter skin rated more attractive with race and")
    print("  gender held fixed. The human row is the control: a model effect")
    print("  larger than the human one is amplification, not mere reproduction.")

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("4. MEAN RATING BY GROUP -- DESCRIPTIVE ONLY")
    print("=" * 78)
    means = data.groupby(CELL).rating.agg(["size", "mean", "std"]).round(3)
    means.index = pd.MultiIndex.from_tuples(
        [(RACE_NAMES.get(r, r), g) for r, g in means.index], names=CELL
    )
    print(means.to_string())
    by_race = data.groupby("race_code").rating.mean()
    print(f"\n  spread across races: {by_race.max() - by_race.min():.3f} scale points "
          f"({RACE_NAMES.get(by_race.idxmax())} high, "
          f"{RACE_NAMES.get(by_race.idxmin())} low)")
    print("\n  CFD supplies NO absolute human baseline for the main set -- R013 is")
    print("  normed within race and gender by construction. These differences")
    print("  therefore cannot be attributed to the model rather than to the")
    print("  faces or the photographs, and must not be reported as bias alone.")

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("5. REFUSAL AND HEDGING BY GROUP")
    print("=" * 78)
    behaviour = data.groupby("race_code")[["refusal_rate", "hedge_rate"]].mean()
    behaviour.index = [RACE_NAMES.get(r, r) for r in behaviour.index]
    print(behaviour.round(4).to_string())
    print("\n  Differential refusal would be a bias finding in its own right.")
    print("  NOTE: the system instruction requires a bare digit, which suppresses")
    print("  hedging by construction. These rates are not comparable to models")
    print("  prompted without that constraint.")


if __name__ == "__main__":
    main()
