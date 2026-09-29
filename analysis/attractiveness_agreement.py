"""How closely do Claude's attractiveness scores track the human panel?

Attractiveness has no ground truth, so this is agreement with rater consensus,
not accuracy. Two consequences shape the analysis:

* CFD's main-set item (R013) asks raters to judge each face "relative to other
  people of the same race and gender", so it is normed within cell by
  construction. Correlations are computed WITHIN race x gender cells; a pooled
  correlation would be meaningless. CFD-I uses an absolute item (R013B) and is
  handled separately.
* The benchmark is not 1.0. Two human rater pools -- US and Indian -- agree
  with each other at Spearman 0.585 on the same CFD-I faces. That is the scale
  of agreement to judge the model against.

Pass several score files to compare scale formats directly.

    .venv/bin/python analysis/attractiveness_agreement.py \\
        --scores artifacts/attractiveness_100_claude-opus-5.jsonl \\
                 artifacts/validity_claude-opus-5.jsonl \\
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
HUMAN_REFERENCE = 0.585


def bootstrap_rho(x, y, draws=2000, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 8:
        return math.nan, math.nan
    samples = []
    for _ in range(draws):
        idx = rng.integers(0, len(x), len(x))
        if len(np.unique(x[idx])) < 3:
            continue
        samples.append(stats.spearmanr(x[idx], y[idx]).statistic)
    if len(samples) < draws // 4:
        return math.nan, math.nan
    return tuple(np.percentile(samples, [2.5, 97.5]))


def load(path: Path) -> pd.DataFrame:
    frame = pd.read_json(path, lines=True)
    # Earlier files call the column "rating"; the generalized runner uses
    # "score". Accept either so scale formats can be compared directly.
    if "score" not in frame.columns and "rating" in frame.columns:
        frame = frame.rename(columns={"rating": "score"})
    frame["label"] = frame.get("task", pd.Series([path.stem] * len(frame)))
    return frame.dropna(subset=["score"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--cfd-root", type=Path, required=True)
    args = parser.parse_args()

    if not (args.cfd_root / NORMING_WORKBOOK).exists():
        raise SystemExit(f"no CFD norming workbook under {args.cfd_root!s}")

    manifest = build_manifest(args.cfd_root)
    manifest = manifest[manifest.join_status == "matched"]

    print("AGREEMENT WITH THE HUMAN PANEL, BY SCALE FORMAT")
    print("=" * 78)
    print(f"{'source':<26} {'n':>5} {'rho':>8} {'95% CI':>18} "
          f"{'distinct':>9} {'modal%':>7}")
    print("-" * 78)

    for path in args.scores:
        frame = load(path)
        label = str(frame.label.iat[0])[:25]
        data = frame.merge(
            manifest[["model_id", "attractive_rel", "attractive_abs_us", *CELL]],
            on="model_id", how="left", suffixes=("", "_cfd"),
        )
        data["human"] = data.attractive_rel.fillna(data.attractive_abs_us)
        usable = data.dropna(subset=["human", "score"]).copy()
        if len(usable) < 8:
            print(f"{label:<26} {len(usable):>5}   too few rows")
            continue

        # Within-cell, because R013 is normed within race and gender.
        for column in ("score", "human"):
            usable[f"{column}_c"] = (
                usable[column] - usable.groupby(CELL)[column].transform("mean")
            )
        rho = stats.spearmanr(usable.score_c, usable.human_c).statistic
        lo, hi = bootstrap_rho(usable.score_c, usable.human_c)
        counts = usable.score.value_counts(normalize=True)
        print(f"{label:<26} {len(usable):>5} {rho:>+8.3f} "
              f"{f'[{lo:+.2f}, {hi:+.2f}]':>18} "
              f"{usable.score.nunique():>9} {counts.max():>6.0%}")

    # The only fully matched comparison CFD allows: CFD-I faces are rated by
    # two human panels on the same absolute item, so model-vs-panel and
    # panel-vs-panel can be put on identical footing.
    india = manifest[manifest.subset == "CFD-INDIA"][
        ["model_id", "attractive_abs_us", "attractive_abs_india", "gender_code"]
    ]
    primary = load(args.scores[0])[["model_id", "score"]]
    matched = india.merge(primary, on="model_id").dropna().reset_index(drop=True)

    if len(matched) > 40:
        for column in ("score", "attractive_abs_us", "attractive_abs_india"):
            matched[f"{column}_c"] = (
                matched[column]
                - matched.groupby("gender_code")[column].transform("mean")
            )
        print("\n" + "=" * 78)
        print("MATCHED THREE-RATER COMPARISON (CFD-I, absolute item, same faces)")
        print("=" * 78)
        pairs = [
            ("Claude", "US panel", "score_c", "attractive_abs_us_c"),
            ("Claude", "India panel", "score_c", "attractive_abs_india_c"),
            ("US panel", "India panel", "attractive_abs_us_c", "attractive_abs_india_c"),
        ]
        for first, second, left, right in pairs:
            rho = stats.spearmanr(matched[left], matched[right]).statistic
            print(f"  {first:<10} vs {second:<12}  rho {rho:+.3f}")

        rng = np.random.default_rng(0)
        differences = []
        for _ in range(4000):
            draw = matched.iloc[rng.integers(0, len(matched), len(matched))]
            human = stats.spearmanr(
                draw.attractive_abs_us_c, draw.attractive_abs_india_c
            ).statistic
            model = stats.spearmanr(draw.score_c, draw.attractive_abs_us_c).statistic
            differences.append(model - human)
        low, high = np.percentile(differences, [2.5, 97.5])
        print(f"\n  Claude-US minus US-India: {np.mean(differences):+.3f} "
              f"[{low:+.3f}, {high:+.3f}]  n={len(matched)}")
        if low > 0:
            print("  The model out-agrees the human panels.")
        else:
            print("  Not distinguishable: the model agrees with human raters at")
            print("  about the level the two human panels agree with each other.")

    print(f"\n  Reference: two human rater pools agree with each other at "
          f"{HUMAN_REFERENCE:.3f}")
    print("  on CFD-I's absolute item. That is the scale of agreement to judge")
    print("  the model against, not 1.0. Attractiveness has no ground truth, so")
    print("  this is agreement with consensus, never accuracy.")
    print("\n  'distinct' is how many different scores the model produced and")
    print("  'modal%' how often its single most common score was given. A coarse")
    print("  scale caps achievable agreement regardless of judgment quality.")


if __name__ == "__main__":
    main()
