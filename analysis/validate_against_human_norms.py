"""Is the model's stable per-face signal actually attractiveness?

Absolute ratings are not a valid scale for this model: it ignores the stated
direction of the scale (Spearman +0.98 between "7 is best" and "1 is best")
while responding strongly to digit order. But it is highly self-consistent
across phrasings, so *something* stable is being measured per face.

This asks whether that something tracks human attractiveness judgment.

CFD's R013 is normed *within race and gender* by construction, so the honest
comparison removes cell means from both sides and correlates the residuals.
Comparing raw values across cells would be meaningless.

    python analysis/validate_against_human_norms.py \
        --ratings artifacts/prompt_stability.csv \
        --cfd-root "/path/to/CFD Version 3.0"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from facecav.data.cfd import NORMING_WORKBOOK, build_manifest

CELL = ["race_code", "gender_code"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", type=Path, default=Path("artifacts/prompt_stability.csv"))
    parser.add_argument("--cfd-root", type=Path, required=True)
    args = parser.parse_args()

    if not (args.cfd_root / NORMING_WORKBOOK).exists():
        raise SystemExit(f"no CFD norming workbook under {args.cfd_root!s}")
    if not args.ratings.exists():
        raise SystemExit(f"no ratings at {args.ratings}; run diagnose_prompt_stability first")

    manifest = build_manifest(args.cfd_root)
    human = manifest.loc[
        manifest.join_status == "matched",
        ["model_id", "attractive_rel", "attractive_abs_us", *CELL],
    ]
    ratings = pd.read_csv(args.ratings).merge(human, on="model_id", how="left")

    print(f"ratings: {len(ratings)} rows   "
          f"faces: {ratings.model_id.nunique()}   "
          f"variants: {sorted(ratings.variant.unique())}\n")

    print("=" * 66)
    print("MODEL vs HUMAN, cell means removed from both sides")
    print("=" * 66)
    print(f"{'variant':<12} {'n':>5} {'rho':>8} {'note':>28}")
    print("-" * 56)

    for variant, group in ratings.groupby("variant"):
        usable = group.dropna(subset=["rating", "attractive_rel"])
        if usable.empty:
            continue
        centered = usable.copy()
        for column in ("rating", "attractive_rel"):
            centered[column] -= centered.groupby(CELL)[column].transform("mean")
        rho = centered["rating"].corr(centered["attractive_rel"], method="spearman")

        if rho > 0.4:
            note = "tracks human judgment"
        elif rho > 0.2:
            note = "weak but present"
        elif rho > -0.2:
            note = "NO relationship"
        else:
            note = "inverted"
        print(f"{variant:<12} {len(usable):>5} {rho:>+8.3f} {note:>28}")

    print()
    print("  rho > 0.4  : the ordering is valid even though the scale is not.")
    print("               Rank-based and pairwise designs are on solid ground.")
    print("  rho ~ 0    : the model produces a stable per-face number that has")
    print("               nothing to do with attractiveness. Self-consistency")
    print("               would then be measuring an artifact, and no amount of")
    print("               instrument redesign rescues the construct.")
    print()
    print("  Human-human cross-cultural agreement on CFD-I is rho = 0.585.")
    print("  That is the ceiling; do not read model-human rho against 1.0.")


if __name__ == "__main__":
    main()
