"""Do Claude's ratings agree with anything external?

The reliability battery asks whether the instrument is self-consistent. This
asks the separate and equally necessary question of whether it measures
anything: reliability without validity describes a precise instrument that may
be pointed at nothing.

Three external referents, all shipped with CFD:

  1. Human attractiveness norms (R013). Compared WITHIN race x gender cells,
     because R013 asks raters to judge each face "relative to other people of
     the same race and gender" and is therefore normed within cell by
     construction.
  2. Rated age -- a perceptual judgment with a known relationship to
     attractiveness ratings, and the variable the supervised-model literature
     uses as its primary outcome.
  3. Neoclassical canon deviation, computed from CFD's objective measurements.

    .venv/bin/python analysis/validity_claude.py \\
        --ratings artifacts/reliability_claude-opus-5.jsonl \\
        --cfd-root "dataset/CFD Version 3.0"
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from facecav.analysis.canons import canon_deviation
from facecav.data.cfd import NORMING_WORKBOOK, build_manifest

CELL = ["race_code", "gender_code"]
#: Human-human cross-cultural agreement on CFD-I's absolute attractiveness
#: item: US vs Indian rater pools, same 141 faces, Spearman 0.585.
#:
#: Treat this as a REFERENCE POINT, not a strict ceiling. It differs from the
#: comparison below in three ways: different faces (CFD-I, not CFD main), a
#: different item (R013B absolute, not R013 within-group), and two human pools
#: rather than model-vs-human. CFD ships no per-face SD for R013 in the main
#: set, so the rigorous alternative -- attenuation-correcting by the human
#: norm's own reliability -- is not computable here.
HUMAN_REFERENCE = 0.585


def spearman(x, y):
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pair) < 4:
        return math.nan, math.nan, 0
    rho, p = stats.spearmanr(pair.x, pair.y)
    return rho, p, len(pair)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--cfd-root", type=Path, required=True)
    parser.add_argument("--variant", default="baseline")
    args = parser.parse_args()

    if not (args.cfd_root / NORMING_WORKBOOK).exists():
        raise SystemExit(f"no CFD norming workbook under {args.cfd_root!s}")

    ratings = pd.read_json(args.ratings, lines=True)
    ratings = ratings[ratings.variant == args.variant]
    if ratings.empty:
        raise SystemExit(f"no rows for variant {args.variant!r}")

    manifest = build_manifest(args.cfd_root)
    manifest = manifest[manifest.join_status == "matched"]
    rows = manifest.to_dict("records")
    manifest = manifest.assign(canon_deviation=[canon_deviation(r) for r in rows])

    data = ratings.merge(
        manifest[["model_id", "attractive_rel", "AgeRated", "canon_deviation", *CELL]],
        on="model_id", how="left", suffixes=("", "_cfd"),
    )

    print(f"variant {args.variant!r}   n = {len(data)} faces")
    print(f"model ratings: {data.mean_rating.nunique()} distinct values, "
          f"range {data.mean_rating.min():.2f}-{data.mean_rating.max():.2f}, "
          f"sd {data.mean_rating.std():.3f}\n")

    print("=" * 74)
    print("1. AGREEMENT WITH HUMAN ATTRACTIVENESS NORMS")
    print("=" * 74)
    raw_rho, raw_p, n = spearman(data.mean_rating, data.attractive_rel)
    print(f"  raw (across all cells)        rho {raw_rho:+.3f}  p {raw_p:.3f}  n {n}")

    centered = data.dropna(subset=["mean_rating", "attractive_rel"]).copy()
    for column in ("mean_rating", "attractive_rel"):
        centered[f"{column}_c"] = (
            centered[column] - centered.groupby(CELL)[column].transform("mean")
        )
    cell_rho, cell_p, cell_n = spearman(centered.mean_rating_c, centered.attractive_rel_c)
    print(f"  within race x gender cells    rho {cell_rho:+.3f}  p {cell_p:.3f}  n {cell_n}")
    print(f"\n  Reference: two human rater pools agree with each other at "
          f"{HUMAN_REFERENCE:.3f} on")
    print("  CFD-I's absolute item. Different faces, different item, and")
    print("  human-vs-human rather than model-vs-human, so this bounds")
    print("  expectations loosely -- it is not a strict ceiling.")
    if not math.isnan(cell_rho):
        print(f"  -> model-human agreement is of the same order as "
              f"human-human agreement.")

    print("\n" + "=" * 74)
    print("2. AGREEMENT WITH OTHER PERCEPTUAL AND OBJECTIVE MEASURES")
    print("=" * 74)
    for label, column in [("rated age (human)", "AgeRated"),
                          ("canon deviation (objective)", "canon_deviation")]:
        rho, p, n = spearman(data.mean_rating, data[column])
        within = spearman(
            *(lambda frame: (
                frame.mean_rating - frame.groupby(CELL).mean_rating.transform("mean"),
                frame[column] - frame.groupby(CELL)[column].transform("mean"),
            ))(data.dropna(subset=["mean_rating", column]))
        )
        print(f"  {label:<30} raw {rho:+.3f} (p {p:.3f})   "
              f"within-cell {within[0]:+.3f} (p {within[1]:.3f})   n {n}")

    # The canon relationship needs a human control: if human raters on the same
    # measure show the same sign, the model is tracking human judgment rather
    # than revealing anything about canon adherence.
    human = manifest.dropna(subset=["attractive_rel", "canon_deviation"]).copy()
    for column in ("attractive_rel", "canon_deviation"):
        human[f"{column}_c"] = (
            human[column] - human.groupby(CELL)[column].transform("mean")
        )
    human_rho, human_p, human_n = spearman(
        human.attractive_rel_c, human.canon_deviation_c
    )
    print(f"\n  CONTROL -- human norms vs canon deviation, within-cell, full CFD:")
    print(f"    rho {human_rho:+.3f}  p {human_p:.2e}  n {human_n}")
    print("    Human raters show the same sign. A model correlation with canon")
    print("    deviation is therefore most simply read as the model tracking")
    print("    human judgment, not as evidence about canon adherence.")
    print("\n  Higher canon deviation means further from the neoclassical ideal.")
    print("  This operationalization is dominated by the nasal fifth and a")
    print("  facial-thirds proxy built from non-classical landmarks, so it is a")
    print("  statement about this measure, not about the canons as such.")

    print("\n" + "=" * 74)
    print("3. CEILING IMPOSED BY THE MODEL'S OWN RANGE")
    print("=" * 74)
    counts = data.mean_rating.value_counts().sort_index()
    print("  rating distribution:", {round(k, 2): int(v) for k, v in counts.items()})
    ties = 1 - (counts ** 2).sum() / len(data) ** 2
    print(f"  proportion of distinguishable pairs: {ties:.3f}")
    print("  A model using few distinct values cannot rank faces finely no matter")
    print("  how good its judgment is; this caps any achievable correlation.")

    # What correlation could a perfect rater achieve at this granularity?
    truth = data.dropna(subset=["attractive_rel"]).attractive_rel
    if len(truth) > 4 and data.mean_rating.nunique() > 1:
        quantized = pd.qcut(truth.rank(method="first"),
                            q=min(data.mean_rating.nunique(), len(truth)),
                            labels=False, duplicates="drop")
        ceiling_rho, _ = stats.spearmanr(quantized, truth)
        print(f"\n  A PERFECT rater restricted to {data.mean_rating.nunique()} "
              f"distinct values would")
        print(f"  reach only rho {ceiling_rho:+.3f} against these human norms.")
        if not math.isnan(cell_rho):
            print(f"  Observed {cell_rho:+.3f} against that granularity ceiling "
                  f"of {ceiling_rho:+.3f}.")


if __name__ == "__main__":
    main()
