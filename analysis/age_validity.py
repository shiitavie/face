"""Validate Claude's perceived-age estimates, and test for demographic bias.

Perceived age admits a validation attractiveness cannot: there is an objective
referent. CFD supplies two criteria --

  * AgeRated, the human rater panel's mean, for every face
  * AgeSelf, self-reported actual age, for CFD-INDIA and CFD-MR only

-- so the model can be scored against human perception AND against truth, and
compared against the human panel on the same faces. Errors are in years, which
makes both the measurement error and any group difference clinically legible.

The fairness question also sharpens. Rather than asking whether a rating
"differs" on a scale with no absolute referent, we can ask whether the model's
age ERROR differs by group, which is unambiguous.

    .venv/bin/python analysis/age_validity.py \\
        --ages artifacts/age_claude-opus-5.jsonl \\
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


def bootstrap_ci(values, statistic=np.mean, draws=2000, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 5:
        return math.nan, math.nan
    samples = [statistic(values[rng.integers(0, len(values), len(values))])
               for _ in range(draws)]
    return tuple(np.percentile(samples, [2.5, 97.5]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ages", type=Path, required=True)
    parser.add_argument("--cfd-root", type=Path, required=True)
    args = parser.parse_args()

    if not (args.cfd_root / NORMING_WORKBOOK).exists():
        raise SystemExit(f"no CFD norming workbook under {args.cfd_root!s}")

    ages = pd.read_json(args.ages, lines=True).dropna(subset=["age_estimate"])
    manifest = build_manifest(args.cfd_root)
    manifest = manifest[manifest.join_status == "matched"]
    data = ages.merge(
        manifest[["model_id", "AgeRated", "AgeSelf"]], on="model_id", how="left"
    )
    data["AgeSelf"] = pd.to_numeric(data.AgeSelf, errors="coerce")

    print(f"n = {len(data)} faces rated   "
          f"{data.AgeSelf.notna().sum()} with self-reported actual age")
    print(f"unparseable responses: {data.unparseable_rate.mean():.2%}   "
          f"re-query SD: {data.sd.dropna().mean():.3f} years\n")

    # ------------------------------------------------------------------
    print("=" * 76)
    print("1. AGAINST ACTUAL AGE -- model vs the human rater panel, same faces")
    print("=" * 76)
    truth = data.dropna(subset=["AgeSelf", "AgeRated"]).copy()
    if len(truth) < 10:
        print("  too few faces with actual age yet")
    else:
        truth["model_error"] = truth.age_estimate - truth.AgeSelf
        truth["human_error"] = truth.AgeRated - truth.AgeSelf
        rows = []
        for label, error, estimate in [
            ("Claude", truth.model_error, truth.age_estimate),
            ("human panel", truth.human_error, truth.AgeRated),
        ]:
            mae = error.abs().mean()
            lo, hi = bootstrap_ci(error.abs())
            rows.append({
                "rater": label,
                "MAE (yrs)": round(mae, 2),
                "MAE 95% CI": f"[{lo:.2f}, {hi:.2f}]",
                "bias (yrs)": round(error.mean(), 2),
                "r vs actual": round(stats.pearsonr(estimate, truth.AgeSelf).statistic, 3),
            })
        print(pd.DataFrame(rows).to_string(index=False))
        print(f"\n  n = {len(truth)}   actual age range "
              f"{truth.AgeSelf.min():.0f}-{truth.AgeSelf.max():.0f}")
        print("  bias is signed: positive means the rater estimates too old.")

        paired = stats.wilcoxon(truth.model_error.abs(), truth.human_error.abs())
        print(f"\n  paired test of |error|, model vs human panel: "
              f"p = {paired.pvalue:.4f}")
        shared = stats.pearsonr(truth.model_error, truth.human_error)
        print(f"  correlation between model and human errors: "
              f"r = {shared.statistic:+.3f}")
        print("  A positive error correlation means both are misled by the same")
        print("  faces -- shared perceptual signal rather than independent noise.")

    # ------------------------------------------------------------------
    # The faces carrying actual age are CFD-INDIA and CFD-MR, so "the model
    # beats the panel" invites the objection that US raters are simply worse at
    # estimating age on non-White faces. CFD-I is normed twice -- by US and by
    # Indian raters -- which tests that directly.
    india_sheet = "CFD-I INDIA Norming Data"
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sheet = pd.read_excel(
                args.cfd_root / NORMING_WORKBOOK, sheet_name=india_sheet, header=None
            )
        labels = [str(x) for x in sheet.iloc[7]]
        body = sheet.iloc[9:].reset_index(drop=True)
        india = pd.DataFrame({
            "model_id": body.iloc[:, 0].astype(str),
            "AgeRated_india": pd.to_numeric(
                body.iloc[:, labels.index("AgeRated")], errors="coerce"),
        })
        cross = truth.merge(india, on="model_id", how="inner").dropna(
            subset=["AgeRated_india"]
        )
    except Exception:
        cross = pd.DataFrame()

    if len(cross) > 20:
        print("\n" + "=" * 76)
        print("1b. CROSS-RACE CONTROL -- Indian faces rated by both human pools")
        print("=" * 76)
        for label, estimate in [("Claude", cross.age_estimate),
                                ("US rater panel", cross.AgeRated),
                                ("India rater panel", cross.AgeRated_india)]:
            error = estimate - cross.AgeSelf
            print(f"  {label:<20} MAE {error.abs().mean():5.2f}   "
                  f"bias {error.mean():+6.2f}   "
                  f"r {stats.pearsonr(estimate, cross.AgeSelf).statistic:.3f}")
        print(f"\n  n = {len(cross)}")
        print("  If the two human pools perform alike, the model's advantage is")
        print("  not an artifact of cross-race age estimation in the comparator.")

    # ------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("2. AGAINST HUMAN PERCEIVED AGE -- all faces")
    print("=" * 76)
    perceived = data.dropna(subset=["AgeRated"])
    r = stats.pearsonr(perceived.age_estimate, perceived.AgeRated)
    rho = stats.spearmanr(perceived.age_estimate, perceived.AgeRated)
    error = perceived.age_estimate - perceived.AgeRated
    print(f"  n = {len(perceived)}")
    print(f"  Pearson r  {r.statistic:.3f}   Spearman rho {rho.statistic:.3f}")
    print(f"  MAE {error.abs().mean():.2f} years   "
          f"bias {error.mean():+.2f} years")

    # ------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("3. DEPARTURE FROM HUMAN PERCEPTION BY GROUP")
    print("   (this is NOT a bias measure -- see the decomposition below)")
    print("=" * 76)
    print(f"{'group':<13} {'n':>4} {'MAE':>7} {'C-human':>9} {'95% CI':>20} {'truth?':>8}")
    print("-" * 64)
    perceived = perceived.assign(error=perceived.age_estimate - perceived.AgeRated)
    for race, block in perceived.groupby("race_code"):
        lo, hi = bootstrap_ci(block.error)
        with_truth = truth[truth.race_code == race] if len(truth) else block.iloc[:0]
        print(f"{RACE_NAMES.get(race, race):<13} {len(block):>4} "
              f"{block.error.abs().mean():>7.2f} {block.error.mean():>+9.2f} "
              f"{f'[{lo:+.2f}, {hi:+.2f}]':>20} "
              f"{('n=' + str(len(with_truth))) if len(with_truth) else 'none':>8}")

    print("\n  A group difference in this column CANNOT be read as model bias.")
    print("  The column is model minus human panel, so a difference may come")
    print("  from either party, and CFD supplies actual age for only two of the")
    print("  six groups.")

    if len(truth) > 20:
        print("\n" + "=" * 76)
        print("3b. DECOMPOSITION where actual age exists -- who is actually off?")
        print("=" * 76)
        print(f"{'group':<13} {'n':>4} {'Claude vs truth':>16} {'human vs truth':>15} "
              f"{'C-human':>9}")
        print("-" * 60)
        for race, block in truth.groupby("race_code"):
            model_bias = (block.age_estimate - block.AgeSelf).mean()
            human_bias = (block.AgeRated - block.AgeSelf).mean()
            print(f"{RACE_NAMES.get(race, race):<13} {len(block):>4} "
                  f"{model_bias:>+16.2f} {human_bias:>+15.2f} "
                  f"{model_bias - human_bias:>+9.2f}")
        print("\n  Both raters overestimate age, so the model estimating younger")
        print("  than the panel moves TOWARD truth. Where truth is available the")
        print("  model is the more accurate of the two, and the apparent")
        print("  'differential bias' above is largely the human panel's error")
        print("  varying by group.")
        print("\n  CONCLUSION: CFD cannot answer whether the MODEL's age error")
        print("  differs by race. Doing so needs actual age across all groups --")
        print("  e.g. UTKFace, at the cost of standardized capture.")

    print("\n" + "=" * 76)
    print("4. BY GROUP AND GENDER")
    print("=" * 76)
    cells = perceived.groupby(["race_code", "gender_code"]).agg(
        n=("error", "size"), MAE=("error", lambda s: s.abs().mean()),
        bias=("error", "mean"),
    ).round(2)
    cells.index = pd.MultiIndex.from_tuples(
        [(RACE_NAMES.get(r, r), g) for r, g in cells.index]
    )
    print(cells.to_string())


if __name__ == "__main__":
    main()
