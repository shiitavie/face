"""Compare measurands across vendors, and against the human panel.

A finding on one model says little about multimodal models as a class, and two
models from the same lab share training data, RLHF and architecture. These
comparisons therefore pair Anthropic against OpenAI.

Three things it reports that a single-model analysis cannot:

* whether the model-beats-humans result on perceived age holds on an
  independent training lineage;
* whether the skin-tone group bias replicates, tested inside a matched-truth
  band so the regression-to-the-mean artifact that destroyed the brow-position
  result cannot explain it;
* how closely the two models agree with EACH OTHER. Agreement higher than
  either model's agreement with truth means their errors are shared rather
  than independent, which points at a common cause.

    .venv/bin/python analysis/cross_vendor.py --cfd-root "dataset/CFD Version 3.0"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy import stats

from facecav.data.cfd import NORMING_WORKBOOK, build_manifest

RACE_NAMES = {"A": "Asian", "L": "Latino", "W": "White"}
#: Band every well-represented group occupies, so position on the scale is held
#: fixed and a residual difference cannot be positional.
MATCHED_BAND = (140, 180)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfd-root", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    if not (args.cfd_root / NORMING_WORKBOOK).exists():
        raise SystemExit(f"no CFD norming workbook under {args.cfd_root!s}")

    manifest = build_manifest(args.cfd_root)
    manifest = manifest[manifest.join_status == "matched"].copy()
    manifest["AgeSelf"] = pd.to_numeric(manifest.AgeSelf, errors="coerce")

    def load(name: str, column: str) -> pd.DataFrame | None:
        path = args.artifacts / name
        if not path.exists():
            return None
        frame = pd.read_json(path, lines=True).dropna(subset=[column])
        return frame.rename(columns={column: "value"})[["model_id", "value"]]

    # ------------------------------------------------------------------
    print("=" * 74)
    print("PERCEIVED AGE -- against self-reported actual age")
    print("=" * 74)
    print(f"{'rater':<22} {'n':>4} {'MAE (yrs)':>10} {'bias':>7} {'r':>7} {'slope':>7}")
    print("-" * 62)

    sources = [
        ("Claude Opus 5", "age_claude-opus-5.jsonl", "age_estimate"),
        ("GPT-5.1", "age_gpt-5.1.jsonl", "score"),
    ]
    loaded = {}
    for label, name, column in sources:
        frame = load(name, column)
        if frame is None:
            continue
        joined = frame.merge(
            manifest[["model_id", "AgeSelf", "AgeRated"]], on="model_id"
        ).dropna(subset=["AgeSelf"])
        loaded[label] = joined
        error = joined.value - joined.AgeSelf
        print(f"{label:<22} {len(joined):>4} {error.abs().mean():>10.2f} "
              f"{error.mean():>+7.2f} "
              f"{stats.pearsonr(joined.value, joined.AgeSelf).statistic:>7.3f} "
              f"{stats.linregress(joined.AgeSelf, joined.value).slope:>7.3f}")

    if loaded:
        panel = next(iter(loaded.values()))
        error = panel.AgeRated - panel.AgeSelf
        print(f"{'human panel':<22} {len(panel):>4} {error.abs().mean():>10.2f} "
              f"{error.mean():>+7.2f} "
              f"{stats.pearsonr(panel.AgeRated, panel.AgeSelf).statistic:>7.3f} "
              f"{stats.linregress(panel.AgeSelf, panel.AgeRated).slope:>7.3f}")

    if len(loaded) == 2:
        first, second = (load(n, c) for _, n, c in sources)
        both = first.merge(second, on="model_id", suffixes=("_a", "_b")).dropna()
        agreement = stats.pearsonr(both.value_a, both.value_b).statistic
        print(f"\n  the two models agree with EACH OTHER at r = {agreement:.3f} "
              f"(n = {len(both)})")
        print("  Higher than either model's agreement with truth means the")
        print("  errors are shared, not independent -- a common cause rather")
        print("  than noise.")

    # ------------------------------------------------------------------
    print("\n" + "=" * 74)
    print(f"SKIN TONE BIAS -- matched-luminance band {MATCHED_BAND}")
    print("=" * 74)
    residuals = {}
    for label, name in [("Claude", "skin_tone_claude-opus-5.jsonl"),
                        ("GPT-5.1", "skin_tone_gpt-5.1.jsonl")]:
        frame = load(name, "score")
        if frame is None:
            continue
        joined = frame.merge(
            manifest[["model_id", "LuminanceMedian", "race_code"]], on="model_id"
        ).dropna(subset=["LuminanceMedian"])
        slope, intercept, *_ = stats.linregress(joined.value, joined.LuminanceMedian)
        joined = joined.assign(
            residual=joined.LuminanceMedian - (slope * joined.value + intercept)
        )
        band = joined[
            joined.race_code.isin(RACE_NAMES)
            & joined.LuminanceMedian.between(*MATCHED_BAND)
        ]
        residuals[label] = band.groupby("race_code").residual.mean()

    if residuals:
        header = "".join(f"{k:>15}" for k in residuals)
        print(f"{'group':<10}{header}")
        print("-" * (10 + 15 * len(residuals)))
        for race in ("L", "A", "W"):
            row = "".join(f"{residuals[k].get(race, float('nan')):>+15.2f}"
                          for k in residuals)
            print(f"{RACE_NAMES[race]:<10}{row}")
        print()
        for label, series in residuals.items():
            if "L" in series and "W" in series:
                print(f"  {label}: Latino-White gap {series['L'] - series['W']:+.2f} "
                      f"luminance units")
        print("\n  Residuals compared inside a band where true lightness is")
        print("  matched, so the positional artifact that explained the")
        print("  brow-position group differences cannot explain these.")


if __name__ == "__main__":
    main()
