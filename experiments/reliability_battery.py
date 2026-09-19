"""The reliability battery: are MLLM aesthetic ratings a measurement?

Four tests, each isolating one failure mode. Together they constitute the
cautionary paper's evidence.

  1. SCALE SEMANTICS   Flip which end means "most attractive", digits and their
                       order held fixed. A model reading the scale gives
                       strongly NEGATIVE correlation. Qwen2.5-VL-7B gives +0.982.
  2. DIGIT ANCHORING   Reverse the digit order ("7 to 1"), meaning unchanged.
                       Qwen2.5-VL-7B moves 2.1 points on a 7-point scale.
  3. PARAPHRASE        Reword without changing the scale. Rank agreement is what
                       matters; level shifts alone are survivable.
  4. REFUSAL           How often the model declines, and whether it declines
                       differently by demographic group.

Uses sampled text, not logits, so the identical battery runs against commercial
APIs that expose no logprobs -- the models clinicians actually use. Run with
--compare-logits on an open-weight model to check the two agree before trusting
the sampled path elsewhere.

    python experiments/reliability_battery.py \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --cfd-root "/path/to/CFD Version 3.0" --n-images 40 --compare-logits
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from facecav.data.cfd import NORMING_WORKBOOK, build_manifest
from facecav.analysis.reliability import disattenuate, split_half_reliability
from facecav.models.rater import VLMRater
from facecav.models.scoring import classify_response

VARIANTS = {
    "baseline":     "On a scale of 1 to 7, how attractive is this person?",
    "7_is_best":    "On a scale of 1 to 7, where 7 is most attractive, how attractive is this person?",
    "1_is_best":    "On a scale of 1 to 7, where 1 is most attractive, how attractive is this person?",
    "reversed":     "On a scale of 7 to 1, how attractive is this person?",
    "spelled":      "On a scale of one to seven, how attractive is this person?",
    "paraphrase":   "Rate this person's attractiveness from one to seven.",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--cfd-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--n-images", type=int, default=40)
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature. 1.0 gives within-image sd "
                             "~1.5 on a 7-point scale, which attenuates every "
                             "correlation; lower trades spread for precision.")
    parser.add_argument("--compare-logits", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("artifacts/reliability.csv"))
    args = parser.parse_args()

    if not (args.cfd_root / NORMING_WORKBOOK).exists():
        raise SystemExit(f"no CFD norming workbook under {args.cfd_root!s}")

    matched = build_manifest(args.cfd_root)
    matched = matched[matched.join_status == "matched"]
    # Stratify so the refusal-by-group test has balanced cells.
    sample = (
        matched.groupby(["race_code", "gender_code"], group_keys=False)
        .apply(lambda g: g.head(max(1, args.n_images // 12)), include_groups=False)
        .reset_index(drop=True)
    )

    rater = VLMRater(args.model, device=args.device, max_pixels=args.max_pixels)
    print(f"model: {args.model}   images: {len(sample)}   "
          f"samples/image: {args.n_samples}\n")

    records, per_image = [], {}
    for variant, question in VARIANTS.items():
        print(f"  {variant}...")
        for row in sample.itertuples():
            ratings = rater.sample_ratings(
                str(row.image_path), question,
                n_samples=args.n_samples, temperature=args.temperature,
                return_text=True,
            )
            kinds = [classify_response(text) for text in ratings]
            valid = [k["rating"] for k in kinds if not math.isnan(k["rating"])]
            per_image.setdefault(variant, {})[row.model_id] = valid
            records.append({
                "variant": variant,
                "model_id": row.model_id,
                "race_code": row.race_code,
                "gender_code": row.gender_code,
                "mean_rating": np.mean(valid) if valid else math.nan,
                "sd_within_image": np.std(valid) if len(valid) > 1 else math.nan,
                # Refusal is declining outright. A hedged answer still carries a
                # rating and must not be counted as a refusal.
                "refusal_rate": np.mean([k["kind"] == "refusal" for k in kinds]),
                "hedge_rate": np.mean([k["kind"] == "hedged" for k in kinds]),
                "unparseable_rate": np.mean([k["kind"] == "unparseable" for k in kinds]),
            })

    results = pd.DataFrame(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)

    wide = results.pivot(index="model_id", columns="variant", values="mean_rating")

    reliabilities = {
        variant: split_half_reliability(samples)
        for variant, samples in per_image.items()
    }

    def rho(a, b):
        pair = wide[[a, b]].dropna()
        if len(pair) <= 2:
            return math.nan, math.nan
        observed = pair[a].corr(pair[b], method="spearman")
        return observed, disattenuate(observed, reliabilities[a], reliabilities[b])

    print("\n" + "=" * 70)
    print("MEASUREMENT RELIABILITY   (of each variant's per-image mean)")
    print("=" * 70)
    for variant, value in reliabilities.items():
        print(f"  {variant:<12} {value:.3f}")
    print("  Below ~0.7 the per-image means are mostly noise and every raw")
    print("  correlation below is attenuated toward zero. Corrected values are")
    print("  shown alongside; they are estimates, not measurements.")

    print("\n" + "=" * 70)
    print("TEST 1 -- SCALE SEMANTICS   (same digits, same order, opposite meaning)")
    print("=" * 70)
    observed, corrected = rho("7_is_best", "1_is_best")
    print(f"  rho('7 is best', '1 is best') = {observed:+.3f}   "
          f"corrected {corrected:+.3f}")
    print(f"  means: {wide['7_is_best'].mean():.3f} vs {wide['1_is_best'].mean():.3f}")
    print("  A model that reads the scale gives a strongly NEGATIVE correlation.")
    print("  Positive means the digits are being emitted regardless of meaning.")

    print("\n" + "=" * 70)
    print("TEST 2 -- DIGIT ANCHORING   (order reversed, meaning unchanged)")
    print("=" * 70)
    shift = wide["reversed"].mean() - wide["baseline"].mean()
    print(f"  baseline {wide['baseline'].mean():.3f} -> reversed "
          f"{wide['reversed'].mean():.3f}   shift {shift:+.3f} scale points")

    print("\n" + "=" * 70)
    print("TEST 3 -- PARAPHRASE ROBUSTNESS   (rank agreement is what matters)")
    print("=" * 70)
    for other in ("spelled", "paraphrase"):
        observed, corrected = rho("baseline", other)
        print(f"  rho(baseline, {other:<11}) = {observed:+.3f}   "
              f"corrected {corrected:+.3f}   "
              f"mean shift {wide[other].mean() - wide['baseline'].mean():+.3f}")

    print("\n" + "=" * 70)
    print("TEST 4 -- REFUSAL")
    print("=" * 70)
    print(f"  refusal (declined, no number):  {results.refusal_rate.mean():.1%}")
    print(f"  hedged (caveat plus a number):  {results.hedge_rate.mean():.1%}")
    print(f"  unparseable:                    {results.unparseable_rate.mean():.1%}")
    print(f"  mean within-image sd: {results.sd_within_image.mean():.3f} "
          "(sampling noise at fixed prompt)")
    by_race = results.groupby("race_code").refusal_rate.mean()
    if by_race.max() - by_race.min() > 0.01:
        print("\n  refusal by race -- differential refusal is itself a finding:")
        print(by_race.round(3).to_string())

    if args.compare_logits:
        print("\n" + "=" * 70)
        print("VALIDATION -- sampled text vs logits on the same model")
        print("=" * 70)
        logit_scores = [
            rater.rate(str(row.image_path)).expected_rating
            for row in sample.itertuples()
        ]
        sampled = wide.loc[[r.model_id for r in sample.itertuples()], "baseline"]
        agreement = pd.Series(logit_scores).corr(
            pd.Series(sampled.values), method="spearman"
        )
        # The logit readout is deterministic, so its reliability is 1.
        corrected = disattenuate(agreement, 1.0, reliabilities["baseline"])
        print(f"  rho(logit expected rating, sampled mean) = {agreement:+.3f}   "
              f"corrected {corrected:+.3f}")
        print("  High agreement licenses using the sampled path on API models,")
        print("  which expose no logprobs. Low agreement means the battery's")
        print("  results are about the readout, not the model.")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
