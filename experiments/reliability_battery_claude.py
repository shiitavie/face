"""The reliability battery against Claude, via the Anthropic API.

Mirrors experiments/reliability_battery.py so results are comparable, with three
deviations forced by the API and reported as such:

  * no assistant prefill (rejected on Claude 5) -- a system instruction is used
  * no temperature control (removed on Opus 5 / Sonnet 5) -- the lever that took
    the local readout from reliability 0.27 to 0.95 is unavailable here
  * no log-probabilities -- the sampled readout only, validated locally at 0.877

Prints a cost estimate and requires --yes before spending anything.

    export ANTHROPIC_API_KEY=...        # or: ant auth login
    python experiments/reliability_battery_claude.py \
        --cfd-root "/path/to/CFD Version 3.0" --n-images 40 --n-samples 16 --yes
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from facecav.analysis.reliability import split_half_reliability
from facecav.data.cfd import NORMING_WORKBOOK, build_manifest
from facecav.models.claude_backend import DEFAULT_MAX_DIMENSION, ClaudeRater

# Identical wording to the local battery.
VARIANTS = {
    "baseline":   "On a scale of 1 to 7, how attractive is this person?",
    "7_is_best":  "On a scale of 1 to 7, where 7 is most attractive, how attractive is this person?",
    "1_is_best":  "On a scale of 1 to 7, where 1 is most attractive, how attractive is this person?",
    "reversed":   "On a scale of 7 to 1, how attractive is this person?",
    "spelled":    "On a scale of one to seven, how attractive is this person?",
    "paraphrase": "Rate this person's attractiveness from one to seven.",
}

PRICES = {  # per million tokens, (input, output)
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--cfd-root", type=Path, required=True)
    parser.add_argument("--n-images", type=int, default=40)
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--max-dimension", type=int, default=DEFAULT_MAX_DIMENSION)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--variants", nargs="+", default=None, choices=list(VARIANTS))
    parser.add_argument("--yes", action="store_true", help="Proceed without confirming the cost.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not (args.cfd_root / NORMING_WORKBOOK).exists():
        raise SystemExit(f"no CFD norming workbook under {args.cfd_root!s}")

    matched = build_manifest(args.cfd_root)
    matched = matched[matched.join_status == "matched"]
    sample = matched.groupby(["race_code", "gender_code"]).head(
        max(1, args.n_images // 12)
    ).reset_index(drop=True)

    selected = {k: VARIANTS[k] for k in (args.variants or VARIANTS)}
    calls = len(selected) * len(sample) * args.n_samples
    image_tokens = (args.max_dimension ** 2 * 0.7) // 750
    writes = len(selected) * len(sample)
    reads = calls - writes
    input_price, _ = PRICES.get(args.model, (5.0, 25.0))
    estimate = (writes * image_tokens * 1.25 + reads * image_tokens * 0.1) * input_price / 1e6

    print(f"model      {args.model}")
    print(f"images     {len(sample)}   variants {len(selected)}   samples {args.n_samples}")
    print(f"calls      {calls:,}   ~{int(image_tokens)} image tokens each")
    print(f"estimate   ${estimate:.2f}  (image cached per image-variant)\n")
    if not args.yes:
        raise SystemExit("re-run with --yes to proceed (this spends money)")

    rater = ClaudeRater(args.model, max_dimension=args.max_dimension)

    out = args.out or Path(f"artifacts/reliability_{args.model.replace('/', '__')}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Append-only JSONL alongside the CSV: this run costs money, so a transient
    # 529 partway through must not mean paying for the whole thing again.
    journal = out.with_suffix(".jsonl")
    done = set()
    if journal.exists():
        with journal.open() as handle:
            for line in handle:
                record = json.loads(line)
                done.add((record["variant"], record["model_id"]))
        print(f"  resuming: {len(done)} image-variants already recorded\n")

    todo = [
        (variant, row)
        for variant in selected
        for row in sample.itertuples()
        if (variant, row.model_id) not in done
    ]

    with journal.open("a") as handle:
        for n, (variant, row) in enumerate(todo, start=1):
            kinds = rater.sample_ratings(
                str(row.image_path), selected[variant],
                n_samples=args.n_samples, concurrency=args.concurrency,
            )
            valid = [k["rating"] for k in kinds if not math.isnan(k["rating"])]
            record = {
                "model": args.model,
                "variant": variant,
                "model_id": row.model_id,
                "race_code": row.race_code,
                "gender_code": row.gender_code,
                "ratings": valid,
                "mean_rating": float(np.mean(valid)) if valid else None,
                "sd_within_image": float(np.std(valid)) if len(valid) > 1 else None,
                "refusal_rate": float(np.mean([k["kind"] == "refusal" for k in kinds])),
                "hedge_rate": float(np.mean([k["kind"] == "hedged" for k in kinds])),
                "unparseable_rate": float(np.mean([k["kind"] == "unparseable" for k in kinds])),
            }
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            if n % 10 == 0 or n == len(todo):
                print(f"    {n}/{len(todo)}   spent this session "
                      f"${rater.usage.cost(*PRICES.get(args.model, (5.0, 25.0))):.2f}")

    results = pd.read_json(journal, lines=True)
    per_image = {
        variant: dict(zip(block.model_id, block.ratings))
        for variant, block in results.groupby("variant")
    }
    results.drop(columns=["ratings"]).to_csv(out, index=False)

    wide = results.pivot(index="model_id", columns="variant", values="mean_rating")
    reliabilities = {v: split_half_reliability(s) for v, s in per_image.items()}

    print("\n" + "=" * 68)
    print(f"RELIABILITY -- {args.model}")
    print("=" * 68)
    for variant, value in reliabilities.items():
        within = results[results.variant == variant].sd_within_image.mean()
        print(f"  {variant:<12} reliability {value:.3f}   within-image sd {within:.3f}")
    print("\n  Temperature cannot be set on this model, so the sampling noise")
    print("  above is whatever the API gives -- there is no lever to reduce it.")

    def rho(a, b):
        if not {a, b} <= set(wide.columns):
            return math.nan
        pair = wide[[a, b]].dropna()
        return pair[a].corr(pair[b], method="spearman") if len(pair) > 2 else math.nan

    print("\n" + "=" * 68)
    print("TESTS")
    print("=" * 68)
    print(f"  scale semantics  rho(7_is_best, 1_is_best) = {rho('7_is_best', '1_is_best'):+.3f}")
    if {"baseline", "reversed"} <= set(wide.columns):
        print(f"  digit anchoring  baseline {wide['baseline'].mean():.3f} -> "
              f"reversed {wide['reversed'].mean():.3f}  "
              f"shift {wide['reversed'].mean() - wide['baseline'].mean():+.3f}")
    for other in ("spelled", "paraphrase"):
        print(f"  paraphrase       rho(baseline, {other:<11}) = {rho('baseline', other):+.3f}")
    print(f"\n  refusal {results.refusal_rate.mean():.1%}   "
          f"hedged {results.hedge_rate.mean():.1%}   "
          f"unparseable {results.unparseable_rate.mean():.1%}")

    spent = rater.usage.cost(*PRICES.get(args.model, (5.0, 25.0)))
    print(f"\n  actual spend ${spent:.2f}  "
          f"(cache reads {rater.usage.cache_read_tokens:,} tokens)")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
