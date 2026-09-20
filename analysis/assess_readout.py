"""Is the sampled readout precise enough to measure anything?

Each per-image mean is a noisy estimate of that face's true rating. Observed
variance across images is therefore true variance PLUS sampling error, and every
correlation computed from these means is attenuated by the ratio between them.

    reliability = tau^2 / (tau^2 + sigma^2 / n_effective)

where tau is the true between-image sd and sigma the within-image sd. Refusals
reduce n_effective, so they cost precision as well as data.

    python analysis/assess_readout.py --ratings artifacts/reliability.csv --n-samples 16
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from facecav.analysis.reliability import disattenuate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", type=Path, default=Path("artifacts/reliability.csv"))
    parser.add_argument("--n-samples", type=int, required=True,
                        help="Samples drawn per image in that run.")
    parser.add_argument("--target-reliability", type=float, default=0.8)
    args = parser.parse_args()

    data = pd.read_csv(args.ratings)
    rows = []
    for variant, group in data.groupby("variant"):
        effective = args.n_samples * (1 - group.refusal_rate.mean())
        within = group.sd_within_image.mean()
        observed_var = group.mean_rating.var()
        error_var = within**2 / effective
        true_var = max(observed_var - error_var, 0.0)
        reliability = true_var / observed_var if observed_var > 0 else math.nan
        rows.append({
            "variant": variant,
            "within_sd": within,
            "n_eff": effective,
            "SE": math.sqrt(error_var),
            "true_sd": math.sqrt(true_var),
            "reliability": reliability,
        })
    summary = pd.DataFrame(rows).set_index("variant")

    print("VARIANCE DECOMPOSITION")
    print("=" * 74)
    print(summary.round(3).to_string())
    print("\n  true_sd is the real spread between faces; SE is the error on each")
    print("  estimate. reliability below ~0.7 means the means are mostly noise.")

    wide = data.pivot(index="model_id", columns="variant", values="mean_rating")
    print("\nKEY CORRELATIONS, OBSERVED AND DISATTENUATED")
    print("=" * 74)
    print(f"{'comparison':<28} {'observed':>9} {'corrected':>10}")
    print("-" * 50)
    for a, b in [("7_is_best", "1_is_best"), ("baseline", "spelled"),
                 ("baseline", "paraphrase"), ("baseline", "reversed")]:
        if a not in wide or b not in wide:
            continue
        pair = wide[[a, b]].dropna()
        observed = pair[a].corr(pair[b], method="spearman")
        corrected = disattenuate(
            observed, summary.loc[a, "reliability"], summary.loc[b, "reliability"]
        )
        print(f"{a + ' vs ' + b:<28} {observed:>+9.3f} {corrected:>+10.3f}")

    print("\nWHAT WOULD MAKE THIS USABLE")
    print("=" * 74)
    tau2 = summary["true_sd"].median() ** 2
    target = args.target_reliability
    print(f"  median true between-image variance: {tau2:.4f}")
    for within in (1.5, 1.0, 0.75, 0.5, 0.25):
        needed = within**2 / (tau2 * (1 / target - 1)) if tau2 > 0 else math.inf
        print(f"  within_sd {within:>4.2f}  ->  {needed:>6.0f} effective samples "
              f"for reliability {target}")
    print("\n  Lowering temperature shrinks within_sd; more samples only helps as")
    print("  the square root, so temperature is by far the cheaper lever.")


if __name__ == "__main__":
    main()
