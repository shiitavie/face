"""Measurement reliability and attenuation correction.

Sampled text ratings are noisy: at temperature 1.0 the within-image standard
deviation is ~1.5 on a 7-point scale, so the mean of 16 samples carries a
standard error near 0.38. Every correlation computed from such means is dragged
toward zero, and comparisons against an exact readout are not like for like --
the same paraphrase comparison measured 0.965 by logits and 0.203 by sampling.

Reporting both the observed and the disattenuated correlation, plus the
reliability used, is the honest way to present a noisy instrument.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np


def disattenuate(observed: float, reliability_x: float, reliability_y: float) -> float:
    """Spearman's correction for attenuation.

    ``observed / sqrt(rel_x * rel_y)``, capped at 1.0 -- sampling error can push
    the raw estimate past unity, which is an artifact of estimation rather than
    a correlation above one.
    """
    if reliability_x <= 0 or reliability_y <= 0:
        return math.nan
    corrected = observed / math.sqrt(reliability_x * reliability_y)
    return float(np.clip(corrected, -1.0, 1.0))


def split_half_reliability(
    samples: Mapping[str, Sequence[float]],
    seed: int = 0,
) -> float:
    """Reliability of a per-item mean, by Spearman-Brown on split halves.

    Splits each item's repeated samples in two, correlates the half-means across
    items, and steps the result up to the full-length measure. Estimates how much
    of the observed between-item variance is signal rather than sampling noise.
    """
    rng = np.random.default_rng(seed)
    first, second = [], []
    for values in samples.values():
        usable = [v for v in values if not math.isnan(v)]
        if len(usable) < 2:
            continue
        shuffled = rng.permutation(usable)
        midpoint = len(shuffled) // 2
        first.append(np.mean(shuffled[:midpoint]))
        second.append(np.mean(shuffled[midpoint:]))

    if len(first) < 3:
        return math.nan
    half = np.corrcoef(first, second)[0, 1]
    if math.isnan(half):
        return math.nan
    # Spearman-Brown: a half-length measure understates the full measure.
    return float(np.clip(2 * half / (1 + half), 0.0, 1.0))
