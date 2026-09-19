"""Correcting correlations for measurement error.

Sampled ratings carry large within-image noise (sd 1.52 on a 7-point scale at
temperature 1.0), which drags every observed correlation toward zero. The
paraphrase correlation measured 0.965 by logits and 0.203 by sampling on the
same model and images -- the gap is attenuation, not a different finding.
"""
import pytest

from facecav.analysis.reliability import (
    disattenuate,
    split_half_reliability,
)


def test_perfect_reliability_leaves_the_correlation_unchanged():
    assert disattenuate(0.5, 1.0, 1.0) == pytest.approx(0.5)


def test_noisy_measures_imply_a_larger_true_correlation():
    assert disattenuate(0.3, 0.5, 0.5) == pytest.approx(0.6)


def test_correction_is_capped_at_one():
    # Noise can make the raw estimate exceed 1 after correction; that is an
    # estimation artifact, not a correlation above unity.
    assert disattenuate(0.9, 0.3, 0.3) == pytest.approx(1.0)


def test_zero_reliability_cannot_be_corrected():
    import math
    assert math.isnan(disattenuate(0.3, 0.0, 0.5))


def test_split_half_reliability_is_high_for_consistent_samples():
    # Each image sampled repeatedly with little noise.
    samples = {f"img{i}": [float(i % 7 + 1)] * 16 for i in range(30)}
    assert split_half_reliability(samples) > 0.95


def test_split_half_reliability_is_low_for_pure_noise():
    import random
    rng = random.Random(0)
    samples = {f"img{i}": [rng.uniform(1, 7) for _ in range(16)] for i in range(30)}
    assert split_half_reliability(samples) < 0.5
