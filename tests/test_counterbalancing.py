"""Counterbalancing must happen in log-odds, not probability, space.

The model picks the second image in ~99% of trials, so both raw probabilities
saturate near zero and averaging them in probability space collapses every
comparison to ~0.5. Position bias is additive in log-odds:

    logit P(pick first) = beta + (theta_first - theta_second)

so the two orders give beta + delta and beta - delta. Their difference recovers
the preference with beta cancelled; their sum estimates the bias.
"""
import math

import pytest

from facecav.analysis.bradley_terry import counterbalanced_preference


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def build(beta, delta):
    """Raw probabilities implied by a position bias and a true preference."""
    return sigmoid(beta + delta), sigmoid(beta - delta)


def test_identical_faces_give_no_preference():
    p_ab, p_ba = build(beta=-5.0, delta=0.0)
    assert counterbalanced_preference(p_ab, p_ba)["preference_a"] == pytest.approx(0.5)


def test_recovers_the_preference_despite_extreme_position_bias():
    # beta = -5 is the regime actually observed: P(pick first) ~ 0.7%.
    p_ab, p_ba = build(beta=-5.0, delta=1.4)
    result = counterbalanced_preference(p_ab, p_ba)
    assert result["preference_logit"] == pytest.approx(1.4, abs=1e-6)
    assert result["preference_a"] == pytest.approx(sigmoid(1.4), abs=1e-6)


def test_preference_is_unchanged_by_the_size_of_the_bias():
    mild = counterbalanced_preference(*build(beta=0.0, delta=0.8))
    severe = counterbalanced_preference(*build(beta=-6.0, delta=0.8))
    assert mild["preference_a"] == pytest.approx(severe["preference_a"], abs=1e-6)


def test_reports_the_position_bias_it_removed():
    result = counterbalanced_preference(*build(beta=-5.0, delta=1.4))
    assert result["position_logit"] == pytest.approx(-5.0, abs=1e-6)


def test_saturated_probabilities_still_separate_faces():
    # The real failure: probability-space averaging maps these onto ~0.489.
    result = counterbalanced_preference(0.0015099908923730254, 0.02460992895066738)
    assert result["preference_a"] < 0.25


def test_survives_probabilities_of_exactly_zero_or_one():
    result = counterbalanced_preference(0.0, 1.0)
    assert math.isfinite(result["preference_logit"])
    assert result["preference_a"] < 0.5
