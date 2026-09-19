"""Bradley-Terry fitting over counterbalanced pairwise preferences.

The comparisons carry graded preference probabilities, not binary wins:
counterbalancing yields P(i preferred over j) in [0, 1]. Thresholding those to
wins would discard the precision that made the instrument usable, so the fit
works on the probabilities directly.
"""
import numpy as np
import pytest

from facecav.analysis.bradley_terry import fit_bradley_terry


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def test_recovers_the_ordering_of_three_items():
    # c beats b beats a
    comparisons = [("a", "b", 0.2), ("b", "c", 0.2), ("a", "c", 0.1)]
    strengths = fit_bradley_terry(comparisons)
    assert strengths["c"] > strengths["b"] > strengths["a"]


def test_recovers_true_strengths_up_to_a_constant():
    truth = {"a": -1.0, "b": 0.0, "c": 1.5, "d": 0.5}
    comparisons = [
        (i, j, sigmoid(truth[i] - truth[j]))
        for i in truth for j in truth if i < j
    ]
    fitted = fit_bradley_terry(comparisons, regularization=0.0)

    centered = {k: v - np.mean(list(truth.values())) for k, v in truth.items()}
    for item, expected in centered.items():
        assert fitted[item] == pytest.approx(expected, abs=0.05)


def test_strengths_are_mean_centered_for_identifiability():
    # Only differences are identifiable, so the fit must pin the level.
    comparisons = [("a", "b", 0.9), ("b", "c", 0.7)]
    strengths = fit_bradley_terry(comparisons)
    assert np.mean(list(strengths.values())) == pytest.approx(0.0, abs=1e-6)


def test_indistinguishable_items_get_equal_strengths():
    comparisons = [("a", "b", 0.5), ("b", "c", 0.5), ("a", "c", 0.5)]
    strengths = fit_bradley_terry(comparisons)
    assert np.allclose(list(strengths.values()), 0.0, atol=1e-3)


def test_uses_graded_probabilities_not_just_the_winner():
    # Both say "a is preferred", but 0.95 is stronger evidence than 0.55 and
    # must produce a wider gap. Thresholding to wins would make these identical.
    narrow = fit_bradley_terry([("a", "b", 0.55)], regularization=0.01)
    wide = fit_bradley_terry([("a", "b", 0.95)], regularization=0.01)
    assert (wide["a"] - wide["b"]) > (narrow["a"] - narrow["b"])


def test_repeated_comparisons_of_the_same_pair_are_pooled():
    strengths = fit_bradley_terry([("a", "b", 0.9), ("a", "b", 0.9), ("a", "b", 0.9)])
    assert strengths["a"] > strengths["b"]
