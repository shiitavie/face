"""Sampling the comparison graph for Bradley-Terry."""
import pytest

from facecav.analysis.bradley_terry import sample_comparison_pairs


FACES = [f"f{i}" for i in range(40)]


def test_every_face_appears_at_least_k_times():
    pairs = sample_comparison_pairs(FACES, comparisons_per_face=6, seed=0)
    counts = {face: 0 for face in FACES}
    for i, j in pairs:
        counts[i] += 1
        counts[j] += 1
    assert min(counts.values()) >= 6


def test_no_face_is_compared_with_itself():
    pairs = sample_comparison_pairs(FACES, comparisons_per_face=6, seed=0)
    assert all(i != j for i, j in pairs)


def test_no_unordered_pair_repeats():
    pairs = sample_comparison_pairs(FACES, comparisons_per_face=6, seed=0)
    assert len({frozenset(p) for p in pairs}) == len(pairs)


def test_graph_is_connected():
    # Bradley-Terry strengths are only comparable within a connected component;
    # a split graph silently yields two incomparable scales.
    pairs = sample_comparison_pairs(FACES, comparisons_per_face=4, seed=0)
    parent = {face: face for face in FACES}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs:
        parent[find(i)] = find(j)
    assert len({find(face) for face in FACES}) == 1


def test_is_deterministic_for_a_given_seed():
    assert (sample_comparison_pairs(FACES, comparisons_per_face=5, seed=7)
            == sample_comparison_pairs(FACES, comparisons_per_face=5, seed=7))


def test_rejects_a_budget_the_population_cannot_support():
    with pytest.raises(ValueError):
        sample_comparison_pairs(["a", "b", "c"], comparisons_per_face=10, seed=0)
