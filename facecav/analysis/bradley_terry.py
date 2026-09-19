"""Bradley-Terry latent scale from counterbalanced pairwise preferences.

The model gives a strong, near-constant "prefer the second image" bias with a
real graded preference underneath. Averaging the two presentations,

    p_ij = (P(i first, i chosen) + 1 - P(j first, j chosen)) / 2

cancels the position offset and leaves a preference probability in [0, 1].

Bradley-Terry then says::

    p_ij = sigmoid(theta_i - theta_j)

and the fit inverts that for the latent strengths ``theta``. Those strengths are
the usable attractiveness measurement -- the absolute 1-7 rating is not, since
the model ignores the stated direction of the scale.

The observations are *probabilities*, not wins. Thresholding them would discard
the graded information that makes the instrument work at all, so the objective
is the cross-entropy between the predicted and observed probabilities rather
than the usual win-count likelihood.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from scipy.optimize import minimize

#: Keeps the fit finite when a pair is observed at exactly 0 or 1.
_EPSILON = 1e-6


def fit_bradley_terry(
    comparisons: Iterable[Sequence],
    regularization: float = 1e-3,
) -> dict[str, float]:
    """Fit latent strengths from ``(item_i, item_j, p_i_preferred)`` triples.

    ``regularization`` is an L2 penalty on the strengths. It keeps items that
    only ever win (or only ever lose) from running off to infinity, which
    happens readily when comparisons are sparsely sampled.

    Returns strengths mean-centered, since only differences are identifiable.
    """
    comparisons = list(comparisons)
    if not comparisons:
        raise ValueError("no comparisons given")

    items = sorted({item for i, j, _ in comparisons for item in (i, j)})
    index = {item: position for position, item in enumerate(items)}

    left = np.array([index[i] for i, _, _ in comparisons])
    right = np.array([index[j] for _, j, _ in comparisons])
    observed = np.clip(
        np.array([float(p) for _, _, p in comparisons]), _EPSILON, 1 - _EPSILON
    )

    def objective(theta):
        difference = theta[left] - theta[right]
        # Cross-entropy between sigmoid(difference) and the observed probability,
        # written via logaddexp so large |difference| stays stable.
        loss = np.sum(
            (1 - observed) * difference + np.logaddexp(0.0, -difference)
        )
        predicted = 1.0 / (1.0 + np.exp(-difference))
        residual = predicted - observed
        gradient = np.zeros_like(theta)
        np.add.at(gradient, left, residual)
        np.add.at(gradient, right, -residual)

        loss += regularization * np.sum(theta**2)
        gradient += 2 * regularization * theta
        return loss, gradient

    result = minimize(
        objective, np.zeros(len(items)), jac=True, method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-10},
    )
    strengths = result.x - result.x.mean()
    return {item: float(strengths[index[item]]) for item in items}


def sample_comparison_pairs(
    faces: Sequence[str],
    comparisons_per_face: int,
    seed: int = 0,
) -> list[tuple[str, str]]:
    """Sample a connected comparison graph over ``faces``.

    Full pairwise is quadratic and infeasible (831 faces is 345,015 pairs), but
    Bradley-Terry does not need completeness -- it needs enough overlap to tie
    every face into one comparison graph. Strengths are only comparable within a
    connected component, so a split graph would silently produce two
    incomparable scales.

    Builds a random cycle through all faces to guarantee connectivity, then adds
    random pairs until every face reaches ``comparisons_per_face``.
    """
    faces = list(faces)
    if comparisons_per_face >= len(faces):
        raise ValueError(
            f"cannot draw {comparisons_per_face} distinct partners per face "
            f"from {len(faces)} faces"
        )

    rng = np.random.default_rng(seed)
    order = list(rng.permutation(faces))

    # A Hamiltonian cycle guarantees one connected component.
    chosen = {
        frozenset((order[i], order[(i + 1) % len(order)]))
        for i in range(len(order))
    }
    counts = {face: 2 for face in faces}

    deficient = [f for f in faces if counts[f] < comparisons_per_face]
    while deficient:
        face = deficient[rng.integers(len(deficient))]
        partners = [
            f for f in faces
            if f != face and frozenset((face, f)) not in chosen
        ]
        if not partners:
            counts[face] = comparisons_per_face  # exhausted; accept what we have
        else:
            partner = partners[rng.integers(len(partners))]
            chosen.add(frozenset((face, partner)))
            counts[face] += 1
            counts[partner] += 1
        deficient = [f for f in faces if counts[f] < comparisons_per_face]

    pairs = sorted(tuple(sorted(pair)) for pair in chosen)
    return [pairs[k] for k in rng.permutation(len(pairs))]
