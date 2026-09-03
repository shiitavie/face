"""Ordinal task score for VCR (spec section 5.3).

VCR defines its task score as the length-normalized log-probability of a class
token, which restricts it to classification. Attractiveness is ordinal, so we
take the expectation over the rating scale instead::

    S_i = sum_r  r * P~(r | I_i)
    P~(r) = exp(z_r) / sum_r' exp(z_r')

with the softmax renormalized over *only* the rating tokens. Renormalizing is
what keeps the score well-defined when the model puts mass elsewhere (a refusal
or a hedge); that displaced mass is reported separately by ``refusal_mass``.

Both functions are differentiable with respect to ``logits``, which VCR step 3
requires.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def _rating_logits(logits: torch.Tensor, rating_token_ids: Sequence[int]) -> torch.Tensor:
    return logits.index_select(-1, torch.as_tensor(rating_token_ids, device=logits.device))


def expected_rating(
    logits: torch.Tensor,
    rating_token_ids: Sequence[int],
    scale_min: int = 1,
) -> torch.Tensor:
    """Expected rating under the renormalized distribution over the scale.

    ``logits`` is ``(..., vocab)`` at the first completion position; the result
    is ``(...)``. ``rating_token_ids`` must be ordered from the lowest rating to
    the highest.
    """
    probabilities = torch.softmax(_rating_logits(logits, rating_token_ids), dim=-1)
    scale = torch.arange(
        scale_min,
        scale_min + len(rating_token_ids),
        device=logits.device,
        dtype=probabilities.dtype,
    )
    return (probabilities * scale).sum(dim=-1)


def refusal_mass(
    logits: torch.Tensor,
    rating_token_ids: Sequence[int],
) -> torch.Tensor:
    """Probability mass falling outside the rating tokens.

    Spec section 5.6: high mass here marks a refusal or hedge. Differential
    refusal across demographic subgroups is a reportable finding, so this is
    recorded per image rather than thresholded away at inference time.
    """
    full = torch.softmax(logits, dim=-1)
    return 1.0 - full.index_select(
        -1, torch.as_tensor(rating_token_ids, device=logits.device)
    ).sum(dim=-1)


SCALE_MIN, SCALE_MAX = 1, 7


class RatingTokenError(RuntimeError):
    """A rating digit does not correspond to a single token for this model.

    Spec section 5.6: the expected-rating score is only well-defined when every
    point on the scale occupies one token at the first completion position. A
    model that fails this is dropped rather than silently mis-scored -- this is
    the reason the scale is 1-7 and not 1-10, since "10" is two tokens under
    every BPE tokenizer we target.
    """


def resolve_rating_tokens(tokenizer) -> list[int]:
    """Token ids for ratings 1..7, ordered lowest to highest.

    Prefers the leading-space variant (`" 1"`): the prompt ends "The rating is",
    so under BPE the following token carries the space. Falls back to the bare
    digit for tokenizers that do not encode space that way.
    """
    ids = []
    for rating in range(SCALE_MIN, SCALE_MAX + 1):
        for candidate in (f" {rating}", f"{rating}"):
            encoded = tokenizer.encode(candidate, add_special_tokens=False)
            if len(encoded) == 1:
                ids.append(encoded[0])
                break
        else:
            raise RatingTokenError(
                f"rating {rating} is not a single token for this tokenizer; "
                "drop this model per spec section 5.6"
            )
    return ids
