"""The ordinal task score (spec section 5.3) -- the method contribution.

S_i = sum_r r * P~(r | I_i), where P~ renormalizes the softmax over only the
seven rating tokens at the first completion position.
"""
import pytest
import torch

from facecav.models.scoring import expected_rating, refusal_mass

VOCAB = 50
RATING_IDS = [10, 11, 12, 13, 14, 15, 16]  # stand-ins for "1".."7"


def test_uniform_mass_over_rating_tokens_scores_the_scale_midpoint():
    logits = torch.zeros(VOCAB)
    assert expected_rating(logits, RATING_IDS).item() == pytest.approx(4.0)


def test_all_mass_on_the_top_token_scores_the_scale_maximum():
    logits = torch.full((VOCAB,), -1e9)
    logits[RATING_IDS[-1]] = 0.0
    assert expected_rating(logits, RATING_IDS).item() == pytest.approx(7.0)


def test_mass_outside_the_rating_tokens_does_not_shift_the_score():
    # Renormalization is what makes the score well-defined when the model
    # wants to say "I can't" -- refusal must move refusal_mass, not the rating.
    logits = torch.zeros(VOCAB)
    baseline = expected_rating(logits, RATING_IDS)

    logits[42] = 20.0  # an unrelated token dominates the raw softmax
    assert expected_rating(logits, RATING_IDS).item() == pytest.approx(baseline.item())


def test_refusal_mass_reports_probability_outside_the_rating_tokens():
    logits = torch.full((VOCAB,), -1e9)
    logits[RATING_IDS[0]] = 0.0
    logits[42] = 0.0  # half the mass lands off-scale
    assert refusal_mass(logits, RATING_IDS).item() == pytest.approx(0.5)


def test_score_is_differentiable_wrt_logits():
    # VCR step 3 needs grad of the task score; without this the whole
    # sensitivity analysis is impossible.
    logits = torch.zeros(VOCAB, requires_grad=True)
    expected_rating(logits, RATING_IDS).backward()
    assert logits.grad is not None
    assert torch.any(logits.grad != 0)


def test_batched_logits_give_one_score_per_row():
    logits = torch.zeros(4, VOCAB)
    assert expected_rating(logits, RATING_IDS).shape == (4,)
