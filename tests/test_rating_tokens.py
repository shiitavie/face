"""Resolving the seven rating tokens for a given tokenizer (spec section 5.6)."""
import pytest

from facecav.models.scoring import RatingTokenError, resolve_rating_tokens


class FakeTokenizer:
    """Minimal stand-in for the HF tokenizer surface we depend on."""

    def __init__(self, vocab):
        self.vocab = vocab

    def encode(self, text, add_special_tokens=False):
        if text in self.vocab:
            return [self.vocab[text]]
        # Absent from the single-token vocab means it takes more than one
        # token -- byte fallback, or a piece split like "10" -> "1" + "0".
        return [0, 0]


def test_prefers_the_leading_space_variant():
    # The prompt ends "The rating is", so the next token carries a leading
    # space; BPE gives " 1" its own id distinct from "1".
    vocab = {f" {d}": 100 + d for d in range(1, 8)} | {f"{d}": 200 + d for d in range(1, 8)}
    assert resolve_rating_tokens(FakeTokenizer(vocab)) == [101, 102, 103, 104, 105, 106, 107]


def test_falls_back_to_the_bare_digit_when_no_space_variant_exists():
    vocab = {f"{d}": 200 + d for d in range(1, 8)} | {" ": 5}
    assert resolve_rating_tokens(FakeTokenizer(vocab)) == [201, 202, 203, 204, 205, 206, 207]


def test_raises_when_a_rating_digit_is_not_a_single_token():
    # Spec 5.6: such a model is dropped rather than silently mis-scored.
    vocab = {f" {d}": 100 + d for d in range(1, 7)} | {" ": 5, "7": 77}
    del vocab[" 6"]
    with pytest.raises(RatingTokenError, match="6"):
        resolve_rating_tokens(FakeTokenizer(vocab))


def test_returns_ids_ordered_from_lowest_rating_to_highest():
    vocab = {f" {d}": 900 - d for d in range(1, 8)}
    ids = resolve_rating_tokens(FakeTokenizer(vocab))
    assert ids == [899, 898, 897, 896, 895, 894, 893]
