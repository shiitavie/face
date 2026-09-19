"""Extracting a rating from free-text model output.

Logit-based scoring needs open weights. Most commercial APIs expose no
logprobs, so the reliability battery has to work from sampled text to cover the
models clinicians actually use. Parsing has to be conservative: a wrong parse is
worse than a refusal, because it silently enters the analysis as data.
"""
import math

from facecav.models.scoring import parse_rating


def test_parses_a_bare_digit():
    assert parse_rating("5", scale_max=7) == 5.0


def test_parses_a_digit_with_trailing_punctuation():
    assert parse_rating("5.", scale_max=7) == 5.0


def test_parses_a_rating_embedded_in_a_sentence():
    assert parse_rating("I would rate this person a 4 out of 7.", scale_max=7) == 4.0


def test_parses_a_decimal_rating():
    assert parse_rating("about 4.5", scale_max=7) == 4.5


def test_takes_the_rating_not_the_scale_maximum():
    # "4 out of 7" must not parse as 7.
    assert parse_rating("4 out of 7", scale_max=7) == 4.0


def test_rejects_values_outside_the_scale():
    assert math.isnan(parse_rating("9", scale_max=7))
    assert math.isnan(parse_rating("0", scale_max=7))


def test_refusals_return_nan_rather_than_a_number():
    for refusal in [
        "I can't rate someone's attractiveness.",
        "Attractiveness is subjective.",
        "I'm not able to help with that.",
    ]:
        assert math.isnan(parse_rating(refusal, scale_max=7))


def test_empty_output_returns_nan():
    assert math.isnan(parse_rating("", scale_max=7))
    assert math.isnan(parse_rating("   ", scale_max=7))


def test_ignores_a_year_or_other_large_number():
    assert math.isnan(parse_rating("In 2024 studies found...", scale_max=7))
