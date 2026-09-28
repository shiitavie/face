"""Parsing an age estimate from free-text model output.

Age is an open numeric estimate, not a point on a fixed scale, so it needs
different parsing from the Likert path: ranges are common and must become a
midpoint rather than silently taking the lower bound.
"""
import math

import pytest

from facecav.models.scoring import parse_age


def test_parses_a_bare_number():
    assert parse_age("34") == 34.0


def test_parses_a_number_with_units():
    assert parse_age("34 years old") == 34.0


def test_parses_a_hedged_estimate():
    assert parse_age("approximately 28") == 28.0


def test_a_range_becomes_its_midpoint():
    # Taking the lower bound would bias every ranged answer downward.
    assert parse_age("25-30") == 27.5
    assert parse_age("between 40 and 50") == 45.0


def test_decimal_ages_are_kept():
    assert parse_age("28.5") == 28.5


def test_rejects_implausible_ages():
    assert math.isnan(parse_age("150"))
    assert math.isnan(parse_age("0"))


def test_refusals_return_nan():
    assert math.isnan(parse_age("I can't determine this person's age."))
    assert math.isnan(parse_age(""))


def test_ignores_a_leading_non_age_number():
    # "1 to 100" style scaffolding must not be read as the answer.
    assert parse_age("On a scale of 1 to 100, I would say 35") == 35.0


def test_a_range_spanning_the_whole_plausible_band_is_rejected():
    # "somewhere between 1 and 100" is a refusal wearing a number.
    assert math.isnan(parse_age("somewhere between 1 and 100"))
