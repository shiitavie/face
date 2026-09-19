"""Neoclassical canon indices computed from CFD measurements."""
import math

import pytest

from facecav.analysis.canons import (
    CORE_CANONS,
    EXTENDED_CANONS,
    canon_deviation,
    canon_indices,
)

# A face that satisfies every canon exactly.
IDEAL = {
    "FaceWidthBZ": 500.0, "EyeWidthAvg": 100.0,   # fifths: 5.0
    "NoseWidth": 100.0,                            # nasal fifth: 5.0
    "EyeDistance": 100.0,                          # extended canons
    # Thirds must share units: all three are already divided by face length.
    "UpperHeadLength": 0.33, "MidfaceLength": 0.33, "ChinLength": 0.33,
}


def test_an_ideal_face_has_zero_deviation():
    assert canon_deviation(IDEAL) == pytest.approx(0.0, abs=1e-9)


def test_every_canon_reports_its_index():
    assert set(canon_indices(IDEAL)) == set(CORE_CANONS)


def test_over_and_under_shooting_are_penalised_equally():
    # Log-ratio makes the measure symmetric: a nose twice the ideal width and
    # one half the ideal width are equally non-canonical.
    wide = dict(IDEAL, NoseWidth=200.0)
    narrow = dict(IDEAL, NoseWidth=50.0)
    assert canon_deviation(wide) == pytest.approx(canon_deviation(narrow))


def test_deviation_grows_with_departure_from_the_ideal():
    mild = canon_deviation(dict(IDEAL, NoseWidth=110.0))
    severe = canon_deviation(dict(IDEAL, NoseWidth=180.0))
    assert severe > mild > 0


def test_indices_are_scale_free():
    # Every CFD measurement is in image pixels, so a uniformly larger face must
    # score identically.
    scaled = {k: v * 2.5 for k, v in IDEAL.items()}
    assert canon_deviation(scaled) == pytest.approx(canon_deviation(IDEAL), abs=1e-9)


def test_facial_thirds_uses_the_widest_disparity():
    thirds = dict(IDEAL, UpperHeadLength=0.495)  # 0.495 / 0.33 / 0.33
    assert canon_indices(thirds)["facial_thirds"] == pytest.approx(1.5)


def test_missing_measurements_yield_nan_rather_than_a_wrong_number():
    partial = {k: v for k, v in IDEAL.items() if k != "NoseWidth"}
    indices = canon_indices(partial)
    assert math.isnan(indices["nasal_fifth"])
    assert math.isnan(canon_deviation(partial))


def test_nonpositive_measurements_yield_nan():
    assert math.isnan(canon_indices(dict(IDEAL, EyeWidthAvg=0.0))["facial_fifths"])


def test_extended_canons_need_eye_distance():
    # CFD records EyeDistance only for CFD-MR and CFD-INDIA; it is empty for all
    # 597 CFD main faces, so these canons cannot support the race comparison.
    without = {k: v for k, v in IDEAL.items() if k != "EyeDistance"}
    assert math.isnan(canon_deviation(without, canons=EXTENDED_CANONS))
    assert not math.isnan(canon_deviation(without))  # core still computes
