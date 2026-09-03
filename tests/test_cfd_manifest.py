"""Stage 0/1a: CFD manifest construction."""
import pytest
from facecav.data.cfd import parse_image_filename


def test_parses_cfd_main_filename():
    r = parse_image_filename("CFD-AF-200-228-N.jpg")
    assert r.model_id == "AF-200"
    assert r.race_code == "A"
    assert r.gender_code == "F"
    assert r.expression == "N"


def test_parses_india_filename_whose_model_id_omits_the_dash():
    # CFD-I norming rows are keyed "IF601-519", not "IF-601-519"
    r = parse_image_filename("CFD-IF-601-519-N.jpg")
    assert r.model_id == "IF601-519"
    assert r.race_code == "I"
    assert r.gender_code == "F"


def test_parses_india_filename_with_replicate_suffix():
    # Four CFD-I subjects have two neutral images; norming keys only the -1 row
    r = parse_image_filename("CFD-IF-644-306-1-N.jpg")
    assert r.model_id == "IF644-306-1"
    assert r.replicate == 1


def test_non_neutral_expression_is_reported_not_dropped():
    r = parse_image_filename("CFD-AF-201-060-HO.jpg")
    assert r.expression == "HO"


# --- manifest construction (uses the real CFD tree on disk) ---

from pathlib import Path
from facecav.data.cfd import build_manifest

CFD_ROOT = Path("dataset/CFD Version 3.0")


@pytest.fixture(scope="module")
def manifest():
    return build_manifest(CFD_ROOT)


def test_one_row_per_neutral_image(manifest):
    assert len(manifest) == 831
    assert set(manifest.expression) == {"N"}


def test_relative_and_absolute_attractiveness_stay_in_separate_columns(manifest):
    # CFD main/MR ask "relative to other people of the same race and gender"
    # (R013); CFD-I asks an absolute first impression (R013B). Merging them
    # into one column would silently invalidate any between-group comparison.
    main = manifest[manifest.subset == "CFD"]
    india = manifest[manifest.subset == "CFD-INDIA"]

    assert main.attractive_rel.notna().all()
    assert main.attractive_abs_us.isna().all()

    assert india.attractive_abs_us.notna().any()
    assert india.attractive_rel.isna().all()

    assert set(manifest.attractive_variable.dropna()) == {"R013", "R013B"}


def test_images_without_a_norming_row_are_flagged_not_dropped(manifest):
    unmatched = manifest[manifest.join_status == "no_norming_row"]
    # four CFD-I "-2" replicates, plus the IM719-221/IM719-220 discrepancy
    assert len(unmatched) == 5
    assert all(mid.endswith("-2") or mid == "IM719-221" for mid in unmatched.model_id)


def test_physical_measurements_are_carried_through(manifest):
    for col in ("NoseWidth", "LipThickness", "FaceWidthCheeks"):
        assert col in manifest.columns
    assert manifest.loc[manifest.subset == "CFD", "NoseWidth"].notna().all()


def test_cfd_india_carries_both_us_and_indian_norming(manifest):
    # CFD-I is normed twice on the same absolute question (R013B): once by US
    # raters, once by Indian raters. Comparing model ratings against each gives
    # a cross-cultural human comparator that costs nothing to obtain.
    india = manifest[manifest.subset == "CFD-INDIA"]
    assert india.attractive_abs_us.notna().any()
    assert india.attractive_abs_india.notna().any()

    both = india.dropna(subset=["attractive_abs_us", "attractive_abs_india"])
    assert len(both) > 100
    # Same faces, different rater pools -- the columns must not be identical.
    assert not (both.attractive_abs_us == both.attractive_abs_india).all()


def test_indian_norming_is_absent_outside_cfd_india(manifest):
    other = manifest[manifest.subset != "CFD-INDIA"]
    assert other.attractive_abs_india.isna().all()
