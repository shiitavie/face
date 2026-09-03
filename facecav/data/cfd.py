"""Chicago Face Database loading.

Filename grammar (CFD 3.0)::

    CFD-{race}{gender}-{n1}-{n2}[-{replicate}]-{expression}.jpg

The norming spreadsheet keys subjects differently per subset: CFD main and
CFD-MR use ``AF-200``, while the India extension uses ``IF601-519`` -- no dash
after the race/gender pair. ``parse_image_filename`` emits the norming key so
images and norming rows join directly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

EXPRESSIONS = frozenset({"N", "HO", "HC", "A", "F"})

_FILENAME = re.compile(
    r"^CFD-([A-Z])([A-Z])-(\d+)-(\d+)(?:-(\d+))?-([A-Z]{1,2})\.jpg$"
)


@dataclass(frozen=True)
class CFDImage:
    """One CFD image file, keyed to its norming-sheet row."""

    model_id: str
    race_code: str
    gender_code: str
    expression: str
    replicate: int | None = None


def parse_image_filename(name: str) -> CFDImage:
    m = _FILENAME.match(name)
    if m is None:
        raise ValueError(f"not a CFD image filename: {name!r}")
    race, gender, n1, n2, replicate, expression = m.groups()

    # India extension omits the dash after the race/gender pair in norming keys.
    if race == "I":
        model_id = f"{race}{gender}{n1}-{n2}"
    else:
        model_id = f"{race}{gender}-{n1}"
    if replicate is not None:
        model_id = f"{model_id}-{replicate}"

    return CFDImage(
        model_id=model_id,
        race_code=race,
        gender_code=gender,
        expression=expression,
        replicate=int(replicate) if replicate is not None else None,
    )


# --- manifest construction ---

import warnings
from pathlib import Path

import pandas as pd

#: Image subdirectory -> norming sheets, primary first. The India extension is
#: normed twice on the same question, once by US raters and once by Indian
#: raters, giving a cross-cultural human comparator on identical faces.
NORMING_SHEETS = {
    "CFD": [("CFD U.S. Norming Data", "us")],
    "CFD-MR": [("CFD-MR U.S. Norming Data", "us")],
    "CFD-INDIA": [
        ("CFD-I U.S. Norming Data", "us"),
        ("CFD-I INDIA Norming Data", "india"),
    ],
}

NORMING_WORKBOOK = "CFD 3.0 Norming Data and Codebook.xlsx"

#: Header rows in every norming sheet (0-indexed).
_ROW_VARID, _ROW_LABEL, _ROW_DATA = 6, 7, 9


def attractiveness_column(var_id: str, rater_pool: str) -> str:
    """Column name for a sheet's attractiveness variable.

    R013 asks for a rating *relative to others of the same race and gender*;
    R013B asks for an absolute first impression. They are different questions
    and must never share a column: R013 has the between-group effect removed by
    construction, so using it as a between-group human baseline is invalid.
    The rater pool is part of the name for the same reason -- US and Indian
    raters are not interchangeable.
    """
    if var_id == "R013":
        return "attractive_rel"
    if var_id == "R013B":
        return f"attractive_abs_{rater_pool}"
    raise ValueError(f"unrecognised attractiveness variable: {var_id!r}")


ATTRACTIVENESS_COLUMNS = ("attractive_rel", "attractive_abs_us", "attractive_abs_india")

_NON_NUMERIC = {
    "image_path", "subset", "model_id", "race_code", "gender_code",
    "expression", "attractive_variable", "join_status",
    "EthnicitySelf", "GenderSelf",
}


def _read_norming_sheet(
    workbook: Path, sheet: str, rater_pool: str
) -> tuple[pd.DataFrame, str]:
    """Return (tidy norming frame, the R013 variant this sheet uses)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = pd.read_excel(workbook, sheet_name=sheet, header=None)

    var_ids = [str(x) for x in raw.iloc[_ROW_VARID]]
    labels = [str(x) for x in raw.iloc[_ROW_LABEL]]
    body = raw.iloc[_ROW_DATA:].reset_index(drop=True)

    # Columns whose VarLabel is blank are the paired _sd columns; drop them.
    keep = {i: labels[i] for i in range(1, len(labels)) if labels[i] != "nan"}
    out = pd.DataFrame({name: body.iloc[:, i].values for i, name in keep.items()})
    out.insert(0, "model_id", body.iloc[:, 0].astype(str).values)
    out = out[out.model_id != "nan"].reset_index(drop=True)

    var_id = next(v for v in var_ids if v.startswith("R013"))
    return out.rename(columns={"Attractive": attractiveness_column(var_id, rater_pool)}), var_id


def build_manifest(root: Path, expression: str = "N") -> pd.DataFrame:
    """Join CFD images to their norming rows.

    One row per image file. Images with no norming row are retained and flagged
    in ``join_status`` rather than dropped, so the CFD distribution\'s own
    inconsistencies stay visible downstream.
    """
    root = Path(root)
    workbook = root / NORMING_WORKBOOK

    rows = []
    for subset in NORMING_SHEETS:
        for path in sorted((root / "Images" / subset).rglob(f"*-{expression}.jpg")):
            image = parse_image_filename(path.name)
            rows.append(
                {
                    "image_path": str(path),
                    "subset": subset,
                    "model_id": image.model_id,
                    "race_code": image.race_code,
                    "gender_code": image.gender_code,
                    "expression": image.expression,
                    "replicate": image.replicate,
                }
            )
    images = pd.DataFrame(rows)

    merged = []
    for subset, sheets in NORMING_SHEETS.items():
        (primary_sheet, primary_pool), *secondary = sheets

        norming, var_id = _read_norming_sheet(workbook, primary_sheet, primary_pool)
        part = images[images.subset == subset].merge(
            norming, on="model_id", how="left", indicator=True
        )
        part["attractive_variable"] = part["_merge"].map({"both": var_id})
        part["join_status"] = part["_merge"].map(
            {"both": "matched", "left_only": "no_norming_row"}
        )
        part = part.drop(columns="_merge")

        # Secondary sheets contribute only their attractiveness rating; the
        # remaining columns duplicate the primary sheet.
        for sheet, rater_pool in secondary:
            extra, extra_var = _read_norming_sheet(workbook, sheet, rater_pool)
            column = attractiveness_column(extra_var, rater_pool)
            part = part.merge(extra[["model_id", column]], on="model_id", how="left")

        merged.append(part)

    manifest = pd.concat(merged, ignore_index=True)

    for column in ATTRACTIVENESS_COLUMNS:
        if column not in manifest.columns:
            manifest[column] = pd.NA

    for column in (c for c in manifest.columns if c not in _NON_NUMERIC):
        manifest[column] = pd.to_numeric(manifest[column], errors="coerce")

    return manifest
