"""Neoclassical canon indices from CFD measurements.

The neoclassical canons were derived from Greek and Roman sculpture and codified
by Renaissance artists. They are known not to hold outside European faces
(Farkas and others), which is exactly why they are worth testing here: a facial
analysis tool that encodes them would describe non-European patients as
deviating from an aesthetic norm.

Each canon is a ratio with an ideal value. Deviation is the absolute log-ratio
against that ideal, so over- and under-shooting count equally -- a nose twice the
canonical width is no more canonical than one half of it, and a plain difference
would wrongly say otherwise.

All CFD measurements are in image pixels, so only ratios are meaningful; every
index here is scale-free by construction.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

Measurements = Mapping[str, float]


def _ratio(numerator: str, denominator: str) -> Callable[[Measurements], float]:
    def compute(measurements: Measurements) -> float:
        try:
            a, b = measurements[numerator], measurements[denominator]
        except KeyError:
            return math.nan
        if not (_positive(a) and _positive(b)):
            return math.nan
        return a / b

    return compute


#: The facial thirds must be compared in consistent units. CFD's ``Forehead``
#: (P015) is a pixel distance, while ``MidfaceLength`` (P058) and ``ChinLength``
#: (P059) are already divided by face length. Mixing them yields ratios in the
#: thousands. ``UpperHeadLength`` (P057) is Forehead/FaceLength, so these three
#: are the consistent triple. They approximate the classical
#: trichion-glabella-subnasale-menton thirds rather than reproducing them.
_THIRDS = ("UpperHeadLength", "MidfaceLength", "ChinLength")


def _thirds(measurements: Measurements) -> float:
    """Widest disparity among the facial thirds; the canon says all are equal."""
    try:
        values = [measurements[name] for name in _THIRDS]
    except KeyError:
        return math.nan
    if not all(_positive(v) for v in values):
        return math.nan
    return max(values) / min(values)


def _positive(value) -> bool:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and value > 0


#: Computable for every CFD face (n=826).
CORE_CANONS: dict[str, tuple[Callable[[Measurements], float], float]] = {
    # Face width spans five eye widths.
    "facial_fifths": (_ratio("FaceWidthBZ", "EyeWidthAvg"), 5.0),
    # Upper, middle and lower thirds are equal.
    "facial_thirds": (_thirds, 1.0),
    # Nasal width relative to face width; the canonical fifth.
    "nasal_fifth": (_ratio("FaceWidthBZ", "NoseWidth"), 5.0),
}

#: Require ``EyeDistance``, which CFD records ONLY for CFD-MR and CFD-INDIA --
#: it is empty for all 597 CFD main faces. These therefore cannot support the
#: race comparison, which lives in CFD main, and are reported separately on the
#: 229 faces that have them.
EXTENDED_CANONS: dict[str, tuple[Callable[[Measurements], float], float]] = {
    # Intercanthal distance equals one eye width.
    "intercanthal_equals_eye": (_ratio("EyeDistance", "EyeWidthAvg"), 1.0),
    # Alar base width equals the intercanthal distance.
    "nasal_equals_intercanthal": (_ratio("NoseWidth", "EyeDistance"), 1.0),
}

CANONS = CORE_CANONS


def canon_indices(measurements: Measurements, canons=None) -> dict[str, float]:
    """Each canon's observed ratio. NaN where a measurement is missing."""
    canons = CORE_CANONS if canons is None else canons
    return {name: index(measurements) for name, (index, _) in canons.items()}


def canon_deviation(measurements: Measurements, canons=None) -> float:
    """Total departure from the canons, as summed absolute log-ratios.

    NaN if any canon cannot be computed -- a partial sum would silently look
    more canonical than a complete one, which is the wrong direction to fail in.
    """
    canons = CORE_CANONS if canons is None else canons
    total = 0.0
    for name, (index, ideal) in canons.items():
        observed = index(measurements)
        if not _positive(observed):
            return math.nan
        total += abs(math.log(observed / ideal))
    return total
