"""The four HBAA numbers the radial-distance correction needs.

The full bench scores thirteen variants; the question here is narrower -- the
Poincare form after the radial distance was corrected, its flat and Euclidean
controls, and the permutation control -- so this module exposes only those four
and the bench runs them on the 107 confirmed pairs.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "hbaa_source", Path(__file__).resolve().parent / "metrics_hbaa.py")
SOURCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOURCE)

METRICS = {
    "hbaa": SOURCE.hbaa,
    "hbaa_euclid": SOURCE.hbaa_euclid,
    "hbaa_flat": SOURCE.hbaa_flat,
    "hbaa_permuted": SOURCE.hbaa_permuted,
}
FREE_CONSTANTS = {name: 0 for name in METRICS}
