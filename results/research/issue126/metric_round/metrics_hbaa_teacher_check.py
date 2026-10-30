"""The HBAA variants that use the teacher's alignment, after the radial fix.

``_alignment`` had its own inline copy of the radial distance, and that copy was
the second place where the factor two was missing. This module exposes the
variants that go through it, so the corrected numbers can be measured without
re-running the whole family.
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
    "hbaa_teacher_curvature": SOURCE.hbaa_teacher_curvature,
    "hbaa_permuted": SOURCE.hbaa_permuted,
}
FREE_CONSTANTS = {name: 0 for name in METRICS}
