"""Closed-form production budgets agree with independent HiGHS linear programs."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from model.calibration import norm_budget_weight

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'tools/research/stage2/contract.py').is_file())
SPEC = importlib.util.spec_from_file_location('issue256_r01_reference',
    ROOT / 'results/research/issue256/rounds/r01/reference.py')
REFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFERENCE)


def test_lp_hand_answer_is_not_a_production_echo():
    assert REFERENCE.reference_weight({'a': (3., 4.), 'b': (4., 3.)}) == pytest.approx(.75)
    assert REFERENCE.reference_global_norm([[3., 2.], [-3., 2.]]) == 2.


def test_independent_linear_program_over_varied_module_scales():
    rng = np.random.default_rng(42)
    for _ in range(24):
        original = np.exp(rng.uniform(-10, 1, 17))
        curvature = np.exp(rng.uniform(-2, 6, 17))
        budgets = {str(i): (float(a), float(b)) for i, (a, b) in enumerate(zip(original, curvature))}
        expected = REFERENCE.reference_weight(budgets)
        actual = norm_budget_weight(budgets)
        assert actual == pytest.approx(expected, rel=2e-6, abs=1e-12)
        assert max(actual * b / a for a, b in budgets.values()) <= 1 + 2e-6
