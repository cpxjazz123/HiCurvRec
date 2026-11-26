"""Deterministic tests for #262 aggregate correlation/validation logic."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "results/research/issue262"))

from aggregate import corr, sensitivity  # noqa: E402


def test_corr_known_answers_and_constant():
    x = [1, 2, 3, 4]
    y = [2, 4, 6, 8]
    result = corr(x, y)
    assert result["n"] == 4
    assert abs(result["pearson"] - 1.0) < 1e-12
    assert abs(result["spearman"] - 1.0) < 1e-12
    const = corr([1, 1, 1], [1, 2, 3])
    assert const["pearson"] is None and const["spearman"] is None


def test_corr_drops_missing_not_zero():
    result = corr([1, None, 3, 4], [1, 2, 3, 4])
    assert result["n"] == 3
    assert result["pearson"] > 0.9


def test_sensitivity_leave_one_out_shape():
    rows = [{"index": i, "x": i - 99, "y": (i - 99) ** 2} for i in range(100, 108)]
    result = sensitivity(rows, "x", "y")
    assert result["all"]["n"] == 8
    assert result["exclude_100"]["n"] == 7
    assert result["exclude_107"]["n"] == 7
    assert result["loo"]["low"] <= result["all"]["pearson"] <= result["loo"]["high"]
