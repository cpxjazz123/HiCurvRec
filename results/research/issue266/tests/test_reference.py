"""Hand-computed checks for Issue #266's independent reference."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import reference


def test_score_endpoints_and_no_clipping():
    low = reference.score(reference.LO_C, reference.LO_G)
    high = reference.score(reference.HI_C, reference.HI_G)
    assert low["S"] == pytest.approx(0.0)
    assert high["S"] == pytest.approx(1.0)
    outside = reference.score(0.0, 0.0)
    assert outside["S"] < 0.0
    assert outside["c_component"] == pytest.approx(0.8 * (-reference.LO_C) /
                                                   (reference.HI_C - reference.LO_C))
    # The value is retained, not clipped to zero.
    assert outside["S"] != 0.0


def test_frozen_historical_bracket_values():
    lower = reference.score(1.8262606859207153, 14.063595328730889)["S"]
    upper = reference.score(1.7692334651947021, 20.140511086361993)["S"]
    assert lower == pytest.approx(0.48640629657478884, abs=1e-15)
    assert upper == pytest.approx(0.5096899206435366, abs=1e-15)


def test_pearson_and_spearman_exact():
    x = [1.0, 2.0, 3.0, 4.0]
    assert reference.pearson(x, [2.0 * value for value in x]) == pytest.approx(1.0)
    assert reference.pearson(x, [-value for value in x]) == pytest.approx(-1.0)
    assert reference.spearman(x, x) == pytest.approx(1.0)


def test_kendall_tau_b_no_ties_hand_value():
    # Five concordant and one discordant pair among six: (5-1)/6.
    assert reference.kendall_tau_b([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(4.0 / 6.0)


def test_kendall_tau_b_ties_hand_value():
    # Pairs: (1,3) C, (2,3) C, (2,1) tied only on x; n0=3, ties_x=1.
    assert reference.kendall_tau_b([1, 1, 2], [1, 2, 3]) == pytest.approx(2.0 / math.sqrt(6.0))


def test_strict_adjacency_and_reversal():
    ascending = [dict(index=i, recall=float(i), S=float(i) / 10) for i in (1, 2, 3)]
    assert reference.adjacent_monotonic(ascending)["all_strict"]
    reversed_rows = [dict(index=1, recall=1.0, S=0.4),
                     dict(index=2, recall=2.0, S=0.2),
                     dict(index=3, recall=3.0, S=0.3)]
    result = reference.adjacent_monotonic(reversed_rows)
    assert not result["all_strict"]
    assert result["violations"][0]["left"] == 1 and result["violations"][0]["right"] == 2


def test_tied_score_is_not_strict():
    rows = [dict(index=1, recall=1.0, S=0.2),
            dict(index=2, recall=2.0, S=0.2),
            dict(index=3, recall=3.0, S=0.3)]
    result = reference.adjacent_monotonic(rows)
    assert not result["all_strict"]
    assert result["pairs"][0]["tie"] is True


def test_inversions_and_violating_arms():
    rows = [dict(index=1, recall=1.0, S=0.5),
            dict(index=2, recall=2.0, S=0.1),
            dict(index=3, recall=3.0, S=0.2)]
    result = reference.inversions(rows)
    assert result["count"] == 2
    assert result["violating_arms"] == [1, 2, 3]
    assert result["pairs"][0] == {"lower_recall": 1, "higher_recall": 2,
                                  "lower_S": 0.5, "higher_S": 0.1}


def test_tied_recall_is_separate_not_an_inversion():
    rows = [dict(index=1, recall=1.0, S=0.5),
            dict(index=2, recall=1.0, S=0.2)]
    result = reference.inversions(rows)
    assert result["count"] == 0
    assert result["tied_recall_pairs"] == [[1, 2]]


def test_tied_score_is_not_counted_as_an_inversion():
    rows = [dict(index=1, recall=1.0, S=0.5),
            dict(index=2, recall=2.0, S=0.5),
            dict(index=3, recall=3.0, S=0.2)]
    result = reference.inversions(rows)
    assert result["count"] == 2
    assert result["tied_score_pairs"] == [[1, 2]]
    assert result["tied_recall_pairs"] == []


def test_missing_and_constant_axes_are_none():
    assert reference.pearson([1.0], [2.0]) is None
    assert reference.pearson([1.0, 1.0], [2.0, 3.0]) is None
    assert reference.spearman([1.0, 1.0], [2.0, 3.0]) is None
    assert reference.kendall_tau_b([1.0, 1.0], [2.0, 3.0]) is None
