"""Issue #272 round 2: the local/global gradient-balance formula."""
import sys
from pathlib import Path
import pytest

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))
from model.global_constraint import balance_lambda


def test_known_answer():
    # lambda0 * fraction * global / local = 5e-6 * 0.1 * 0.2 / 1e-4 = 1e-3
    assert balance_lambda(5e-6, 1e-4, 0.2, 0.1) == pytest.approx(1e-3, rel=1e-12)


def test_zero_and_negative_rejected():
    for args in ((0.0, 1e-4, 0.2, 0.1), (5e-6, 0.0, 0.2, 0.1),
                 (5e-6, 1e-4, 0.0, 0.1), (5e-6, 1e-4, 0.2, 0.0),
                 (5e-6, -1e-4, 0.2, 0.1)):
        with pytest.raises(ValueError):
            balance_lambda(*args)
