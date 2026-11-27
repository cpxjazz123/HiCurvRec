"""Hardcoded optima include a non-encoder bottleneck and inactive modules."""
import math
import pytest

from model.calibration import norm_budget_weight


def assert_known_budget(weight):
    assert weight == pytest.approx(0.25, abs=1e-12), 'maximal module-safe coefficient must be 0.25'


def test_query_not_encoder_can_be_the_limiting_module():
    assert_known_budget(norm_budget_weight({'encoder': (4., 2.), 'Query1': (1., 4.),
                                           'codebook0': (3., 3.)}))


def test_inactive_zero_curvature_group_does_not_constrain_weight():
    assert norm_budget_weight({'encoder': (3., 4.), 'decoder': (9., 0.),
                               'inactive': (0., 0.)}) == pytest.approx(0.75)


@pytest.mark.parametrize('budgets', [
    {}, {'zero': (0., 0.)}, {'zero-original': (0., 1.)},
    {'nan': (math.nan, 1.)}, {'infinity': (1., math.inf)}, {'negative': (-1., 1.)},
])
def test_degenerate_or_nonfinite_budget_is_rejected(budgets):
    with pytest.raises((ValueError, RuntimeError)):
        norm_budget_weight(budgets)
