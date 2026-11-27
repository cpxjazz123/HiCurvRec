"""Each valid policy mutation must fail its named consumer-visible assertion."""
import pytest
import model.calibration as calibration
from test_budget_known_answers import assert_known_budget

BUDGETS = {'encoder': (4., 2.), 'Query1': (1., 4.), 'codebook0': (3., 3.)}


def test_real_policy_passes_the_designated_assertion():
    assert_known_budget(calibration.norm_budget_weight(BUDGETS))


@pytest.mark.parametrize('mutation', ['maximum', 'mean', 'omit_query', 'square_denominator', 'flip_sign'])
def test_policy_mutation_is_caught_by_exact_optimal_budget_assertion(monkeypatch, mutation):
    original = calibration.norm_budget_weight

    def mutated(budgets):
        if mutation == 'maximum':
            return max(a / b for a, b in budgets.values())
        if mutation == 'mean':
            return sum(a / b for a, b in budgets.values()) / len(budgets)
        if mutation == 'omit_query':
            return original({name: value for name, value in budgets.items() if name != 'Query1'})
        if mutation == 'square_denominator':
            return original({name: (a, b * b) for name, (a, b) in budgets.items()})
        return -original(budgets)

    monkeypatch.setattr(calibration, 'norm_budget_weight', mutated)
    # No import error, crash, timeout or arbitrary exception counts as caught.
    with pytest.raises(AssertionError, match='maximal module-safe coefficient must be 0.25'):
        assert_known_budget(calibration.norm_budget_weight(BUDGETS))
