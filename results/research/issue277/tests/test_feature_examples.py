"""Known-answer checks for the six features (hand-computed values frozen in FEATURE_CONTRACT.md)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path('.').resolve()))
from tools.issue252 import layer_metrics as lm  # noqa: E402


def test_catalogue_hand_values():
    sid = [[1, 1, 1, 0], [1, 1, 2, 0], [1, 2, 1, 0], [1, 2, 2, 0], [2, 2, 2, 0], [2, 2, 3, 0]]
    cat = lm.catalogue_metrics(sid)
    assert cat['distinct_l2_prefixes'] == 3
    assert cat['l3_collision'] == 0.0
    assert abs(cat['prefix_entropy_bits'][1] - 1.584962500721156) < 1e-12
    assert abs(cat['prefix_entropy_bits'][2] - 2.584962500721156) < 1e-12
    assert abs(cat['l3_refinement_gain_bits'] - 1.0) < 1e-12


def test_validation_hand_values():
    sid = [[1, 1, 1, 0], [1, 1, 2, 0], [1, 2, 1, 0], [1, 2, 2, 0], [2, 2, 2, 0], [2, 2, 3, 0]]
    val = lm.validation_metrics(sid, [{'history': [0], 'seen_history': [], 'target': 2}])
    assert val['effective_rows'] == 1
    assert abs(val['l1_positive_hit_rate'] - 1.0) < 1e-12
    assert abs(val['l1_negative_hit_probability'] - 0.5) < 1e-12
    assert abs(val['l1_hard_er'] - 2.0) < 1e-12
    assert abs(val['positive_hit_rates_by_level'][1] - 0.0) < 1e-12
    assert abs(val['negative_hit_probabilities_by_level'][1] - 0.25) < 1e-12


def test_collision_is_duplicate_fraction_not_pairwise():
    sid = [[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [2, 2, 2, 0]]
    cat = lm.catalogue_metrics(sid)
    assert abs(cat['l3_collision'] - 0.6) < 1e-12          # 1 - 2/5
    pairwise = 1 - (4 * 3 + 0) / (5 * 4)
    assert abs(pairwise - 0.4) < 1e-12 and abs(cat['l3_collision'] - pairwise) > 1e-6


def test_er_is_pos_over_neg():
    sid = [[1, 1, 1, 0], [1, 1, 2, 0], [1, 2, 1, 0], [1, 2, 2, 0], [2, 2, 2, 0], [2, 2, 3, 0]]
    val = lm.validation_metrics(sid, [{'history': [0], 'seen_history': [], 'target': 2}])
    assert abs(val['l1_hard_er'] - val['l1_positive_hit_rate'] / val['l1_negative_hit_probability']) < 1e-12


if __name__ == '__main__':
    test_catalogue_hand_values()
    test_validation_hand_values()
    test_collision_is_duplicate_fraction_not_pairwise()
    test_er_is_pos_over_neg()
    print('all feature checks passed')
