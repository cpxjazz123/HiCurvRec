"""Hand-answer prefix diagnosis, explicitly excluding the collision suffix."""
import numpy as np


def test_best_prefix_distinguishes_all_three_levels_without_suffix():
    from tools.issue256.r03.validation_diagnosis import best_prefix

    sids = np.array([[1, 2, 3, 768], [1, 2, 3, 769], [1, 2, 4, 768],
                     [1, 4, 9, 768], [7, 8, 9, 768]], dtype=np.int64)
    targets = np.array([0, 0, 0, 0, 0])
    selected = np.array([[1], [2], [3], [4], [0]])
    # Another item in the same triple is still L3, not a four-code match.
    assert best_prefix(sids, targets, selected).tolist() == [3, 2, 1, 0, 3]
