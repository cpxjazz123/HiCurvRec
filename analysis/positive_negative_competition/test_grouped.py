import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "crowding_diagnostic"))
import competition_grouped as G


class GroupedCompetitionTests(unittest.TestCase):
    def test_three_stratifications_partition_rows_once(self):
        popularity = np.array([1, 3, 2, 4, 5, 8, 7, 6, 9, 10], dtype=float)
        successors = np.array([1, 2, 1, 4, 1, 2, 1, 3, 2, 1])
        hardness = np.array([.1, .8, .2, .9, .3, .7, .4, .6, .5, .0])
        groups = G.make_strata(popularity, successors, hardness, minimum_size=1)
        self.assertEqual(set(groups), {"popularity_decile", "successor_count", "negative_hardness_tercile"})
        for strata in groups.values():
            membership = np.sum(np.stack(list(strata.values())), axis=0)
            np.testing.assert_array_equal(membership, np.ones(10, dtype=int))
            self.assertEqual(sum(int(mask.sum()) for mask in strata.values()), 10)

    def test_small_stratum_is_undefined(self):
        x = np.arange(8, dtype=float)[:, None]
        pairs = {"anchors": np.array([0, 1]), "positives": np.array([2, 3]),
                 "negatives": np.array([[4], [5]])}
        got = G.group_metrics(x, pairs, np.array([6, 7]), np.array([True, False]), minimum_size=2)
        self.assertEqual(got["size"], 1)
        self.assertEqual(got["metrics"], "undefined")

    def test_hard_competition_is_margin_below_point_one(self):
        margins = np.array([[-.2, .05], [.1, .2], [.09, -.01]])
        got = G.hard_competition_rate(margins)
        self.assertEqual(got, {"value": 4 / 6, "n": 6, "definition": "fraction of margins m < 0.1"})

    def test_group_metrics_preserve_global_scale_invariance(self):
        x = np.array([[0.0], [1.0], [1.1], [1.3], [2.0], [3.0], [4.0], [5.0]])
        pairs = {"anchors": np.array([0, 1]), "positives": np.array([1, 2]),
                 "negatives": np.array([[2, 3], [3, 4]])}
        selected = np.array([True, True])
        ref = np.array([5, 6, 7])
        first = G.group_metrics(x, pairs, ref, selected, minimum_size=1)
        second = G.group_metrics(x * 17.0, pairs, ref, selected, minimum_size=1)
        self.assertEqual(first["metrics"]["pncr_by_epsilon"], second["metrics"]["pncr_by_epsilon"])
        self.assertEqual(first["metrics"]["error_rate_margin"], second["metrics"]["error_rate_margin"])
        self.assertEqual(first["metrics"]["behavioral_auc"], second["metrics"]["behavioral_auc"])
        self.assertEqual(first["metrics"]["hard_competition_rate"], second["metrics"]["hard_competition_rate"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
