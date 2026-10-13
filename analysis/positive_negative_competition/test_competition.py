import sys
sys.dont_write_bytecode = True
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "crowding_diagnostic"))
import unittest

import numpy as np

import competition as C
from crowding_diagnostic import distances



class CompetitionTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[0.0], [1.0], *[[1.1 + i * 0.01] for i in range(10)],
                           [0.01], [0.02], [0.03]])
        self.pairs = {"anchors": np.array([0]), "positives": np.array([1]),
                      "negatives": np.arange(2, 12, dtype=np.int64)[None, :]}
        self.reference = np.array([12, 13, 14])

    def test_exact_pncr_and_error_count(self):
        result = C.measure_level(self.x, self.pairs, self.reference)
        self.assertEqual(result["error_count"], 0)
        self.assertEqual(result["triple_count"], 10)
        self.assertEqual(result["pncr"]["value"], 0.0)
        changed = self.x.copy()
        changed[2] = 0.5
        result = C.measure_level(changed, self.pairs, self.reference)
        self.assertEqual(result["triple_count"], 10)
        self.assertEqual(result["error_rate_margin"]["value"], 0.1)

    def test_scale_invariance(self):
        a = C.measure_level(self.x, self.pairs, self.reference)
        b = C.measure_level(self.x * 17.0, self.pairs, self.reference)
        np.testing.assert_allclose(a["margins"], b["margins"], rtol=1e-12, atol=1e-12)
        self.assertEqual(a["pncr"], b["pncr"])

    def test_intrusion_and_coverage(self):
        x = np.array([[0.0], [1.0], [2.0], [0.5], [4.0], [3.0]])
        pairs = {"anchors": np.array([0]), "positives": np.array([[1, 2]]),
                 "negatives": np.array([[3, 4]])}
        got = C.neighborhood_metrics(x, pairs, k_values=(1, 2, 3), successor_lists=[[1, 2]])
        self.assertEqual(got["hard_negative_intrusion"]["value"], 1.0)
        self.assertEqual(got["positive_coverage_at_K"]["1"]["value"], 0.0)
        self.assertEqual(got["positive_coverage_at_K"]["3"]["value"], 1.0)
        self.assertEqual(got["matched_negative_coverage_at_K"]["2"]["value"], 0.5)

    def test_local_scale_is_per_anchor(self):

        x = np.array([[0.0], [1.0], [1.0125], [10.0], [11.0], [12.0],
                      [-0.05], [0.05], [0.1], [-0.1], [0.2], [0.3]])
        pairs = {"anchors": np.array([0, 3]), "positives": np.array([1, 4]),
                 "negatives": np.array([[2], [5]])}
        got = C.measure_level(x, pairs, np.arange(6, 12))
        scales = got["local_scales"]
        self.assertGreater(scales[1] / scales[0], 30.0)
        self.assertLess(max(got["margins"][:, 0]) / min(got["margins"][:, 0]), 1.3)

    def test_multiplicity_partitions_triples(self):
        multiplicity = np.array([1, 2, 1])
        groups = C.multiplicity_groups(multiplicity)
        self.assertEqual(sum(int(v.sum()) for v in groups.values()), 3)
        self.assertEqual(sum(int(v.sum()) for v in groups.values()) * 10, 30)

    def test_auc_matches_canonical_convention(self):
        pos = np.array([1.0, 2.0])
        neg = np.array([[2.0, 3.0], [1.0, 4.0]])
        self.assertEqual(C.behavioral_auc(pos, neg), distances.auc_neg_farther_than_pos(pos, neg))

    def test_counts_and_empty_undefined(self):
        empty = {"anchors": np.array([], dtype=int), "positives": np.array([], dtype=int),
                 "negatives": np.empty((0, 10), dtype=int)}
        result = C.measure_level(np.zeros((3, 1)), empty, np.arange(3))
        self.assertEqual(result["pncr"]["value"], "undefined")
        self.assertEqual(result["pncr"]["n"], 0)
        for stat in C.flatten_statistics(result):
            self.assertIn("n", stat)


if __name__ == "__main__":
    unittest.main(verbosity=2)
