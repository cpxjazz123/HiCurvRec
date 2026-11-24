"""Hand-calculated catalogue entropy and exact negative-population diagnostics."""
import unittest
import numpy as np


class LayerMetricTests(unittest.TestCase):
    def setUp(self):
        self.rows = np.array([[0, 0, 0, 0], [0, 0, 1, 0],
                              [0, 1, 0, 0], [1, 0, 0, 0]], dtype=np.int64)

    def test_catalogue_conditional_entropy_and_counter_is_not_a_code(self):
        from tools.issue252.layer_metrics import catalogue_metrics
        actual = catalogue_metrics(self.rows)
        self.assertEqual(actual['distinct_l2_prefixes'], 3)
        self.assertEqual(actual['distinct_l3_triples'], 4)
        self.assertEqual(actual['l3_collision'], 0.)
        self.assertAlmostEqual(actual['l3_refinement_gain_bits'], .5, places=12)
        self.rows[:, 3] = [100, 101, 102, 103]
        self.assertEqual(catalogue_metrics(self.rows), actual)

    def test_exact_legal_negative_population_hand_answer(self):
        from tools.issue252.layer_metrics import validation_metrics
        rows = [{'history': [0], 'seen_history': [0], 'target': 1}]
        actual = validation_metrics(self.rows, rows)
        self.assertEqual(actual['effective_rows'], 1)
        self.assertEqual(actual['l1_positive_hit_rate'], 1.)
        self.assertEqual(actual['l1_negative_hit_probability'], .5)
        self.assertEqual(actual['l1_hard_er'], 2.)
        self.assertEqual(actual['conditional_log_enrichment_bits']['status'], 'NA')
        self.assertEqual(actual['conditional_g3']['status'], 'MISSING')

    def test_empty_and_fully_excluded_rows_are_not_zero_imputed(self):
        from tools.issue252.layer_metrics import validation_metrics
        rows = [{'history': [], 'seen_history': [], 'target': 1},
                {'history': [0], 'seen_history': [0, 1, 2, 3], 'target': 1}]
        actual = validation_metrics(self.rows, rows)
        self.assertEqual(actual['effective_rows'], 0)
        self.assertEqual(actual['excluded_empty_history'], 1)
        self.assertEqual(actual['excluded_empty_negative_population'], 1)
        self.assertEqual(actual['l1_hard_er']['status'], 'NA')
        with self.assertRaisesRegex(ValueError, 'global'):
            validation_metrics(self.rows, [{'history': [4], 'target': 1, 'seen_history': []}])


if __name__ == '__main__':
    unittest.main()
