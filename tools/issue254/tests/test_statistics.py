import unittest
import numpy as np
from tools.issue254.statistics import exact_mcnemar, paired_interval, cluster_interval, holm_adjust, adjusted_association


class StatisticsTests(unittest.TestCase):
    def test_exact_discordants(self):
        self.assertEqual(exact_mcnemar(0, 0), 1.0)
        self.assertEqual(exact_mcnemar(3, 0), .25)
        self.assertEqual(exact_mcnemar(1, 1), 1.0)

    def test_paired_resampling_keeps_degenerate_effect(self):
        np.testing.assert_array_equal(paired_interval(np.ones(7), repetitions=100), [1., 1.])
        np.testing.assert_array_equal(paired_interval(np.zeros(7), repetitions=100), [0., 0.])
        a = paired_interval(np.array([-1, 0, 0, 1, 1]), repetitions=100)
        b = paired_interval(np.array([-1, 0, 0, 1, 1]), repetitions=100)
        np.testing.assert_array_equal(a, b)

    def test_cluster_ratio_not_unweighted_item_mean(self):
        np.testing.assert_array_equal(cluster_interval(np.array([2.]), np.array([4.]), repetitions=20), [.5, .5])
        # Every cluster has effect .25, despite unequal cluster populations.
        np.testing.assert_array_equal(cluster_interval(np.array([1., 5.]), np.array([4., 20.]), repetitions=20), [.25, .25])

    def test_holm_preserves_original_order(self):
        np.testing.assert_allclose(holm_adjust([.04, .01, .03]), [.06, .03, .06])

    def test_cluster_adjusted_linear_association(self):
        x = np.tile(np.arange(4.), 4)
        y = 2 + 3 * x
        result = adjusted_association({'x': x}, y, np.repeat(np.arange(4), 4))
        self.assertAlmostEqual(result['coefficients'][0]['coefficient_per_sd'], 3 * x.std())
        self.assertEqual(result['population'], 16)
        self.assertEqual(result['clusters'], 4)


    def test_training_frequency_input_hash_not_asset_entry(self):
        import hashlib
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import pandas as pd
        from tools.issue254.statistics import training_frequencies, ROOT
        runtime = ROOT / 'results/research/issue254/runtime'
        runtime.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=runtime) as directory:
            root = Path(directory)
            path = root / 'results/stage0_build_parquet/train.parquet'
            path.parent.mkdir(parents=True)
            path.write_bytes(b'frozen training fixture')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = {'assets': [], 'inputs': {'train': digest}}
            with patch('pandas.read_parquet', return_value=pd.DataFrame({'target': [0, 0, 2]})):
                counts, observed = training_frequencies(root, manifest)
            self.assertEqual(counts.to_dict(), {0: 2, 2: 1})
            self.assertEqual(observed, digest)
            with self.assertRaisesRegex(ValueError, 'source changed'):
                training_frequencies(root, {'inputs': {'train': 'wrong'}})


if __name__ == '__main__':
    unittest.main()
