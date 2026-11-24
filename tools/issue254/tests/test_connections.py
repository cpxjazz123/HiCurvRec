import unittest
import numpy as np
from tools.issue254.connections import sign_concordance


class ConnectionTests(unittest.TestCase):
    def test_observed_sign_labels_and_missing_mask(self):
        gates = np.array([[.8,.1,.1],[.1,.8,.1],[.1,.1,.8]])
        ratios = np.array([[.7,.0,.3],[.7,.0,.3],[np.nan,np.nan,np.nan]])
        self.assertEqual(sign_concordance(gates,ratios,np.array([True,True,False])).tolist(),
                         ['same','different','unavailable'])

    def test_concordance_rejects_malformed_valid_distribution(self):
        with self.assertRaisesRegex(ValueError,'distribution'):
            sign_concordance(np.array([[.2,.5,.3]]),np.array([[.3,.3,.3]]),np.array([True]))


if __name__ == '__main__':
    unittest.main()
