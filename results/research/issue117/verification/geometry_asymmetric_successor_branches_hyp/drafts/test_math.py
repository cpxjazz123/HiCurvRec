import importlib.util
import pathlib
import sys
import unittest

import torch

# Import the frozen candidate directly by source-tree path, not via its package.
_SOURCE_ROOT = pathlib.Path("/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_c117_asymmetric_successor_branches")
_MODULE_PATH = _SOURCE_ROOT / "model" / "hyperbolic.py"
_SPEC = importlib.util.spec_from_file_location("candidate_hyperbolic", _MODULE_PATH)
_hyp = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _hyp
_SPEC.loader.exec_module(_hyp)


class HyperbolicReferenceMathTests(unittest.TestCase):
    def test_curved_distance_matches_independent_reference_values(self):
        # For x=(.3,.4), y=(-.2,.1): ||x-y||^2=.34, ||x||^2=.25,
        # ||y||^2=.05, hence z=1+0.68/(.75*.95). The expected value
        # is independently evaluated as acosh(z)^2.
        x = torch.tensor([[0.3, 0.4]], dtype=torch.float64)
        y = torch.tensor([[-0.2, 0.1]], dtype=torch.float64)
        got = _hyp.squared_distance(x, y, c=1.0)
        self.assertAlmostEqual(float(got.item()), 1.6646490870593222, places=12)

        # Origin to (.5,0): z=1+0.5/(1*.75)=5/3.
        origin = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
        half = torch.tensor([[0.5, 0.0]], dtype=torch.float64)
        got2 = _hyp.squared_distance(origin, half, c=1.0)
        self.assertAlmostEqual(float(got2.item()), 1.2069489608125816, places=12)

    def test_flat_limit_is_four_times_euclidean_squared_distance(self):
        x = torch.tensor([[0.3, 0.4], [0.0, 0.0]], dtype=torch.float64)
        y = torch.tensor([[-0.2, 0.1], [0.5, 0.0]], dtype=torch.float64)
        expected = torch.tensor([1.36, 1.0], dtype=torch.float64)
        torch.testing.assert_close(_hyp.squared_distance(x, y, c=0.0), expected, rtol=0, atol=1e-14)
        torch.testing.assert_close(_hyp.squared_distance(x, y, c=-2.0), expected, rtol=0, atol=1e-14)

    def test_distance_matrix_has_independently_known_pairwise_values(self):
        x = torch.tensor([[0.3, 0.4], [0.0, 0.0]], dtype=torch.float64)
        y = torch.tensor([[-0.2, 0.1], [0.5, 0.0]], dtype=torch.float64)
        matrix = _hyp.squared_distance_matrix(x, y, c=1.0)
        self.assertAlmostEqual(float(matrix[0, 0]), 1.6646490870593222, places=12)
        self.assertAlmostEqual(float(matrix[1, 1]), 1.2069489608125816, places=12)
        torch.testing.assert_close(_hyp.squared_distance_matrix(x, y, c=0.0),
                                   torch.tensor([[1.36, 0.8], [0.2, 1.0]], dtype=torch.float64),
                                   rtol=0, atol=1e-14)

    def test_stable_arcosh_and_closed_ball_projection(self):
        u = torch.tensor([0.0, 2.0 / 3.0], dtype=torch.float64)
        expected = torch.tensor([0.0, 0.9624236501192069], dtype=torch.float64)
        torch.testing.assert_close(_hyp.arcosh_stable(u), expected, rtol=0, atol=1e-14)
        vectors = torch.tensor([[1.2, 1.6], [0.3, 0.4]], dtype=torch.float64)
        projected = _hyp.project_ball(vectors, radius=0.9)
        expected_projection = torch.tensor([[0.54, 0.72], [0.3, 0.4]], dtype=torch.float64)
        torch.testing.assert_close(projected, expected_projection, rtol=0, atol=1e-14)
        torch.testing.assert_close(torch.linalg.vector_norm(projected, dim=-1),
                                   torch.tensor([0.9, 0.5], dtype=torch.float64),
                                   rtol=0, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
