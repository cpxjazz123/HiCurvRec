import importlib.util
import math
import pathlib
import unittest

import torch


_SOURCE = pathlib.Path("/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_c117_asymmetric_successor_branches/model/hyperbolic.py")
_SPEC = importlib.util.spec_from_file_location("candidate_hyperbolic", _SOURCE)
_hyperbolic = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_hyperbolic)


class HyperbolicReferenceTests(unittest.TestCase):
    def test_flat_limit_is_four_times_euclidean_squared_distance(self):
        x = torch.tensor([[0.0, 0.0], [0.3, -0.4]], dtype=torch.float64)
        y = torch.tensor([[0.5, 0.0], [-0.1, 0.2]], dtype=torch.float64)
        # Independently evaluated: 4*(0.5^2 + 0.5^2) = 2 and
        # 4*((0.4)^2 + (-0.6)^2) = 2.08.
        expected = torch.tensor([1.0, 2.08], dtype=torch.float64)
        actual = _hyperbolic.squared_distance(x, y, c=0.0)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-12)

    def test_curved_origin_to_point_reference_value(self):
        # For x=0, y=(0.6,0), c=1: denominator=1-0.36=0.64,
        # u=2*0.36/0.64=1.125; 1+u=2.125 and
        # sqrt(u*(u+2))=1.875. Thus arcosh(2.125)=ln(4), and D^2=(ln 4)^2.
        x = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
        y = torch.tensor([[0.6, 0.0]], dtype=torch.float64)
        actual = _hyperbolic.squared_distance(x, y, c=1.0)
        self.assertAlmostEqual(actual.item(), math.log(4.0) ** 2, places=12)

    def test_curved_distance_is_symmetric_and_matrix_matches_pairwise(self):
        x = torch.tensor([[0.0, 0.0], [0.3, 0.4]], dtype=torch.float64)
        y = torch.tensor([[0.6, 0.0], [-0.2, 0.1]], dtype=torch.float64)
        pairwise = _hyperbolic.squared_distance(x[:, None, :], y[None, :, :], c=1.0)
        reverse = _hyperbolic.squared_distance(y[:, None, :], x[None, :, :], c=1.0)
        matrix = _hyperbolic.squared_distance_matrix(x, y, c=1.0)
        torch.testing.assert_close(pairwise, reverse.T, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(matrix, pairwise, rtol=1e-12, atol=1e-12)
        self.assertEqual(tuple(matrix.shape), (2, 2))

    def test_projection_enforces_closed_radius_and_preserves_inside_point(self):
        x = torch.tensor([[3.0, 4.0], [0.3, 0.4]], dtype=torch.float64)
        projected = _hyperbolic.project_ball(x, radius=0.9)
        norms = torch.linalg.vector_norm(projected, dim=-1)
        self.assertAlmostEqual(norms[0].item(), 0.9, places=12)
        torch.testing.assert_close(projected[1], x[1], rtol=0.0, atol=1e-12)
