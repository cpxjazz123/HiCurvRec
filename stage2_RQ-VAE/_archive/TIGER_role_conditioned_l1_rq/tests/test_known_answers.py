"""Hand-derived hard decisions and metric answers on minimal inputs."""
import math
import unittest
import torch
from test_reference import load_module


class KnownAnswerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geo = load_module('geometry')
        cls.selection = load_module('selection')

    def test_hand_answers_common_tangent_and_spherical_wrap(self):
        x = torch.tensor([[0., 0.]], dtype=torch.float64)
        y = torch.tensor([[.5, 0.]], dtype=torch.float64)
        torch.testing.assert_close(self.geo.expert_squared(x, y), torch.full((1, 1, 3), .25, dtype=x.dtype), rtol=1e-12, atol=1e-12)
        # A 2*pi tangent displacement returns to the unit sphere origin.
        wrapped = self.geo.expert_squared(x, torch.tensor([[2 * math.pi, 0.]], dtype=x.dtype))
        self.assertLess(wrapped[0, 0, 2].item(), 1e-26)

    def test_balancing_cannot_be_replaced_by_argmin(self):
        cost = torch.tensor([[0., 1e-9], [0., 3e-9]], dtype=torch.float64)
        words = torch.eye(2, dtype=cost.dtype)
        balanced = self.selection.select(cost, words, balanced=True, epsilon=.003, iterations=50)
        self.assertEqual(balanced["ids"].tolist(), [1, 0])
        inference = self.selection.select(cost, words, balanced=True, epsilon=.003, iterations=50, training=False)
        self.assertEqual(inference["ids"].tolist(), [0, 0])
