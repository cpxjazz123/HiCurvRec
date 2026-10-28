"""The measured curvature must reach the level's own assignment and commitment.

The derivation is only half of the mechanism: if the layer keeps using a fixed
curvature, or the trainer measures the statistic and never applies it, the loop is
open and the candidate is the previous mechanism with extra bookkeeping.
"""
import unittest
from types import SimpleNamespace

import torch

from model.curvature_control import exp0_c, level_curvatures, measure_levels, squared_distance_c
from model.layers import VQLayer


def _layer(curvature=1.0, use_behaviour=True, use_geometry=True):
    layer = VQLayer(4, 2, beta=0.25, sk_epsilon=-1.0, sk_iters=-1,
                    frechet=True, use_behaviour=use_behaviour,
                    use_curvature=use_geometry)
    with torch.no_grad():
        layer.get_code_embs().copy_(torch.tensor(
            [[0.10, 0.0], [0.30, 0.25], [-0.35, 0.35], [0.60, 0.0]]))
    layer.frechet.set_weights(torch.ones(8))   # the table is indexed by ITEM
    layer.frechet.curvature = float(curvature)
    return layer


class LayerTests(unittest.TestCase):

    def test_the_layer_reads_the_curvature_it_is_given(self):
        layer = _layer(curvature=2.0)
        self.assertEqual(layer.curvature(), 2.0)

    def test_the_cost_is_the_curved_distance_of_that_curvature(self):
        latent = torch.tensor([[0.25, 0.0], [-0.1, 0.05]])
        low = _layer(curvature=0.0)
        high = _layer(curvature=2.0)
        cost_low = low.assignment_cost(latent)
        cost_high = high.assignment_cost(latent)
        point = exp0_c(latent, 2.0)
        mapped_codes = exp0_c(high.get_code_embs(), 2.0)
        expected = squared_distance_c(point.unsqueeze(1), mapped_codes.unsqueeze(0), 2.0)
        self.assertTrue(torch.allclose(cost_high, expected, atol=1e-5))
        euclid = (exp0_c(latent, 0.0).unsqueeze(1) - low.get_code_embs().unsqueeze(0)).pow(2).sum(-1)
        self.assertTrue(torch.allclose(cost_low, euclid, atol=1e-6))
        self.assertFalse(torch.allclose(cost_low, cost_high))

    def test_the_assignment_can_change_with_the_curvature(self):
        """A radius-sensitive ordering is what the mechanism claims to buy.

        With this latent the Euclidean metric prefers the off-axis code 1 and the
        ball of curvature 4 prefers the radial code 0: the two metrics do not rank
        the same codes, which is the only way the measured curvature can change a
        SID.
        """
        latent = torch.tensor([[0.35, 0.05]])
        chosen = {c: int(torch.argmin(_layer(curvature=c).assignment_cost(latent)))
                  for c in (0.0, 4.0)}
        self.assertEqual(chosen[0.0], 1)
        self.assertEqual(chosen[4.0], 0)

    def test_zero_curvature_is_the_geometry_off_arm(self):
        layer = _layer(curvature=0.0, use_geometry=False)
        latent = torch.tensor([[0.2, 0.1]])
        self.assertEqual(layer.curvature(), 0.0)
        cost = layer.assignment_cost(latent)
        euclid = (latent.unsqueeze(1) - layer.get_code_embs().unsqueeze(0)).pow(2).sum(-1)
        self.assertTrue(torch.allclose(cost, euclid, atol=1e-6))


class MeasurementTests(unittest.TestCase):

    def test_the_measurement_updates_one_value_per_level(self):
        residuals = [torch.randn(6, 4) * 0.1, torch.randn(6, 4) * 0.1, torch.randn(6, 4) * 0.1]
        prefixes = torch.tensor([[0, 0, 0], [0, 1, 1], [1, 1, 0], [2, 2, 2],
                                 [2, 2, 1], [0, 0, 1]])
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
        state = {}
        measure_levels(residuals, prefixes, edges, curvature_state=state,
                       use_behaviour=True, use_geometry=True, epoch=1)
        self.assertEqual(sorted(state["values"]), [0, 1, 2])
        for value in state["values"].values():
            self.assertGreater(value, 0.0)

    def test_the_measurement_respects_the_switches(self):
        residuals = [torch.randn(4, 4)] * 3
        prefixes = torch.tensor([[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1]])
        state = {}
        measure_levels(residuals, prefixes, [(0, 1), (2, 3)], curvature_state=state,
                       use_behaviour=False, use_geometry=True, epoch=1)
        values = level_curvatures(state, use_behaviour=False, use_geometry=True)
        self.assertTrue(all(value == 1.0 for value in values))


if __name__ == "__main__":
    unittest.main()
