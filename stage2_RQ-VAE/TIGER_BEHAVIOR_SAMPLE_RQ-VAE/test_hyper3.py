"""Acceptance tests for the additive hyperbolic mechanisms and controls."""
import unittest

import torch

import hyper_losses3 as losses


class Hyper3Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.z = torch.randn(24, 5, dtype=torch.float32, requires_grad=True)
        self.levels = torch.stack((self.z, self.z * 0.7, self.z * 0.4))
        self.book = torch.randn(3, 7, 5, dtype=torch.float32, requires_grad=True)
        self.anchors = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        self.positives = torch.tensor([[8, 9], [9, 10], [10, 11], [11, 12],
                                       [12, 13], [13, 14], [14, 15], [15, 16]])
        self.mask = torch.ones_like(self.positives, dtype=torch.bool)
        self.negatives = torch.tensor([[17, 18], [18, 19], [19, 20], [20, 21],
                                       [21, 22], [22, 23], [16, 17], [17, 18]])
        self.behavior = torch.tensor(1.2345, dtype=torch.float32, requires_grad=True)
        self.kwargs = dict(z=self.z, anchors=self.anchors, positives=self.positives,
                           positive_mask=self.mask, negatives=self.negatives,
                           codebook=self.book, residual_levels=self.levels, curvature=1.0)

    def run_loss(self, name, strength=losses.STRENGTHS[1], **kwargs):
        args = dict(self.kwargs)
        args.update(kwargs)
        return losses.mechanism_loss3(name, strength=strength, **args)

    def test_t1_additive_identity(self):
        for name in losses.ALL_ARMS:
            result = self.run_loss(name, strength=0.0)
            self.assertTrue(torch.equal(losses.additive_total(self.behavior, result), self.behavior), name)

    def test_t2_gradient_reachability(self):
        for name in losses.ALL_ARMS:
            self.z.grad = None
            self.book.grad = None
            result = self.run_loss(name)
            result["losses"][name].backward(retain_graph=True)
            self.assertIsNotNone(self.z.grad, name + " encoder")
            self.assertGreater(torch.count_nonzero(self.z.grad).item(), 0, name + " encoder")
            if name in ("geo", "hassign"):
                self.assertIsNotNone(self.book.grad, name + " codebook")
                self.assertGreater(torch.count_nonzero(self.book.grad).item(), 0, name + " codebook")
            if name == "hassign":
                quantized, ids = losses.quantize_hyperbolic(self.z, self.book[0], curvature=1.0)
                self.assertTrue(torch.equal(quantized.detach(), self.book[0][ids].detach()))
                self.z.grad = None
                quantized.sum().backward()
                self.assertGreater(torch.count_nonzero(self.z.grad).item(), 0, "hassign STE")

    def test_t3_finite_strength_ladder_and_ball_extremes(self):
        extremes = torch.tensor([[0.99999, 0.0, 0.0], [1.0001, 0.0, 0.0],
                                 [0.0, 0.99999, 0.0], [0.0, 1.0001, 0.0]], dtype=torch.float32)
        extreme_levels = extremes.unsqueeze(0).expand(3, -1, -1).clone()
        extreme_kwargs = dict(z=extremes, anchors=torch.tensor([0, 1]),
                              positives=torch.tensor([[2], [3]]),
                              positive_mask=torch.ones((2, 1), dtype=torch.bool),
                              negatives=torch.tensor([[3], [2]]),
                              codebook=torch.tensor([[[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]],
                                                     [[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]],
                                                     [[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]]]),
                              residual_levels=extreme_levels, curvature=4.0)
        for name in losses.ALL_ARMS:
            for strength in losses.STRENGTHS:
                result = self.run_loss(name, strength=strength)
                self.assertTrue(torch.isfinite(result["losses"][name]).all().item(), (name, strength))
            mapped = losses.map_to_ball(extremes, curvature=4.0)
            self.assertTrue(torch.isfinite(mapped).all().item(), name + " mapped extremes")
            distances = losses.poincare_distance(mapped[:2], mapped[2:], curvature=4.0)
            self.assertTrue(torch.isfinite(distances).all().item(), name + " extreme distances")
            extreme_result = losses.mechanism_loss3(name, **extreme_kwargs)
            self.assertTrue(torch.isfinite(extreme_result["losses"][name]).all().item(),
                            name + " extreme mechanism output")

    def test_t4_curvature_changes_loss(self):
        low = self.run_loss("geo", curvature=0.25)["losses"]["geo"]
        high = self.run_loss("geo", curvature=4.0)["losses"]["geo"]
        self.assertFalse(torch.allclose(low, high, rtol=1e-6, atol=1e-7))

    def test_t5_median_radius_pinned(self):
        radii = []
        for curvature in losses.CURVATURES:
            radii.append(torch.linalg.vector_norm(losses.map_to_ball(self.z, curvature), dim=-1).median())
        target = radii[0]
        for radius in radii[1:]:
            self.assertLessEqual(abs((radius - target).item()), losses.RADIUS_TOLERANCE)

    def test_t6_radial_control_inert_margin_order(self):
        base = self.run_loss("geo")["tensors"]["relative_margins"]
        controlled = self.run_loss("radial")["tensors"]["relative_margins"]
        self.assertTrue(torch.equal(torch.argsort(base), torch.argsort(controlled)))

    def test_t7_eucpair_faithful_pair(self):
        hyper = self.run_loss("geo", strength=losses.STRENGTHS[1])
        euclid = self.run_loss("eucpair", strength=losses.STRENGTHS[1])
        self.assertEqual(hyper["losses"]["geo"].shape, euclid["losses"]["eucpair"].shape)
        self.assertEqual(hyper["weight"], euclid["weight"])
        self.assertEqual(sum(p.numel() for p in losses.parameters_for("geo", **self.kwargs)),
                         sum(p.numel() for p in losses.parameters_for("eucpair", **self.kwargs)))
        self.assertFalse(torch.allclose(hyper["tensors"]["pair_distances"],
                                        euclid["tensors"]["pair_distances"]))

    def test_t8_deterministic_same_seed(self):
        for name in losses.ALL_ARMS:
            torch.manual_seed(42)
            first = self.run_loss(name)["losses"][name].detach().clone()
            torch.manual_seed(42)
            second = self.run_loss(name)["losses"][name].detach().clone()
            self.assertTrue(torch.equal(first, second), name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
