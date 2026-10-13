"""Contract tests for the additive Euclidean crowding mechanisms."""
import unittest

import torch

import crowding_losses2 as losses


class Crowd2Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.encoder = torch.nn.Linear(5, 4, bias=False)
        self.codebook = torch.nn.Parameter(torch.randn(2, 6, 4))
        self.x = torch.randn(9, 5)
        encoded = self.encoder(self.x)
        levels = []
        for book in self.codebook:
            assignment = torch.softmax(-torch.cdist(encoded, book).square() / 0.25, dim=-1)
            levels.append(encoded + 0.01 * (assignment @ book))
        self.residual_levels = torch.stack(levels)
        self.z = self.residual_levels.mean(0)
        self.anchors = torch.tensor([0, 1, 2, 3])
        self.positives = torch.tensor([[4, 5, 6], [5, -1, -1], [6, 7, -1], [7, 8, -1]])
        self.positive_mask = self.positives >= 0
        self.negatives = torch.tensor([[1, 2], [0, 3], [0, 1], [0, 2]])
        self.behavior = (self.residual_levels[:, self.anchors] -
                         self.residual_levels[:, self.positives[:, 0]]).square().mean()
        self.kwargs = dict(z=self.z, anchors=self.anchors, positives=self.positives,
                           positive_mask=self.positive_mask, negatives=self.negatives,
                           codebook=self.codebook, residual_levels=self.residual_levels,
                           pmi_weights=torch.linspace(0.2, 1.0, self.positives.shape[1]).expand(self.anchors.shape[0], -1),
                           behavior_loss=self.behavior)

    def run_loss(self, name, strength=losses.STRENGTHS[1]):
        return losses.mechanism_loss2(name, strength=strength, **self.kwargs)

    def test_t1_additivity(self):
        for name in losses.ALL_ARMS:
            result = self.run_loss(name, strength=0.0)
            self.assertTrue(torch.equal(losses.additive_total(self.behavior, result), self.behavior), name)

    def test_t2_gradient_reachability(self):
        for name in losses.ALL_ARMS:
            term = self.run_loss(name)["losses"][name]
            grads = torch.autograd.grad(term, (self.encoder.weight, self.codebook),
                                        retain_graph=True, allow_unused=True)
            self.assertIsNotNone(grads[0], name + " encoder")
            self.assertGreater(torch.count_nonzero(grads[0]).item(), 0, name + " encoder")
            self.assertIsNotNone(grads[1], name + " codebook")
            self.assertGreater(torch.count_nonzero(grads[1]).item(), 0, name + " codebook")

    def test_t3_ste_deep_residual_gradient(self):
        for name in losses.ALL_ARMS:
            result = self.run_loss(name)
            grads = torch.autograd.grad(result["losses"][name], result["tensors"]["residual_levels"],
                                        retain_graph=True, allow_unused=True)[0]
            self.assertIsNotNone(grads, name)
            self.assertGreater(torch.count_nonzero(grads[1]).item(), 0, name)
            self.assertIn("not detached", result["gradient_path"])

    def test_t4_scale_is_relative_order_invariant(self):
        result = self.run_loss("scale")
        original = self.z
        scaled = result["tensors"]["scaled_latents"]
        self.assertTrue(torch.equal(torch.argsort(torch.cdist(original, original).flatten()),
                                    torch.argsort(torch.cdist(scaled, scaled).flatten())))
        self.assertTrue(torch.equal(result["losses"]["scale"],
                                    losses.scale_added_term(original, result["scale_factor"])))

    def test_t5_finite_scalar_contract(self):
        for name in losses.ALL_ARMS:
            for strength in (0.0, *losses.STRENGTHS):
                value = self.run_loss(name, strength)["losses"][name]
                self.assertEqual(value.ndim, 0, (name, strength))
                self.assertTrue(torch.isfinite(value).item(), (name, strength))

    def test_t6_determinism(self):
        for name in losses.ALL_ARMS:
            a = self.run_loss(name)["losses"][name]
            b = self.run_loss(name)["losses"][name]
            self.assertTrue(torch.equal(a, b), name)

    def test_t7_multipos_multiplicity_response(self):
        result = self.run_loss("multipos")
        per_anchor = result["tensors"]["multipos_per_anchor"]
        self.assertNotEqual(per_anchor[0].item(), per_anchor[1].item())
        self.assertEqual(int(self.positive_mask[0].sum()), 3)
        self.assertEqual(int(self.positive_mask[1].sum()), 1)

    def test_t8_tensor_semantics_documented(self):
        for name in losses.ALL_ARMS:
            result = self.run_loss(name)
            self.assertTrue(result["tensor_semantics"])
            self.assertTrue(result["gradient_target"])
            self.assertIn("not detached", result["gradient_path"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
