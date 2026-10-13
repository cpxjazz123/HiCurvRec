"""Tests for auxiliary anti-collapse losses and soft quantization."""

import unittest

import torch
from torch import nn

from model.layers import VQLayer


class AntiCollapseTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.encoder = nn.Linear(4, 3, bias=False)
        self.layers = nn.ModuleList([VQLayer(8, 3) for _ in range(3)])
        with torch.no_grad():
            for layer in self.layers:
                layer.embed.weight.copy_(torch.randn_like(layer.embed.weight) * 0.3)
        self.inputs = torch.randn(6, 4)
        z = self.encoder(self.inputs)
        traces = []
        residual = z
        for layer in self.layers:
            traces.append(residual)
            q, _, _, _ = layer(residual)
            residual = residual - q
        self.residuals = torch.stack(traces)

    def test_t1_align_loss_value_and_codebook_gradient(self):
        import anti_collapse_losses as losses

        z = self.residuals[0]
        self.assertEqual(losses.align_loss(self.residuals, z), 0)
        q = self.layers[0].get_code_embs()[0].expand_as(z)
        loss = losses.align_loss(self.residuals, q)
        self.assertGreater(loss.item(), 0)
        loss.backward()
        self.assertGreater(self.layers[0].get_code_embs().grad.norm().item(), 0)

    def test_t2_diversity_uniform_skewed_and_collapsed(self):
        import anti_collapse_losses as losses

        layer = VQLayer(4, 1)
        with torch.no_grad():
            layer.embed.weight.zero_()
        uniform_residuals = torch.zeros(1, 4, 1)
        self.assertEqual(losses.diversity_loss(uniform_residuals, [layer], 0.5), 0)
        with torch.no_grad():
            layer.embed.weight.copy_(torch.tensor([[0.0], [20.0], [40.0], [60.0]]))
        collapsed_residuals = torch.zeros(1, 4, 1)
        collapsed = losses.diversity_loss(collapsed_residuals, [layer], 0.5)
        self.assertGreater(collapsed.item(), 0.7)
        skewed = losses.diversity_loss(torch.tensor([[[0.0], [0.0], [0.0], [20.0]]]), [layer], 0.5)
        self.assertGreater(skewed.item(), 0.0)

    def test_t3_diversity_has_codebook_gradient_at_each_level(self):
        import anti_collapse_losses as losses

        loss = losses.diversity_loss(self.residuals, self.layers, 0.5)
        loss.backward()
        for layer in self.layers:
            grad = layer.get_code_embs().grad
            self.assertIsNotNone(grad)
            self.assertGreater(grad.norm().item(), 0.0)

    def test_t4_gumbel_gradients_and_hard_forward_codeword(self):
        import anti_collapse_losses as losses

        soft = losses.gumbel_quantized_representation(self.residuals, self.layers, 0.5)
        soft.square().sum().backward()
        self.assertGreater(self.encoder.weight.grad.norm().item(), 0.0)
        for layer in self.layers:
            self.assertGreater(layer.get_code_embs().grad.norm().item(), 0.0)

        hard_layers = nn.ModuleList([VQLayer(8, 3) for _ in range(3)])
        with torch.no_grad():
            for layer in hard_layers:
                layer.embed.weight.zero_()
            hard_layers[0].embed.weight.copy_(self.layers[0].embed.weight.detach())
        hard = losses.gumbel_quantized_representation(self.residuals.detach(), hard_layers, 0.5, hard=True)
        self.assertTrue(any(torch.equal(hard[0], codeword) for codeword in hard_layers[0].get_code_embs()))
        hard.square().sum().backward()
        self.assertGreater(hard_layers[0].get_code_embs().grad.norm().item(), 0.0)

    def test_t5_real_vq_indices_are_unchanged(self):
        import anti_collapse_losses as losses

        before = [layer(residual)[3].clone() for layer, residual in zip(self.layers, self.residuals)]
        losses.align_loss(self.residuals, self.residuals[0] + 1)
        losses.diversity_loss(self.residuals, self.layers, 0.5)
        losses.gumbel_quantized_representation(self.residuals, self.layers, 0.5)
        losses.compose(torch.tensor(1.0), torch.tensor(0.0))
        after = [layer(residual)[3] for layer, residual in zip(self.layers, self.residuals)]
        for original, selected in zip(before, after):
            self.assertTrue(torch.equal(original, selected))

    def test_t6_zero_term_composition_is_identical(self):
        import anti_collapse_losses as losses

        behavior = torch.tensor(0.1234567, requires_grad=True)
        self.assertIs(losses.compose(behavior, torch.tensor(0.0)), behavior)

    def test_t7_determinism_at_seed_42(self):
        import anti_collapse_losses as losses

        def run():
            torch.manual_seed(42)
            return losses.gumbel_quantized_representation(self.residuals, self.layers, 0.5)

        self.assertTrue(torch.equal(run(), run()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
