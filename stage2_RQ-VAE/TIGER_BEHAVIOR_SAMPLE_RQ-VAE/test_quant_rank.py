"""Tests for ranking supervision on straight-through quantized RQ representations."""

import unittest

import torch
from torch import nn
import torch.nn.functional as F
import geom_losses
import quant_rank_config
import quant_rank_loss
from model.layers import VQLayer


class QuantizedRankingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.encoder = nn.Linear(5, 3, bias=False)
        self.layers = nn.ModuleList([VQLayer(8, 3) for _ in range(3)])
        with torch.no_grad():
            for layer in self.layers:
                layer.embed.weight.copy_(torch.randn_like(layer.embed.weight) * 0.3)
        self.inputs = torch.tensor([
            [1.0, 0.2, 0.3, -0.4, 0.5],
            [0.1, 1.1, -0.2, 0.3, -0.1],
            [-0.4, 0.2, 1.2, 0.1, 0.2],
            [0.7, -0.8, 0.1, 1.0, 0.4],
            [-0.3, 0.4, -0.9, 0.2, 1.1],
            [0.9, 0.3, -0.5, -0.7, 0.6],
        ])
        self.residuals, self.token_ids, self.raw_quantized = self.encode_and_quantize()
        self.anchors = torch.tensor([0, 1])
        self.positives = torch.tensor([2, 3])
        self.negatives = torch.tensor([4, 5])

    def encode_and_quantize(self):
        z = self.encoder(self.inputs)
        residual = z
        traces, tokens, quantized = [], [], []
        for layer in self.layers:
            traces.append(residual)
            q, _, _, ids = layer(residual)
            tokens.append(ids)
            quantized.append(torch.nn.functional.embedding(ids, layer.get_code_embs()))
            residual = residual - q
        return torch.stack(traces), torch.stack(tokens), torch.stack(quantized)

    def loss(self, *, levels=None, negatives=None):
        residuals, tokens, _ = self.encode_and_quantize()
        return quant_rank_loss.quantized_ranking_loss(
            residuals, tokens, self.layers,
            self.anchors, self.positives,
            self.negatives if negatives is None else negatives,
            arm="E1", weight=0.2,
            level_weights=quant_rank_config.LEVEL_WEIGHTS if levels is None else levels,
        )

    def test_t1_codebook_and_encoder_gradients_reach_all_levels(self):
        self.assertEqual(quant_rank_config.ACTIVE_GROUP, "A")
        self.assertEqual(quant_rank_config.GROUP_CONFIGS["A"]["level_weights"], (1.0, 1.0, 1.0))
        self.assertEqual(quant_rank_config.GROUP_CONFIGS["B"]["level_weights"], (1.0, 1.0, 1.0))
        self.loss().backward()
        self.assertIsNotNone(self.encoder.weight.grad)
        self.assertGreater(self.encoder.weight.grad.norm().item(), 0.0)
        for layer in self.layers:
            grad = layer.get_code_embs().grad
            self.assertIsNotNone(grad)
            self.assertGreater(grad.norm().item(), 0.0)
        for active in range(3):
            self.encoder.zero_grad(set_to_none=True)
            for layer in self.layers:
                layer.zero_grad(set_to_none=True)
            weights = tuple(1.0 if i == active else 0.0 for i in range(3))
            residuals, tokens, _ = self.encode_and_quantize()
            single = quant_rank_loss.quantized_ranking_loss(
                residuals, tokens, self.layers,
                torch.tensor([0]), torch.tensor([2]),
                torch.tensor([{0: 1, 1: 5, 2: 4}[active]]),
                arm="E1", weight=0.2, level_weights=weights,
            )
            single.backward()
            self.assertIsNotNone(self.encoder.weight.grad)
            self.assertGreater(self.encoder.weight.grad.norm().item(), 0.0)
            self.assertGreater(
                self.layers[active].get_code_embs().grad.norm().item(), 0.0,
                msg="active level {} gradient".format(active),
            )

    def test_t2_forward_value_is_sum_of_quantized_levels_and_codebook_changes_loss(self):
        representation = quant_rank_loss.quantized_representation(
            self.residuals, self.token_ids, self.layers,
        )
        self.assertTrue(torch.equal(representation.detach(), self.raw_quantized.sum(dim=0)))
        raw = self.raw_quantized.sum(dim=0)
        expected = 0.2 * F.softplus(
            geom_losses.distance("E1", raw[self.anchors], raw[self.positives], 1.0)
            - geom_losses.distance("E1", raw[self.anchors], raw[self.negatives], 1.0)
        ).mean()
        self.assertTrue(torch.equal(self.loss().detach(), expected))
        before = self.loss().detach()
        with torch.no_grad():
            self.layers[0].embed.weight.add_(0.37)
        self.assertFalse(torch.equal(before, self.loss().detach()))

    def test_t3_ste_gradient_is_finite_nonzero_and_robust_to_far_codebook(self):
        loss = self.loss()
        loss.backward()
        for layer in self.layers:
            grad = layer.get_code_embs().grad
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.norm().item(), 0.0)
        with torch.no_grad():
            for layer in self.layers:
                layer.embed.weight.add_(10000.0)
        perturbed = self.loss()
        self.assertTrue(torch.isfinite(perturbed))
        self.assertNotEqual(loss.detach().item(), perturbed.detach().item())

    def test_t4_zero_weight_composition_is_behavior_identity(self):
        behavior = torch.tensor(0.1234567, requires_grad=True)
        zero = quant_rank_loss.quantized_ranking_loss(
            self.residuals, self.token_ids, self.layers,
            self.anchors, self.positives, self.negatives,
            arm="E1", weight=0.0,
        )
        self.assertTrue(torch.equal(quant_rank_loss.compose_loss(behavior, zero), behavior))
        self.assertTrue(torch.equal(behavior, torch.tensor(0.1234567)))

    def test_t5_real_negative_items_change_loss(self):
        baseline = self.loss()
        negative_four = self.loss(negatives=torch.tensor([4, 4]))
        negative_five = self.loss(negatives=torch.tensor([5, 5]))
        self.assertFalse(torch.equal(negative_four, negative_five))
        changed_ids = self.token_ids.clone()
        changed_ids[0, 4] = (changed_ids[0, 4] + 1) % self.layers[0].n_embed
        changed_loss = quant_rank_loss.quantized_ranking_loss(
            self.residuals, changed_ids, self.layers,
            self.anchors, self.positives, self.negatives,
            arm="E1", weight=0.2,
        )
        self.assertFalse(torch.equal(baseline, changed_loss))

    def test_t6_determinism_at_seed_42(self):
        def run():
            torch.manual_seed(42)
            return self.loss().detach()
        self.assertTrue(torch.equal(run(), run()))

    def test_t7_level_weights_route_gradient_only_to_selected_levels(self):
        for selected in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
            self.encoder.zero_grad(set_to_none=True)
            for layer in self.layers:
                layer.zero_grad(set_to_none=True)
            residuals, tokens, _ = self.encode_and_quantize()
            active = selected.index(1.0)
            single = quant_rank_loss.quantized_ranking_loss(
                residuals, tokens, self.layers,
                torch.tensor([0]), torch.tensor([2]),
                torch.tensor([{0: 1, 1: 5, 2: 4}[active]]),
                arm="E1", weight=0.2, level_weights=selected,
            )
            single.backward()
            self.assertIsNotNone(self.encoder.weight.grad)
            self.assertGreater(self.encoder.weight.grad.norm().item(), 0.0)
            for i, layer in enumerate(self.layers):
                grad = layer.get_code_embs().grad
                if selected[i]:
                    self.assertIsNotNone(grad)
                    self.assertGreater(grad.norm().item(), 0.0)
                else:
                    self.assertTrue(grad is None or torch.count_nonzero(grad) == 0)

    def test_t8_loss_does_not_change_selected_token_ids(self):
        before = [layer(residual)[3] for layer, residual in zip(self.layers, self.residuals)]
        self.loss().backward()
        after = [layer(residual)[3] for layer, residual in zip(self.layers, self.residuals)]
        for original, selected in zip(before, after):
            self.assertTrue(torch.equal(original, selected))


if __name__ == "__main__":
    unittest.main(verbosity=2)
