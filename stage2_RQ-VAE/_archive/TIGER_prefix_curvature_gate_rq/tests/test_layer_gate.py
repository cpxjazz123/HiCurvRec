"""Known-answer and reference-consistency tests for the layer-wise gate.

Every expected value here is hand-computable or comes from an independent
numpy reference; nothing is read from the production module's own output.
"""
import math
import unittest

import numpy as np
import torch

from model.layer_gate import (
    LayerExpertAttention, cumulative_prefixes, aggregate_layer_losses, LEVELS, EXPERTS,
)
from model.geometry import expert_squared, fixed_squared
from tests.reference_layer_gate import (
    reference_gate_weights, reference_prefixes, reference_layer_proxy,
)


def tiny_tokens(seed=7):
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randn(5, 4, 32, generator=generator)
    mask = torch.tensor([[True, True, True, True],
                         [True, True, False, False],
                         [False, False, False, False],
                         [True, False, True, False],
                         [True, True, True, True]])
    return tokens, mask


class LayerGateKnownAnswers(unittest.TestCase):
    def test_weights_normalise_over_experts_per_level(self):
        module = LayerExpertAttention()
        torch.manual_seed(3)
        with torch.no_grad():
            module.query.normal_()
        tokens, mask = tiny_tokens()
        weights, attention, effective = module(tokens, mask)
        self.assertEqual(tuple(weights.shape), (5, LEVELS, EXPERTS))
        torch.testing.assert_close(weights.sum(-1), torch.ones(5, LEVELS), atol=1e-6, rtol=0)
        torch.testing.assert_close(attention.sum(-1), torch.ones(5, LEVELS, EXPERTS), atol=1e-6, rtol=0)

    def test_masked_tokens_receive_exactly_zero_attention(self):
        module = LayerExpertAttention()
        tokens, mask = tiny_tokens()
        _, attention, effective = module(tokens, mask)
        for level in range(LEVELS):
            for expert in range(EXPERTS):
                for row in range(tokens.shape[0]):
                    for slot in range(tokens.shape[1]):
                        if not mask[row, slot]:
                            self.assertEqual(float(attention[row, level, expert, slot]), 0.0)
        # The null slot is valid exactly when no real token is valid.
        for row in range(tokens.shape[0]):
            self.assertEqual(bool(effective[row, -1]), not bool(mask[row].any()))

    def test_one_level_never_changes_another_level(self):
        module = LayerExpertAttention()
        tokens, mask = tiny_tokens()
        before, _, _ = module(tokens, mask)
        with torch.no_grad():
            module.query[1] += 0.5
        after, _, _ = module(tokens, mask)
        torch.testing.assert_close(after[:, 0], before[:, 0], atol=0, rtol=0)
        self.assertFalse(torch.allclose(after[:, 1], before[:, 1]))

    def test_reference_gate_matches_production(self):
        module = LayerExpertAttention()
        torch.manual_seed(11)
        with torch.no_grad():
            module.query.normal_()
            module.key.weight.normal_(std=0.3)
            module.value.weight.normal_(std=0.3)
        tokens, mask = tiny_tokens(seed=5)
        weights, attention, _ = module(tokens, mask)
        expected_weights, expected_attention = reference_gate_weights(
            tokens, mask, module.query.detach().numpy(),
            module.key.weight.detach().numpy(), module.key.bias.detach().numpy(),
            module.value.weight.detach().numpy(), module.value.bias.detach().numpy(),
            module.score[0].weight.detach().numpy(), module.score[0].bias.detach().numpy(),
            module.score[2].weight.detach().numpy(), module.score[2].bias.detach().numpy(),
            module.null_token.detach().numpy())
        np.testing.assert_allclose(weights.detach().numpy(), expected_weights, rtol=0, atol=2e-5)
        np.testing.assert_allclose(attention.detach().numpy(), expected_attention, rtol=0, atol=2e-5)

    def test_cumulative_prefix_is_the_exact_running_sum(self):
        codes = [torch.randn(6, 32) for _ in range(3)]
        prefixes = cumulative_prefixes(codes)
        self.assertEqual(tuple(prefixes.shape), (3, 6, 32))
        expected = reference_prefixes([c.numpy() for c in codes])
        np.testing.assert_array_equal(prefixes.numpy(), expected)
        torch.testing.assert_close(prefixes[2], codes[0] + codes[1] + codes[2], atol=0, rtol=0)

    def test_prefix_uses_hard_codewords_not_the_surrogate(self):
        hard = torch.ones(2, 4)
        soft = torch.full((2, 4), 0.25)
        prefix = cumulative_prefixes([hard, hard])
        torch.testing.assert_close(prefix[0], hard, atol=0, rtol=0)
        torch.testing.assert_close(prefix[1], 2 * hard, atol=0, rtol=0)
        self.assertFalse(torch.allclose(prefix[1], hard + soft))

    def test_layer_loss_aggregation_uses_equal_first_version_weights(self):
        values = [torch.tensor(3.0), torch.tensor(6.0), torch.tensor(9.0)]
        total = aggregate_layer_losses(values)
        self.assertAlmostEqual(float(total), 6.0, places=6)
        self.assertAlmostEqual(float(aggregate_layer_losses(values, weights=(0.5, 0.25, 0.25))), 5.25, places=6)

    def test_reference_layer_proxy_matches_the_moment_identity(self):
        generator = torch.Generator().manual_seed(19)
        q = torch.randn(9, 32, generator=generator, dtype=torch.float64)
        triples = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        references = np.arange(9)
        expected = reference_layer_proxy(q.numpy(), triples, references,
                                         lambda x, y: fixed_squared(torch.as_tensor(x),
                                                                    torch.as_tensor(y),
                                                                    scale=1.0))
        from model.estimator import sampled_values
        predicted = sampled_values(q, triples, references,
                                   lambda x, y: fixed_squared(x, y, scale=1.0),
                                   paired_squared=lambda x, y: fixed_squared(
                                       x[:, None, :], y[:, None, :], scale=1.0).reshape(-1))
        np.testing.assert_allclose(predicted.numpy(), expected, rtol=0, atol=1e-9)

    def test_euclidean_expert_is_the_plain_squared_distance(self):
        generator = torch.Generator().manual_seed(23)
        x = torch.randn(32, generator=generator)
        y = torch.randn(32, generator=generator)
        plain = (x - y).square().sum()
        values = expert_squared(x, y, scale=1.0, curvature_on=False)
        self.assertEqual(tuple(values.shape), (3,))
        torch.testing.assert_close(values[1], plain)
        torch.testing.assert_close(expert_squared(x, y, scale=1.0, curvature_on=True)[1], plain)
        self.assertTrue(torch.isfinite(expert_squared(x, y, scale=1.0, curvature_on=True)).all())

    def test_zero_curvature_limit_of_the_spherical_expert(self):
        # A vanishing tangent norm makes the spherical chord angle degenerate;
        # the distance must stay finite and equal the Euclidean limit.
        x = torch.zeros(32)
        y = torch.ones(32) * 1e-6
        values = expert_squared(x, y, scale=1.0, curvature_on=True)
        self.assertTrue(torch.isfinite(values).all())
        np.testing.assert_allclose(values[1].numpy(), (x - y).square().sum().numpy(), rtol=1e-6, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
