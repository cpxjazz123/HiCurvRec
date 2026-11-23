"""Independent numerical and gradient contracts for the two attention axes."""
import importlib.util
import math
from pathlib import Path
import unittest

import torch


_SPEC = importlib.util.spec_from_file_location(
    "issue251_attention", Path(__file__).resolve().parent.parent / "model" / "attention.py"
)
_ATTENTION = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_ATTENTION)
ExpertAttention = _ATTENTION.ExpertAttention


class ExpertAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_hand_computed_two_normalizations(self):
        for dtype, tolerance in ((torch.float32, 2e-6), (torch.float64, 1e-12)):
            model = ExpertAttention().to(dtype=dtype)
            with torch.no_grad():
                model.query.zero_()
                model.query[:, 0] = torch.tensor([-2., -1., 0., 1., 2.], dtype=dtype)
                model.query[:, 1] = torch.tensor([1., -1., 2., 0., -2.], dtype=dtype)
                model.key.weight.zero_()
                model.key.bias.zero_()
                model.key.weight[0, 0] = 2.
                model.key.weight[1, 1] = -1.
                model.value.weight.zero_()
                model.value.bias.zero_()
                model.value.weight[0, 0] = 0.5
                model.value.weight[1, 1] = 2.
                model.value.bias[0] = 0.25
                model.score[0].weight.zero_()
                model.score[0].bias.zero_()
                model.score[0].weight[0, 0] = 0.7
                model.score[0].weight[0, 1] = -0.2
                model.score[0].bias[0] = 0.1
                model.score[2].weight.zero_()
                model.score[2].bias.fill_(0.3)
                model.score[2].weight[0, 0] = 1.3
            tokens = torch.zeros(1, 3, 32, dtype=dtype)
            tokens[0, :, :2] = torch.tensor([[1., 2.], [-2., 0.5], [3., -1.]], dtype=dtype)
            weights, attention, effective = model(tokens, torch.tensor([[True, True, True]]))
            expected_attention, scores = [], []
            for q0, q1 in zip((-2., -1., 0., 1., 2.), (1., -1., 2., 0., -2.)):
                logits = [(q0 * 2. * x - q1 * y) / math.sqrt(32.)
                          for x, y in ((1., 2.), (-2., 0.5), (3., -1.))]
                exponentials = [math.exp(x - max(logits)) for x in logits]
                probabilities = [x / sum(exponentials) for x in exponentials]
                expected_attention.append(probabilities + [0.])
                h0 = sum(p * (0.5 * x + 0.25) for p, x in zip(probabilities, (1., -2., 3.)))
                h1 = sum(p * 2. * y for p, y in zip(probabilities, (2., 0.5, -1.)))
                scores.append(1.3 * math.tanh(0.7 * h0 - 0.2 * h1 + 0.1) + 0.3)
            exp_scores = [math.exp(x - max(scores)) for x in scores]
            expected_weights = [x / sum(exp_scores) for x in exp_scores]
            torch.testing.assert_close(attention[0], torch.tensor(expected_attention, dtype=dtype), atol=tolerance, rtol=tolerance)
            torch.testing.assert_close(weights[0], torch.tensor(expected_weights, dtype=dtype), atol=tolerance, rtol=tolerance)
            torch.testing.assert_close(attention.sum(-1), torch.ones(1, 5, dtype=dtype))
            torch.testing.assert_close(weights.sum(-1), torch.ones(1, dtype=dtype))
            self.assertEqual(effective.tolist(), [[True, True, True, False]])
            self.assertGreater(float(weights.max() - weights.min()), 0.01)

    def test_padding_mask_and_repeated_items(self):
        for dtype in (torch.float32, torch.float64):
            model = ExpertAttention().to(dtype=dtype)
            tokens = torch.randn(2, 3, 32, dtype=dtype)
            mask = torch.tensor([[True, False, True], [False, True, False]])
            weights, attention, _ = model(tokens, mask)
            altered = tokens.clone()
            altered[~mask] = 1000.
            padded = torch.cat((altered, torch.randn(2, 2, 32, dtype=dtype) * 1000.), dim=1)
            padded_mask = torch.cat((mask, torch.zeros(2, 2, dtype=torch.bool)), dim=1)
            other_weights, other_attention, _ = model(padded, padded_mask)
            torch.testing.assert_close(weights, other_weights)
            torch.testing.assert_close(attention[..., :3], other_attention[..., :3])
            self.assertTrue(torch.equal(other_attention[..., 3:], torch.zeros_like(other_attention[..., 3:])))
            self.assertTrue(torch.equal(attention[:, :, :3].masked_select(~mask[:, None, :]), torch.zeros(15, dtype=dtype)))
            repeated, _, _ = model(tokens[[0, 0]], mask[[0, 0]])
            torch.testing.assert_close(repeated[0], repeated[1], rtol=0.,
                                       atol=8*torch.finfo(dtype).eps)

    def test_permutation_equivariance_with_fixed_null_slot(self):
        model = ExpertAttention().double()
        tokens = torch.randn(2, 4, 32, dtype=torch.float64)
        mask = torch.tensor([[True, False, True, True], [False, False, False, False]])
        weights, attention, effective = model(tokens, mask)
        permutation = torch.tensor([2, 0, 3, 1])
        other_weights, other_attention, other_effective = model(tokens[:, permutation], mask[:, permutation])
        torch.testing.assert_close(weights, other_weights, atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(attention[:, :, permutation], other_attention[:, :, :4])
        torch.testing.assert_close(attention[:, :, -1], other_attention[:, :, -1])
        self.assertTrue(torch.equal(effective[:, permutation], other_effective[:, :4]))
        self.assertTrue(torch.equal(effective[:, -1], other_effective[:, -1]))

    def test_single_token_degeneracy(self):
        model = ExpertAttention().double()
        tokens = torch.randn(2, 3, 32, dtype=torch.float64)
        mask = torch.tensor([[False, True, False], [True, False, False]])
        weights, attention, _ = model(tokens, mask)
        torch.testing.assert_close(weights, torch.full((2, 5), 0.2, dtype=torch.float64))
        self.assertTrue(torch.equal(attention[0, :, 1], torch.ones(5, dtype=torch.float64)))
        self.assertTrue(torch.equal(attention[1, :, 0], torch.ones(5, dtype=torch.float64)))
        gradient, = torch.autograd.grad(weights[:, 0].sum(), model.query)
        self.assertTrue(torch.equal(gradient, torch.zeros_like(gradient)))

    def test_empty_and_all_invalid_use_fixed_null(self):
        for dtype in (torch.float32, torch.float64):
            model = ExpertAttention().to(dtype=dtype)
            buffers = dict(model.named_buffers())
            self.assertIn("null_token", buffers)
            self.assertTrue(torch.equal(buffers["null_token"], torch.zeros_like(buffers["null_token"])))
            self.assertFalse(buffers["null_token"].requires_grad)
            self.assertNotIn("null_token", dict(model.named_parameters()))
            for count in (0, 3):
                weights, attention, effective = model(
                    torch.randn(2, count, 32, dtype=dtype), torch.zeros(2, count, dtype=torch.bool))
                self.assertTrue(torch.isfinite(weights).all())
                self.assertTrue(torch.isfinite(attention).all())
                self.assertEqual(attention.shape, (2, 5, count + 1))
                self.assertTrue(torch.equal(attention[..., -1], torch.ones(2, 5, dtype=dtype)))
                self.assertTrue(torch.equal(attention[..., :-1], torch.zeros(2, 5, count, dtype=dtype)))
                self.assertEqual(effective.tolist(), [[False] * count + [True]] * 2)
                torch.testing.assert_close(weights, torch.full((2, 5), 0.2, dtype=dtype))
            weights, attention, effective = model(torch.empty(0, 0, 32, dtype=dtype), torch.empty(0, 0, dtype=torch.bool))
            self.assertEqual(weights.shape, (0, 5))
            self.assertEqual(attention.shape, (0, 5, 1))
            self.assertEqual(effective.shape, (0, 1))

    def test_independent_queries_and_all_gradient_groups_train(self):
        model = ExpertAttention().double()
        torch.testing.assert_close(model.query @ model.query.T, torch.eye(5, dtype=torch.float64), atol=3e-7, rtol=3e-7)
        tokens = torch.randn(3, 4, 32, dtype=torch.float64, requires_grad=True)
        weights, attention, _ = model(tokens, torch.ones(3, 4, dtype=torch.bool))
        self.assertGreater(float((attention[:, 0] - attention[:, 1]).abs().max()), 1e-5)
        coefficients = torch.tensor([0.3, -0.8, 1.7, -0.2, 0.9], dtype=torch.float64)
        loss = (weights * coefficients).sum()
        loss.backward()
        self.assertTrue(torch.all(model.query.grad.norm(dim=-1) > 1e-10))
        for group in (model.key, model.value, model.score):
            gradients = [p.grad for p in group.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(float(g.square().sum()) for g in gradients), 1e-14)
        self.assertGreater(float(tokens.grad.norm()), 1e-10)
        before = model.query.detach().clone()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(parameter.grad, alpha=-0.01)
        self.assertTrue(torch.all((before - model.query).norm(dim=-1) > 0.))

    def test_smooth_directional_finite_differences_of_both_outputs(self):
        model = ExpertAttention().double()
        tokens = (torch.randn(1, 2, 32, dtype=torch.float64) * 0.2).requires_grad_()
        mask = torch.ones(1, 2, dtype=torch.bool)
        groups = [tokens, model.query, model.key.weight, model.value.weight,
                  model.score[0].weight, model.score[2].weight]
        # Independent probes exercise both softmax outputs; attention cannot depend on V/score.
        for output_index in (0, 1):
            output = model(tokens, mask)[output_index]
            probe = torch.randn_like(output)
            gradients = torch.autograd.grad((output * probe).sum(), groups, allow_unused=True)
            for target, gradient in zip(groups, gradients):
                direction = torch.randn_like(target)
                direction /= direction.norm()
                original = target.detach().clone()
                epsilon = 1e-5
                with torch.no_grad():
                    target.copy_(original + epsilon * direction)
                    plus = float((model(tokens, mask)[output_index] * probe).sum())
                    target.copy_(original - epsilon * direction)
                    minus = float((model(tokens, mask)[output_index] * probe).sum())
                    target.copy_(original)
                numerical = (plus - minus) / (2. * epsilon)
                analytical = 0. if gradient is None else float((gradient * direction).sum())
                self.assertAlmostEqual(numerical, analytical, delta=2e-9)


if __name__ == "__main__":
    unittest.main()
