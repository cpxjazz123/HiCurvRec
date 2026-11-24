"""Issue253 ambient/scalar oracles; independent of production arithmetic."""
import importlib.util
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "stage2_RQ-VAE").is_dir())
TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TREE))
from tools.issue253.reference import ambient_squared, scalar_attention


def load(name):
    spec = importlib.util.spec_from_file_location("three253_" + name, TREE / "model" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


geometry = load("geometry")
ExpertAttention = load("attention").ExpertAttention


def reference_attention(model, tokens, mask):
    arrays = [p.detach().cpu().numpy() for p in (
        model.query, model.key.weight, model.key.bias,
        model.value.weight, model.value.bias, model.score[0].weight,
        model.score[0].bias, model.score[2].weight, model.score[2].bias)]
    return scalar_attention(tokens.detach().cpu().numpy(), mask.cpu().numpy(), *arrays)


class ThreeGeometryTests(unittest.TestCase):
    def test_exact_curvatures_and_no_removed_expert_computation(self):
        self.assertEqual(geometry.CURVATURES, (-1, 0, 1))
        x = torch.tensor([[.3, -.2]], dtype=torch.double)
        y = torch.tensor([[-.4, .6]], dtype=torch.double)
        with patch.object(geometry, "_hyperbolic_squared", wraps=geometry._hyperbolic_squared) as negative, \
             patch.object(geometry, "_sphere_squared", wraps=geometry._sphere_squared) as positive:
            result = geometry.expert_squared(x, y)
        self.assertEqual(result.shape, (1, 1, 3))
        self.assertEqual([call.args[3] for call in negative.call_args_list], [-1],
                         "only curvature -1 is evaluated, never -3 then sliced")
        self.assertEqual([call.args[4] for call in positive.call_args_list], [1],
                         "only curvature +1 is evaluated, never +3 then sliced")

    def test_common_scale_ambient_matrix_broadcast_and_row_pairs(self):
        generator = torch.Generator().manual_seed(253)
        for dtype in (torch.float32, torch.float64):
            x = torch.randn(3, 32, generator=generator, dtype=dtype) * .08
            y = torch.randn(4, 32, generator=generator, dtype=dtype) * .09
            expected = torch.from_numpy(ambient_squared(x.numpy(), y.numpy(), scale=1.7)).to(dtype)
            atol, rtol = (1e-5, 1e-4) if dtype == torch.float32 else (1e-9, 1e-8)
            matrix = geometry.expert_squared(x, y, scale=1.7)
            torch.testing.assert_close(matrix, expected, atol=atol, rtol=rtol)
            torch.testing.assert_close(geometry.expert_squared(x[:, None], y[None], scale=1.7),
                                       expected, atol=atol, rtol=rtol)
            pairs = geometry.expert_squared(x[:, None], y[:3, None], scale=1.7)
            torch.testing.assert_close(pairs[:, 0], expected[torch.arange(3), torch.arange(3)],
                                       atol=atol, rtol=rtol)
            torch.testing.assert_close(matrix, geometry.expert_squared(x / 1.7, y / 1.7),
                                       atol=atol, rtol=rtol)
            torch.testing.assert_close(geometry.expert_squared(x * 2, y * 2, scale=3.4), matrix,
                                       atol=atol, rtol=rtol)

    def test_zero_curvature_limit_from_common_tangent_scaling(self):
        # s^2 D_{+/-1}(x/s,y/s)^2 = D_{+/-1/s^2}(x,y)^2.
        # This checks the production maps, not just the reference's c=0 branch.
        x = torch.tensor([[.3, -.2], [-.1, .4]], dtype=torch.double)
        y = torch.tensor([[-.4, .1], [.2, .6]], dtype=torch.double)
        euclidean = (x[:, None] - y[None]).square().sum(-1)
        errors = []
        for scale in (10., 100., 10000.):
            limit = geometry.expert_squared(x, y, scale=scale) * scale ** 2
            errors.append(float((limit - euclidean[..., None]).abs().max()))
        self.assertLess(errors[1], errors[0] / 50)
        self.assertLess(errors[2], errors[1] / 1000)
        torch.testing.assert_close(limit, euclidean[..., None].expand(-1, -1, 3), atol=1e-9, rtol=1e-8)
        off = geometry.expert_squared(x, y, curvature_on=False)
        torch.testing.assert_close(off, euclidean[..., None].expand(-1, -1, 3), atol=0, rtol=1e-14)

    def test_ambient_directional_gradient_for_inputs_and_mixing(self):
        x = torch.tensor([[.3, -.2], [-.1, .4]], dtype=torch.double, requires_grad=True)
        y = torch.tensor([[-.4, .1], [.2, .6]], dtype=torch.double, requires_grad=True)
        weights = torch.tensor([[.2, .3, .5], [.7, .2, .1]], dtype=torch.double, requires_grad=True)
        arguments = (x, y, weights)
        actual = geometry.mixed_cost(x, y, weights, scale=1.3).sum()
        gradients = torch.autograd.grad(actual, arguments)
        directions = (torch.tensor([[.2, -.3], [.1, .4]], dtype=torch.double),
                      torch.tensor([[-.1, .2], [.4, .3]], dtype=torch.double),
                      torch.tensor([[.1, -.2, .3], [-.4, .2, .1]], dtype=torch.double))
        arrays = [a.detach().numpy() for a in arguments]
        for index, (gradient, direction) in enumerate(zip(gradients, directions)):
            plus, minus = [a.copy() for a in arrays], [a.copy() for a in arrays]
            plus[index] += 1e-6 * direction.numpy()
            minus[index] -= 1e-6 * direction.numpy()
            def reference_loss(values):
                return np.sum(ambient_squared(values[0], values[1], scale=1.3) * values[2][:, None, :])
            numerical = (reference_loss(plus) - reference_loss(minus)) / 2e-6
            torch.testing.assert_close((gradient * direction).sum(), torch.tensor(numerical),
                                       atol=1e-6, rtol=1e-5)

    def test_fixed_metric_has_no_trainable_gate_route(self):
        torch.manual_seed(253)
        model = ExpertAttention().double()
        tokens = torch.randn(2, 2, 32, dtype=torch.double, requires_grad=True)
        weights = model(tokens, torch.ones(2, 2, dtype=torch.bool))[0]
        x = torch.tensor([[.3, -.2], [-.1, .4]], dtype=torch.double, requires_grad=True)
        y = torch.tensor([[-.4, .1], [.2, .6]], dtype=torch.double, requires_grad=True)
        expected = torch.from_numpy(ambient_squared(x.detach().numpy(), y.detach().numpy(), scale=1.3)).mean(-1)
        fixed = geometry.fixed_squared(x, y, scale=1.3)
        torch.testing.assert_close(fixed, expected, atol=1e-9, rtol=1e-8)
        gate_inputs = (tokens,) + tuple(model.parameters())
        gradients = torch.autograd.grad(fixed.sum(), gate_inputs, allow_unused=True, retain_graph=True)
        self.assertTrue(all(g is None for g in gradients), "fixed supervision cannot train gate by a bypass")
        mixed = geometry.mixed_cost(x, y, weights, scale=1.3)
        routed = torch.autograd.grad(mixed.sum(), (model.query, model.key.weight, model.value.weight,
                                                  model.score[0].weight, model.score[2].weight, tokens))
        self.assertTrue(all(torch.isfinite(g).all() and g.norm() > 0 for g in routed))

    def test_invalid_common_scale_and_nonfinite_norms_fail_closed(self):
        x = torch.tensor([[.2, -.3]], dtype=torch.double)
        for scale in (0., -1., float("nan"), float("inf")):
            for function in (geometry.expert_squared, geometry.fixed_squared):
                with self.subTest(scale=scale, function=function.__name__), self.assertRaises(ValueError):
                    function(x, x, scale=scale)
            with self.assertRaises(ValueError):
                geometry.mixed_cost(x, x, torch.full((1, 3), 1/3, dtype=x.dtype), scale=scale)
        for bad in (float("nan"), float("inf")):
            broken = x.clone(); broken[0, 0] = bad
            with self.assertRaises(ValueError):
                geometry.expert_squared(broken, x)
            with self.assertRaises(ValueError):
                geometry.fixed_squared(x, broken)


class ThreeAttentionTests(unittest.TestCase):
    def test_scalar_reference_for_two_tokens_and_all_masks(self):
        torch.manual_seed(253)
        for dtype in (torch.float32, torch.float64):
            model = ExpertAttention().to(dtype=dtype)
            tokens = torch.randn(4, 2, 32, dtype=dtype)
            mask = torch.tensor([[True, True], [True, False], [False, True], [False, False]])
            tokens[~mask] = float("nan")
            weights, attention, effective = model(tokens, mask)
            expected_weights, expected_attention = reference_attention(model, tokens, mask)
            atol, rtol = (1e-5, 1e-4) if dtype == torch.float32 else (1e-9, 1e-8)
            torch.testing.assert_close(weights, torch.tensor(expected_weights, dtype=dtype), atol=atol, rtol=rtol)
            torch.testing.assert_close(attention, torch.tensor(expected_attention, dtype=dtype), atol=atol, rtol=rtol)
            self.assertEqual(weights.shape, (4, 3))
            self.assertEqual(attention.shape, (4, 3, 3))
            self.assertEqual(effective.tolist(), [[True, True, False], [True, False, False],
                                                [False, True, False], [False, False, True]])
            self.assertTrue(torch.equal(attention.masked_select(~effective[:, None]),
                                        torch.zeros(21, dtype=dtype)))
            torch.testing.assert_close(weights.sum(-1), torch.ones(4, dtype=dtype))
            torch.testing.assert_close(attention.sum(-1), torch.ones(4, 3, dtype=dtype))

    def test_query_order_is_not_a_hardcoded_topological_role(self):
        torch.manual_seed(253)
        model = ExpertAttention().double()
        tokens = torch.randn(2, 2, 32, dtype=torch.double)
        mask = torch.ones(2, 2, dtype=torch.bool)
        weights, attention, _ = model(tokens, mask)
        permutation = torch.tensor([2, 0, 1])
        with torch.no_grad():
            model.query.copy_(model.query[permutation])
        other_weights, other_attention, _ = model(tokens, mask)
        torch.testing.assert_close(other_weights, weights[:, permutation], atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(other_attention, attention[:, permutation], atol=1e-12, rtol=1e-12)

    def test_three_query_serialization_and_shared_parameter_capacity(self):
        torch.manual_seed(253)
        model = ExpertAttention().double()
        self.assertEqual(tuple(model.query.shape), (3, 32))
        parameters = dict(model.named_parameters())
        self.assertEqual(set(parameters), {"query", "key.weight", "key.bias", "value.weight", "value.bias",
                                           "score.0.weight", "score.0.bias", "score.2.weight", "score.2.bias"})
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = ExpertAttention().double()
        restored.load_state_dict(torch.load(buffer, map_location="cpu"), strict=True)
        self.assertEqual(tuple(restored.state_dict()["query"].shape), (3, 32))
        tokens = torch.randn(2, 2, 32, dtype=torch.double)
        mask = torch.tensor([[True, True], [False, False]])
        for before, after in zip(model(tokens, mask), restored(tokens, mask)):
            self.assertTrue(torch.equal(before, after))

    def test_behavior_off_is_exact_uniform_three_for_repeats_and_empty(self):
        from model.structure_gate import TopologyAttention
        class Tokens(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.values = torch.nn.Parameter(torch.arange(64., dtype=torch.double).reshape(1, 2, 32) / 64)
            def tokens_for_items(self, store, ids, device, chunk=1024):
                return self.values.expand(len(ids), -1, -1), torch.ones(len(ids), 2, dtype=torch.bool)
        model = TopologyAttention().double()
        model.tokens = Tokens()
        for ids in ([3, 1, 3, 0], torch.tensor([3, 1, 3, 0]), []):
            _, weights = model.gates_for_items(None, ids, "cpu", behavior_on=False)
            self.assertEqual(tuple(weights.shape), (len(ids), 3))
            self.assertTrue(torch.equal(weights, torch.full((len(ids), 3), 1/3, dtype=torch.double)))
            self.assertFalse(weights.requires_grad)
            self.assertEqual(model.last_attention.shape, (len(ids), 3, 3))
            self.assertEqual(model.last_token_mask.shape, (len(ids), 3))
        _, live = model.gates_for_items(None, [3, 1, 3, 0], "cpu", behavior_on=True)
        self.assertTrue(live.requires_grad)
        self.assertTrue(torch.equal(live[0], live[2]))


if __name__ == "__main__":
    unittest.main()
