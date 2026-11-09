"""Regression oracles for Issue 139's discrete weights and VQ gradient paths.

These are small CPU unit tests, not a replacement for the frozen-data trainer
or four-GPU admission.  The Euclidean gradient oracle is written analytically;
the curved oracle uses the independent arcosh distance, not the candidate's
Mobius-distance implementation.
"""
from __future__ import annotations

import copy
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

TREE = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "model" / "mixed_rq.py").is_file())
sys.path.insert(0, str(TREE))

from model.mixed_rq import (  # noqa: E402
    AlphaTable, MixedRQ, MixedVQLevel, encode_tokens, mixed_cost, point_cost,
)
from model.model import MixedRQVAE  # noqa: E402


def _tradeoff_quantizer(level: int) -> MixedRQ:
    """Earlier levels select a zero code; only the requested level disagrees.

    There the exact Euclidean half costs are [1, 4], the other half [4, 1].
    The low weight must select code 1 and the high weight code 0, without ties.
    """
    rq = MixedRQ([2, 2, 2], AlphaTable(), curvature=0.0).double().eval()
    with torch.no_grad():
        for index, layer in enumerate(rq.levels, 1):
            layer.embed.weight.zero_()
            if index < level:
                layer.embed.weight[1].fill_(10.0)
            elif index == level:
                layer.embed.weight[0, 0] = 1.0
                layer.embed.weight[1, 0] = 2.0
                layer.embed.weight[0, 16] = 2.0
                layer.embed.weight[1, 16] = 1.0
    return rq


def _model() -> MixedRQVAE:
    config = SimpleNamespace(hidden_sizes=(), codebook_dim=32, dropout=0.0,
                             codebook_size=[2, 2, 2], codebook_num=3, beta=0.25,
                             sk_epsilon=-1.0, sk_iters=-1)
    model = MixedRQVAE(config, in_dim=32, curvature=0.0).double().eval()
    with torch.no_grad():
        linear = next(layer for layer in model.encoder.modules()
                      if isinstance(layer, torch.nn.Linear))
        linear.weight.copy_(torch.eye(32, dtype=torch.float64))
        linear.bias.zero_()
        source = _tradeoff_quantizer(2)
        for target, origin in zip(model.rq.levels, source.levels):
            target.embed.weight.copy_(origin.embed.weight)
    return model


class AlphaInheritanceTests(unittest.TestCase):
    def test_unseen_prefixes_inherit_the_nearest_ancestor_level(self):
        table = AlphaTable()
        table.set(1, (), 0.75)
        self.assertEqual(table.get(2, (7,)), 0.75)
        self.assertEqual(table.get(3, (7, 9)), 0.75)
        table.set(2, (7,), 0.25)
        self.assertEqual(table.get(3, (7, 9)), 0.25)
        self.assertEqual(table.get(3, (8, 9)), 0.75)
        table.set(3, (7, 9), 0.5)
        self.assertEqual(table.get(3, (7, 9)), 0.5)

    def test_global_overrides_change_l2_and_l3_assignments(self):
        z = torch.zeros(3, 32, dtype=torch.float64)
        for level in (2, 3):
            with self.subTest(level=level):
                rq = _tradeoff_quantizer(level)
                low = encode_tokens(z, rq, override={(level, ()): 0.25})
                high = encode_tokens(z, rq, override={(level, ()): 0.75})
                self.assertTrue(torch.equal(low[:, level - 1], torch.ones(3, dtype=torch.long)))
                self.assertTrue(torch.equal(high[:, level - 1], torch.zeros(3, dtype=torch.long)))
                self.assertEqual(rq.alpha_table.to_dict(), AlphaTable().to_dict(),
                                 "candidate evaluation must not mutate the incumbent table")

    def test_trial_override_equals_committed_table_including_descendants(self):
        z = torch.zeros(2, 32, dtype=torch.float64)
        cases = ((2, (1, ())), (3, (2, (0,))), (2, (2, ())), (3, (3, ())))
        for tradeoff_level, key in cases:
            with self.subTest(tradeoff_level=tradeoff_level, key=key):
                rq = _tradeoff_quantizer(tradeoff_level)
                trial = encode_tokens(z, rq, override={key: 0.25})
                rq.alpha_table.set(key[0], key[1], 0.25)
                committed = encode_tokens(z, rq)
                self.assertTrue(torch.equal(trial, committed),
                                "trial and accepted weight must generate identical full SIDs")
                self.assertTrue(torch.equal(committed[:, tradeoff_level - 1],
                                            torch.ones(2, dtype=torch.long)),
                                "an inherited low weight must prefer the second code")

    def test_alpha_payload_roundtrip_retains_local_and_global_entries(self):
        table = AlphaTable()
        table.set(1, (), 0.75)
        table.set(2, (), 0.25)
        table.set(2, (4,), 0.5)
        table.set(3, (4, 9), 0.75)
        loaded = AlphaTable.from_dict(copy.deepcopy(table.to_dict()))
        self.assertEqual(loaded.to_dict(), table.to_dict())
        self.assertEqual(loaded.get(3, (4, 8)), 0.5)
        self.assertEqual(loaded.get(2, (5,)), 0.25)


class QuantizationLossTests(unittest.TestCase):
    @staticmethod
    def _euclidean_half_fixture():
        layer = MixedVQLevel(1, curvature=1.0, beta=0.25).double().eval()
        with torch.no_grad():
            layer.embed.weight.zero_()
            layer.embed.weight[0, :16] = torch.linspace(-0.1, 0.1, 16, dtype=torch.float64)
        r_e = torch.linspace(-0.3, 0.5, 48, dtype=torch.float64).view(3, 16).requires_grad_()
        r_h = torch.zeros(3, 16, dtype=torch.float64, requires_grad=True)
        alpha = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
        return layer, r_e, r_h, alpha

    def test_beta_weights_the_encoder_commitment_gradient(self):
        layer, r_e, r_h, alpha = self._euclidean_half_fixture()
        _, loss, _, _, _, _ = layer(r_e, r_h, alpha)
        actual = torch.autograd.grad(loss, r_e)[0]
        difference = r_e.detach() - layer.embed.weight.detach()[:, :16]
        # d[2 alpha ||r-e||^2]/dr = 4 alpha (r-e); only beta*commitment reaches r.
        expected = 0.25 * 4.0 * alpha[:, None] * difference / (3 * 32)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_commitment_does_not_add_a_second_codebook_gradient(self):
        layer, r_e, r_h, alpha = self._euclidean_half_fixture()
        _, loss, _, _, _, _ = layer(r_e, r_h, alpha)
        actual = torch.autograd.grad(loss, layer.embed.weight)[0]
        difference = r_e.detach() - layer.embed.weight.detach()[:, :16]
        expected = torch.zeros_like(layer.embed.weight)
        # No beta multiplies the codebook term; commitment is stopped at e.
        expected[0, :16] = (-4.0 * alpha[:, None] * difference).sum(0) / (3 * 32)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_zero_curvature_all_cost_paths_use_the_same_euclidean_units(self):
        generator = torch.Generator().manual_seed(280)
        residual = torch.randn(4, 32, dtype=torch.float64, generator=generator)
        code = torch.randn(1, 32, dtype=torch.float64, generator=generator)
        for weight in (0.25, 0.5, 0.75):
            with self.subTest(alpha=weight):
                alpha = torch.full((4,), weight, dtype=torch.float64)
                layer = MixedVQLevel(1, curvature=0.0, beta=0.25).double().eval()
                with torch.no_grad():
                    layer.embed.weight.copy_(code)
                r_e, r_h = residual[:, :16], residual[:, 16:]
                expected = 2 * (weight * (r_e - code[:, :16]).square().sum(-1)
                                + (1 - weight) * (r_h - code[:, 16:]).square().sum(-1))
                matrix = mixed_cost(r_e, r_h, code[:, :16], code[:, 16:], alpha, 0.0)
                scalar = point_cost(r_e, r_h, code[:, :16], code[:, 16:], alpha, 0.0)
                _, loss, _, _, _, _ = layer(r_e, r_h, alpha)
                torch.testing.assert_close(matrix[:, 0], expected, rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(scalar, expected, rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(loss, (1 + 0.25) * expected.mean() / 32,
                                           rtol=1e-12, atol=1e-12)

    def test_zero_curvature_training_loss_equals_scaled_joint_squared_error(self):
        layer = MixedVQLevel(1, curvature=0.0, beta=0.25).double().eval()
        with torch.no_grad():
            layer.embed.weight.zero_()
        residual = torch.zeros(1, 32, dtype=torch.float64)
        residual[0, 0] = 1.0
        residual[0, 16] = 1.0
        _, loss, _, _, _, _ = layer(residual[:, :16], residual[:, 16:],
                                     torch.tensor([0.5], dtype=torch.float64))
        self.assertAlmostEqual(float(loss.detach()), 1.25 * 2.0 / 32, places=14)

    def test_one_sgd_step_uses_the_two_distinct_gradient_paths(self):
        layer, r_e, r_h, alpha = self._euclidean_half_fixture()
        start_e = r_e.detach().clone()
        start_code = layer.embed.weight.detach().clone()
        difference = start_e - start_code[:, :16]
        expected_e = start_e - 0.2 * 0.25 * 4 * alpha[:, None] * difference / (3 * 32)
        expected_code = start_code.clone()
        expected_code[0, :16] += 0.2 * (4 * alpha[:, None] * difference).sum(0) / (3 * 32)
        optimizer = torch.optim.SGD([r_e, r_h, layer.embed.weight], lr=0.2)
        _, loss, _, _, _, _ = layer(r_e, r_h, alpha)
        loss.backward()
        optimizer.step()
        torch.testing.assert_close(r_e, expected_e, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(layer.embed.weight, expected_code, rtol=1e-12, atol=1e-12)

    def test_curved_loss_value_and_both_gradients_match_an_independent_oracle(self):
        generator = torch.Generator().manual_seed(282)
        layer = MixedVQLevel(1, curvature=1.0, beta=0.25).double().eval()
        with torch.no_grad():
            layer.embed.weight.copy_(0.02 * torch.randn(1, 32, generator=generator,
                                                        dtype=torch.float64))
        r_e = (0.04 * torch.randn(3, 16, generator=generator, dtype=torch.float64)).requires_grad_()
        r_h = (0.03 * torch.randn(3, 16, generator=generator, dtype=torch.float64)).requires_grad_()
        alpha = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
        _, actual, _, _, _, _ = layer(r_e, r_h, alpha)
        actual_grad = torch.autograd.grad(actual, (r_e, r_h, layer.embed.weight))

        oracle_e = r_e.detach().clone().requires_grad_()
        oracle_h = r_h.detach().clone().requires_grad_()
        oracle_code = layer.embed.weight.detach().clone().requires_grad_()

        def cost(e, h, code):
            tangent = code[:, 16:]
            radius = tangent.square().sum(-1, keepdim=True).sqrt()
            ball = torch.tanh(radius) * tangent / radius
            argument = 1 + 2 * (h - ball).square().sum(-1) / (
                (1 - h.square().sum(-1)) * (1 - ball.square().sum(-1)))
            curved = torch.acosh(argument).square()
            flat = (e - code[:, :16]).square().sum(-1)
            return 2 * (alpha * flat + (1 - alpha) * curved / 4)

        expected = (cost(oracle_e.detach(), oracle_h.detach(), oracle_code)
                    + 0.25 * cost(oracle_e, oracle_h, oracle_code.detach())).mean() / 32
        expected_grad = torch.autograd.grad(expected, (oracle_e, oracle_h, oracle_code))
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
        for actual_part, expected_part in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual_part, expected_part, rtol=1e-9, atol=1e-11)

    def test_next_residual_keeps_encoder_gradient_but_detaches_selected_code(self):
        layer = MixedVQLevel(1, curvature=0.0).double().eval()
        r_e = torch.ones(2, 16, dtype=torch.float64, requires_grad=True)
        r_h = torch.full((2, 16), 0.5, dtype=torch.float64, requires_grad=True)
        (next_e, next_h), _, _, _, _, _ = layer(r_e, r_h, torch.full((2,), 0.5))
        derivatives = torch.autograd.grad(next_e.sum() + next_h.sum(),
                                          (r_e, r_h, layer.embed.weight), allow_unused=True)
        torch.testing.assert_close(derivatives[0], torch.ones_like(r_e))
        torch.testing.assert_close(derivatives[1], torch.ones_like(r_h))
        self.assertIsNone(derivatives[2])


class CheckpointTests(unittest.TestCase):
    def test_state_dict_roundtrip_retains_alpha_and_exported_sids(self):
        original = _model()
        original.rq.alpha_table.set(1, (), 0.75)
        original.rq.alpha_table.set(2, (0,), 0.25)
        original.rq.alpha_table.set(3, (0, 1), 0.75)
        data = torch.zeros(5, 32, dtype=torch.float64)
        before = original.get_indices(data)
        memory = io.BytesIO()
        torch.save(original.state_dict(), memory)
        memory.seek(0)
        restored = _model()
        restored.load_state_dict(torch.load(memory, weights_only=True))
        self.assertEqual(restored.rq.alpha_table.to_dict(), original.rq.alpha_table.to_dict())
        self.assertTrue(torch.equal(restored.get_indices(data), before),
                        "a checkpoint reload changed the global item IDs")

    def test_wrapper_table_reference_tracks_a_broadcast_replacement(self):
        model = _model()
        replacement = AlphaTable()
        replacement.set(2, (), 0.25)
        model.rq.alpha_table = replacement
        self.assertIs(model.alpha_table, model.rq.alpha_table,
                      "wrapper must not retain the stale pre-broadcast table")


class InitializationReplayTests(unittest.TestCase):
    def test_rank_zero_centers_are_replayed_before_the_next_residual(self):
        """Unit-level collective replay only; the Gloo integration is separate."""
        reference = MixedVQLevel(2, curvature=0.0)
        replica = MixedVQLevel(2, curvature=0.0)
        r_e = torch.arange(64, dtype=torch.float32).view(4, 16) * 0.01
        r_h = r_e + 0.02
        broadcasts = []

        def record(tensor, src, *args, **kwargs):
            self.assertEqual(src, 0)
            broadcasts.append(tensor.detach().clone())

        with patch("torch.distributed.is_initialized", return_value=True), \
                patch("torch.distributed.get_rank", return_value=0), \
                patch("torch.distributed.broadcast", side_effect=record), \
                patch("model.utils.kmeans", side_effect=lambda samples, count: samples[:count]):
            reference.init_codebook(r_e, r_h, torch.device("cpu"))
        self.assertTrue(broadcasts, "codebook initialization never broadcasts rank-0 centers")
        pending = list(broadcasts)

        def replay(tensor, src, *args, **kwargs):
            self.assertEqual(src, 0)
            self.assertTrue(pending, "ranks entered a different number of broadcasts")
            tensor.copy_(pending.pop(0))

        with patch("torch.distributed.is_initialized", return_value=True), \
                patch("torch.distributed.get_rank", return_value=1), \
                patch("torch.distributed.broadcast", side_effect=replay), \
                patch("model.utils.kmeans", side_effect=lambda samples, count: samples[:count] + 1):
            replica.init_codebook(r_e, r_h, torch.device("cpu"))
        self.assertFalse(pending, "not every rank-0 broadcast was consumed")
        torch.testing.assert_close(replica.embed.weight, reference.embed.weight, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
