"""Trainer-level tests for additive quantized-ranking arm wiring."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

STAGE2 = Path(__file__).resolve().parent
if str(STAGE2) not in sys.path:
    sys.path.insert(0, str(STAGE2))

import train_rqvae as trainer
import quant_rank_config
from model import RQVAE


class QuantizedWiringTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = RQVAE(trainer._tokenizer_config(), in_dim=768)
        self.inputs = torch.randn(9, 768)
        self.source = torch.arange(3)
        self.positive = torch.arange(3, 6)
        self.negative = torch.arange(6, 9)

    def test_t1_new_arms_resolve_frozen_configuration_and_only_add_term(self):
        for quant_arm, group_a in (("B_E", "E0"), ("B_H", "H1")):
            added = trainer._resolve_quant_arm(quant_arm)
            baseline = trainer._resolve_geometry_arm(group_a)
            self.assertEqual(added["shared"], baseline["shared"])
            self.assertEqual(added["geometry"], baseline["geometry"])
            self.assertEqual(added["group_a_geometry_arm"], group_a)
            self.assertEqual(added["level_weights"], (1.0, 1.0, 1.0))
            self.assertTrue(added["quantized_ranking"])
        self.assertEqual(quant_rank_config.GROUP_CONFIGS["B"]["level_weights"], (1.0, 1.0, 1.0))

    def test_t2_zero_weight_composition_is_behavior_bit_identical(self):
        behavior = torch.tensor(0.1234567, requires_grad=True)
        zero = behavior.new_zeros(())
        self.assertTrue(torch.equal(trainer._compose_quantized_loss(behavior, zero), behavior))

    def test_t3_quantized_term_has_encoder_and_each_codebook_gradient(self):
        residuals = trainer._residuals_with_trace(self.model, self.inputs)
        tokens = self.model.get_indices(self.inputs).transpose(0, 1).contiguous()
        term = trainer._quantized_ranking_loss(
            residuals, tokens, self.model,
            self.source, self.positive, self.negative,
        )
        term.backward()
        enc_grads = tuple(parameter.grad for parameter in self.model.encoder.parameters())
        code_grads = tuple(
            layer.get_code_embs().grad for layer in self.model.rq.vq_layers
        )
        self.assertTrue(any(g is not None and torch.count_nonzero(g) for g in enc_grads))
        self.assertEqual(len(code_grads), 3)
        self.assertTrue(all(g is not None and torch.count_nonzero(g) for g in code_grads))

    def test_t4_loss_does_not_change_real_rq_token_ids(self):
        residuals = trainer._residuals_with_trace(self.model, self.inputs)
        before = self.model.get_indices(self.inputs).transpose(0, 1).contiguous()
        trainer._quantized_ranking_loss(
            residuals, before, self.model,
            self.source, self.positive, self.negative,
        )
        after = self.model.get_indices(self.inputs).transpose(0, 1).contiguous()
        self.assertTrue(torch.equal(before, after))

    def test_t5_new_output_directories_are_distinct_from_group_a(self):
        arms = ("B_E", "B_H", "E0", "H1")
        paths = [trainer._ARM_DIR[arm] for arm in arms]
        self.assertEqual(len(set(paths)), 4)

    def test_t6_determinism_at_seed_42(self):
        def run():
            torch.manual_seed(42)
            model = RQVAE(trainer._tokenizer_config(), in_dim=768)
            inputs = torch.randn(9, 768)
            residuals = trainer._residuals_with_trace(model, inputs)
            tokens = model.get_indices(inputs).transpose(0, 1).contiguous()
            return trainer._quantized_ranking_loss(
                residuals, tokens, model,
                self.source, self.positive, self.negative,
            ).detach()
        self.assertTrue(torch.equal(run(), run()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
