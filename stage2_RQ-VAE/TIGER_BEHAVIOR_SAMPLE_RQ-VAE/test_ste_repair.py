"""The STE repair: what it changes, and just as importantly what it does not.

``VQLayer.forward`` returns ``x + (x_q - x).detach()``. Chaining that through
``residual = residual - quant`` gives level 2 the value ``-(x_q - residual).detach()``
-- the right residual, with the encoder cut off. Every behaviour-supervision level past
the first therefore trained nothing at all: measured on the baseline checkpoint, L1 put
10.22 of gradient norm on the encoder, L2 and L3 put exactly zero.

The repair rebuilds the chain from the quantised VALUE instead of the straight-through
wrapper. Three properties have to hold and are pinned here:

  1. the forward values are bit-identical, so no loss, metric or codeword can move;
  2. L2 and L3 reach the encoder afterwards;
  3. neither before nor after does the behaviour term touch a codebook, which is what
     the plan requires of it.
"""

from __future__ import annotations

import unittest

import torch

import train_rqvae as trainer
from model import RQVAE


class STERepairTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = RQVAE(trainer._tokenizer_config(), in_dim=8)
        self.inputs = torch.randn(6, 8)

    def trace(self, carry_encoder_gradients: bool):
        return trainer._residuals_with_trace(
            self.model, self.inputs, carry_encoder_gradients=carry_encoder_gradients,
        )

    def norms(self, residual):
        """(encoder gradient norm, codebook gradient norm) of ``residual``."""
        encoder = tuple(self.model.encoder.parameters())
        codebooks = [layer.get_code_embs() for layer in self.model.rq.vq_layers]
        grads = torch.autograd.grad(
            residual, encoder + tuple(codebooks),
            grad_outputs=torch.ones_like(residual),
            retain_graph=True, allow_unused=True,
        )
        split = len(encoder)
        total = lambda values: sum(  # noqa: E731
            float(value.norm()) for value in values if value is not None
        )
        return total(grads[:split]), total(grads[split:])

    def test_t1_forward_values_are_bit_identical(self):
        broken = self.trace(False)
        repaired = self.trace(True)
        self.assertEqual(broken.shape, repaired.shape)
        self.assertTrue(torch.equal(broken, repaired))

    def test_t2_without_the_repair_only_the_first_level_reaches_the_encoder(self):
        residuals = self.trace(False)
        first, _ = self.norms(residuals[0])
        self.assertGreater(first, 0.0)
        for level in (1, 2):
            norm, _ = self.norms(residuals[level])
            self.assertEqual(norm, 0.0, "L%d already reached the encoder" % (level + 1))

    def test_t3_the_repair_connects_all_three_levels(self):
        residuals = self.trace(True)
        for level in (0, 1, 2):
            norm, _ = self.norms(residuals[level])
            self.assertGreater(norm, 0.0, "L%d cannot reach the encoder" % (level + 1))

    def test_t4_neither_path_lets_the_behaviour_term_reach_a_codebook(self):
        for carry in (False, True):
            residuals = self.trace(carry)
            for level in (0, 1, 2):
                _, codebook_norm = self.norms(residuals[level])
                self.assertEqual(codebook_norm, 0.0)

    def test_t5_the_repair_is_scoped_to_the_arm_that_asks_for_it(self):
        # The baseline must keep the original path, or its checkpoint would no longer
        # be reproducible from this source.
        self.assertNotIn("resample", trainer._STE_FIX_ARMS)
        self.assertIn("STE_FIX", trainer._STE_FIX_ARMS)


if __name__ == "__main__":
    unittest.main(verbosity=2)