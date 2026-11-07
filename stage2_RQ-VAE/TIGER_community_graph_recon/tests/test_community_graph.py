"""What the community-guided graph reconstruction must satisfy.

The mechanism adds one loss to the frozen baseline: the quantised
representation of an item must predict the distribution of its behaviour
neighbours, weighted by the community model's own geometry. These tests are
the mechanism's contract, and they were written before the implementation so
that they failed for the right reason.

The four properties that matter:

1. **The edge weight is bounded and it never invents an edge.** A real
   behaviour edge keeps a weight in ``[1, 2]``, a non-edge has weight zero,
   and the community's contribution to the weight is the affinity alone.
2. **The target is the behaviour neighbourhood, renormalised inside the
   candidate set.** A constant weight over the neighbours gives the uniform
   target; a varying weight gives a different one, and the difference is the
   community's whole contribution.
3. **Both gradient paths exist.** After one backward of the term alone, the
   encoder's parameters and the codebook's parameters have both received a
   non-zero gradient: the straight-through path reaches the encoder and the
   un-detached codeword path reaches the codebook.
4. **The gradient budget is enforced.** The term's contribution to the
   encoder's and the codebook's gradients never exceeds the configured share
   of the original objective's, and the stricter of the two scalings wins.

Run:
    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
        tests/test_community_graph.py
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import RQVAE  # noqa: E402
from model.community_graph import (  # noqa: E402
    CandidateSet,
    GradientBudget,
    affinity_weights,
    behaviour_edges,
    community_graph_loss,
    neighbour_target,
    quantised_views,
)


class Config:
    hidden_sizes = (16, 16)
    codebook_num = 3
    codebook_size = (8, 8, 8)
    codebook_dim = 4
    vq_type = "vq"
    beta = 0.25
    sk_epsilon = 0.003
    sk_iters = 5
    ema_decay = 0.99
    fix_code_embs = False
    dropout = 0.0


def _model() -> RQVAE:
    torch.manual_seed(0)
    return RQVAE(Config(), in_dim=6)


class EdgeWeightTest(unittest.TestCase):
    """The community's weight on an observed behaviour edge."""

    def test_a_real_edge_keeps_a_weight_between_one_and_two(self):
        free = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        ball = torch.tensor([[0.0, 0.0], [0.1, 0.0]])
        edges = torch.tensor([[0], [1]])
        weights = affinity_weights(free, ball, edges, curvature=1.0)
        self.assertEqual(tuple(weights.shape), (1,))
        self.assertTrue(1.0 <= float(weights[0]) <= 2.0)

    def test_a_non_edge_has_weight_zero(self):
        free = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        ball = torch.tensor([[0.0, 0.0], [0.1, 0.0]])
        edges = torch.tensor([[0], [2]])          # 2 is not an item
        weights = affinity_weights(free, ball, edges, curvature=1.0)
        self.assertEqual(float(weights[0]), 0.0)

    def test_the_affinity_never_exceeds_one(self):
        """s_ij is an exponential of a non-positive number."""
        free = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        ball = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        edges = torch.tensor([[0], [1]])
        weights = affinity_weights(free, ball, edges, curvature=1.0)
        self.assertTrue(1.0 <= float(weights[0]) <= 2.0)

    def test_a_closer_pair_gets_a_larger_weight(self):
        """The community geometry decides how much an observed edge counts."""
        free = torch.tensor([[0.0, 0.0], [0.1, 0.0], [3.0, 0.0]])
        ball = torch.tensor([[0.0, 0.0], [0.1, 0.0], [3.0, 0.0]])
        edges = torch.tensor([[0, 0], [1, 2]])
        weights = affinity_weights(free, ball, edges, curvature=1.0)
        self.assertGreater(float(weights[0]), float(weights[1]))


class BehaviourGraphTest(unittest.TestCase):
    """The graph is the training split's own, and it is undirected."""

    def test_the_edges_come_from_the_sequences(self):
        edges = behaviour_edges(
            [[1, 2, 3], [3, 4]],
            n_items=6,
        )
        pairs = {(int(a), int(b)) for a, b in zip(edges[0], edges[1])}
        self.assertIn((1, 2), pairs)
        self.assertIn((2, 3), pairs)
        self.assertIn((3, 4), pairs)
        for a, b in pairs:
            self.assertLess(a, b)

    def test_the_graph_reads_only_the_sequences_it_is_handed(self):
        """The graph has no other source: the trainer builds the sequences from
        the train split alone, and this function sees nothing else."""
        import inspect

        signature = inspect.signature(behaviour_edges)
        self.assertEqual(list(signature.parameters), ["sequences", "n_items"])


class TargetTest(unittest.TestCase):
    """The target over the candidate set."""

    def test_a_constant_weight_gives_the_uniform_target(self):
        weights = torch.tensor([2.0, 2.0, 2.0])
        target = neighbour_target(weights)
        self.assertTrue(torch.allclose(target, torch.full((3,), 1.0 / 3.0), atol=1e-6))

    def test_a_varying_weight_moves_the_target(self):
        """This is the community's whole contribution to the term."""
        weights = torch.tensor([3.0, 1.0])
        target = neighbour_target(weights)
        self.assertGreater(float(target[0]), float(target[1]))
        self.assertLess(abs(float(target.sum()) - 1.0), 1e-6)

    def test_a_zero_weight_carries_no_target_mass(self):
        weights = torch.tensor([0.0, 2.0, 2.0])
        target = neighbour_target(weights)
        self.assertEqual(float(target[0]), 0.0)
        self.assertLess(abs(float(target.sum()) - 1.0), 1e-6)


class GradientPathTest(unittest.TestCase):
    """The two paths the term must have."""

    def _candidates(self) -> CandidateSet:
        return CandidateSet(
            anchors=torch.tensor([0, 1]),
            candidate_rows=torch.tensor([[0, 1, 2], [1, 0, 2]]),
            candidate_weights=torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
            candidate_mask=torch.tensor([[True, True, True], [True, True, True]]),
        )

    def test_the_term_reaches_the_encoder_and_the_codebook(self):
        model = _model()
        batch = torch.randn(2, 6)
        candidate_batch = torch.randn(3, 6)
        views = quantised_views(model, batch)
        candidate_views = quantised_views(model, candidate_batch)
        loss = community_graph_loss(views[0], views[1],
                                    candidate_views[0], candidate_views[1],
                                    self._candidates(), temperature=0.2)
        loss.backward()

        encoder_grad = sum(
            float(p.grad.abs().sum()) for p in model.encoder.parameters() if p.grad is not None)
        codebook_grad = sum(
            float(p.grad.abs().sum()) for p in model.rq.parameters() if p.grad is not None)
        self.assertGreater(encoder_grad, 0.0,
                           "the straight-through path did not reach the encoder")
        self.assertGreater(codebook_grad, 0.0,
                           "the un-detached codeword path did not reach the codebook")

    def test_the_two_views_carry_the_same_value_and_different_gradients(self):
        model = _model()
        batch = torch.randn(3, 6)
        ste, cb = quantised_views(model, batch)
        self.assertTrue(torch.allclose(ste, cb, atol=1e-6))

        ste.sum().backward(retain_graph=True)
        ste_encoder = sum(
            float(p.grad.abs().sum()) for p in model.encoder.parameters() if p.grad is not None)
        ste_codebook = sum(
            float(p.grad.abs().sum()) for p in model.rq.parameters() if p.grad is not None)
        model.zero_grad(set_to_none=True)

        cb.sum().backward()
        cb_encoder = sum(
            float(p.grad.abs().sum()) for p in model.encoder.parameters() if p.grad is not None)
        cb_codebook = sum(
            float(p.grad.abs().sum()) for p in model.rq.parameters() if p.grad is not None)
        model.zero_grad(set_to_none=True)

        self.assertTrue(ste_encoder > 0.0 and cb_encoder == 0.0,
                        "the straight-through view must reach the encoder and not the codebook")
        self.assertTrue(cb_codebook > 0.0 and ste_codebook == 0.0,
                        "the codeword view must reach the codebook and not the encoder")

    def test_the_quantised_view_is_the_sum_of_the_three_codewords(self):
        """q_i is the sum over the levels, not the level-1 code.

        The sum is checked against the model's own forward: the three codeword
        vectors the quantiser actually chose, added together.
        """
        model = _model()
        batch = torch.randn(2, 6)
        ste, codewords = quantised_views(model, batch)
        with torch.no_grad():
            encoded = model.encoder(batch)
            residual = encoded
            total = torch.zeros_like(encoded)
            for layer in model.rq.vq_layers:
                distances = layer._distances(residual)
                indices = layer._indices(distances, infer_use_sk=False)
                code = layer.get_code_embs()[indices]
                total = total + code
                residual = residual - code
        self.assertTrue(torch.allclose(codewords, total, atol=1e-5))
        self.assertTrue(torch.allclose(ste, total, atol=1e-5))


class MaskedCandidateTest(unittest.TestCase):
    """A padded position must not poison the loss."""

    def test_a_masked_position_does_not_produce_nan(self):
        """The target has no mass there, but its log-probability is -inf, and
        0 * -inf is NaN: the padded positions must be zeroed explicitly."""
        model = _model()
        anchor = torch.randn(1, 6)
        candidate = torch.randn(3, 6)
        views = quantised_views(model, anchor)
        candidate_views = quantised_views(model, candidate)
        candidates = CandidateSet(
            anchors=torch.tensor([0]),
            candidate_rows=torch.tensor([[0, 1, 2]]),
            candidate_weights=torch.tensor([[1.0, 0.0, 0.0]]),
            candidate_mask=torch.tensor([[True, False, False]]),
        )
        loss = community_graph_loss(views[0], views[1],
                                    candidate_views[0], candidate_views[1],
                                    candidates, temperature=0.2)
        self.assertTrue(bool(torch.isfinite(loss)), "a padded position poisoned the loss")

    def test_an_anchor_with_every_position_masked_is_finite(self):
        model = _model()
        anchor = torch.randn(1, 6)
        candidate = torch.randn(2, 6)
        views = quantised_views(model, anchor)
        candidate_views = quantised_views(model, candidate)
        candidates = CandidateSet(
            anchors=torch.tensor([0]),
            candidate_rows=torch.tensor([[0, 1]]),
            candidate_weights=torch.tensor([[0.0, 0.0]]),
            candidate_mask=torch.tensor([[False, False]]),
        )
        loss = community_graph_loss(views[0], views[1],
                                    candidate_views[0], candidate_views[1],
                                    candidates, temperature=0.2)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(float(loss), 0.0)


class GradientBudgetTest(unittest.TestCase):
    """The term's share of the gradient."""

    def test_the_budget_caps_the_new_term_at_its_share(self):
        budget = GradientBudget(share=0.10)
        budget.observe(base_encoder=1.0, new_encoder=100.0,
                       base_codebook=1.0, new_codebook=100.0)
        self.assertLess(budget.scale, 1.0)

    def test_the_budget_leaves_a_small_term_alone(self):
        budget = GradientBudget(share=0.10)
        budget.observe(base_encoder=1.0, new_encoder=0.01,
                       base_codebook=1.0, new_codebook=0.01)
        self.assertEqual(budget.scale, 1.0)

    def test_the_budget_takes_the_stricter_of_the_two_groups(self):
        """The encoder is fine with the term; the codebook is not. The codebook wins."""
        budget = GradientBudget(share=0.10)
        budget.observe(base_encoder=1.0, new_encoder=0.001,
                       base_codebook=1.0, new_codebook=50.0)
        strict = budget.scale
        budget.observe(base_encoder=1.0, new_encoder=50.0,
                       base_codebook=1.0, new_codebook=0.001)
        self.assertEqual(budget.scale, strict)

    def test_the_budget_scale_is_finite_when_a_group_has_no_gradient(self):
        budget = GradientBudget(share=0.10)
        budget.observe(base_encoder=1.0, new_encoder=0.0,
                       base_codebook=0.0, new_codebook=0.0)
        self.assertTrue(math.isfinite(budget.scale))
        self.assertEqual(budget.scale, 1.0)


if __name__ == "__main__":
    unittest.main()
