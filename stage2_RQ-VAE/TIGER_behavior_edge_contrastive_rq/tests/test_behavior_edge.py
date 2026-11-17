"""Tests for B1: train-only behaviour pairs, matched negatives and contrastive gradient.

These are properties of the mechanism, not of the trainer wiring:
* an undirected train edge produces both orientations;
* a sampled negative is never a train neighbour and is degree-stratum matched;
* the contrastive loss sends gradient to anchor, positive and negatives;
* the model exposes the straight-through quantized latent used by the loss.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from behavior_pairs import build_behaviour_pairs, longest_sessions  # noqa: E402
from losses.behavior_contrastive import behaviour_contrastive_loss  # noqa: E402
from model import RQVAE  # noqa: E402


def _synthetic_edges():
    users = [0, 0, 1, 1, 2, 3]
    histories = [
        [0, 1],
        [0, 1, 2, 3],  # same user, longer session wins
        [1, 2],
        [1, 2, 3, 4],
        [3],
        [4],
    ]
    targets = [2, 4, 3, 0, 4, 0]
    return users, histories, targets


class BehaviourPairsTest(unittest.TestCase):
    def test_longest_session_per_user_and_both_orientations(self):
        users, histories, targets = _synthetic_edges()
        sessions = longest_sessions(users, histories, targets)
        self.assertEqual(sessions[0], [0, 1, 2, 3, 4])
        valid = np.arange(5, dtype=np.int64)
        embeddings = np.eye(5, dtype=np.float32)
        pairs = build_behaviour_pairs(
            users, histories, targets, valid, embeddings,
            negatives_per_positive=2, semantic_candidates=4,
            holdout_fraction=0.0, seed=7,
        )
        # The longest sessions add the 0-4 edge, so five distinct undirected
        # edges are expanded to both directions.
        self.assertEqual(pairs.positive_edges, 5)
        self.assertEqual(len(pairs.positives), 10)
        edge_set = {(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)}
        for anchor, positive in pairs.positives.tolist():
            self.assertIn((min(anchor, positive), max(anchor, positive)), edge_set)
        degrees = pairs.degree.tolist()
        for anchor, positive, row_negatives in zip(
            pairs.positives[:, 0], pairs.positives[:, 1], pairs.negatives
        ):
            self.assertEqual(len(set(row_negatives.tolist())), 2)
            for negative in row_negatives.tolist():
                self.assertNotEqual(anchor, negative)
                self.assertNotEqual(positive, negative)
                self.assertNotIn((min(anchor, negative), max(anchor, negative)),
                                 edge_set)
                self.assertEqual(degrees[anchor], degrees[negative])

    def test_holdout_edges_are_disjoint_and_never_used(self):
        valid = np.arange(20, dtype=np.int64)
        embeddings = np.eye(20, dtype=np.float32)
        users = [0]
        histories = [list(range(20))]
        targets = [0]
        pairs = build_behaviour_pairs(
            users, histories, targets, valid, embeddings,
            negatives_per_positive=2, semantic_candidates=8,
            holdout_fraction=0.4, seed=11,
        )
        fit = {(min(int(left), int(right)), max(int(left), int(right)))
               for left, right in pairs.positives.tolist()}
        held = {tuple(edge) for edge in pairs.holdout_edges.tolist()}
        self.assertTrue(fit.isdisjoint(held))
        self.assertEqual(len(fit | held), 20)
        # A held-out edge is still a true relation and must not be sampled as a
        # negative for its own endpoints.
        for anchor, row_negatives in zip(pairs.positives[:, 0], pairs.negatives):
            for negative in row_negatives.tolist():
                self.assertNotIn((min(anchor, negative), max(anchor, negative)), held)

    def test_contrastive_gradient_reaches_all_pair_sides(self):
        torch.manual_seed(0)
        anchor = torch.randn(4, 8, requires_grad=True)
        positive = torch.randn(4, 8, requires_grad=True)
        negatives = torch.randn(4, 3, 8, requires_grad=True)
        loss = behaviour_contrastive_loss(anchor, positive, negatives)
        loss.backward()
        self.assertIsNotNone(anchor.grad)
        self.assertIsNotNone(positive.grad)
        self.assertIsNotNone(negatives.grad)
        self.assertGreater(float(anchor.grad.abs().sum()), 0.0)
        self.assertGreater(float(positive.grad.abs().sum()), 0.0)
        self.assertGreater(float(negatives.grad.abs().sum()), 0.0)

    def test_contrastive_prefers_a_true_positive(self):
        anchor = torch.tensor([[1.0, 0.0]])
        negative = torch.tensor([[[0.0, 1.0]]])
        good = behaviour_contrastive_loss(anchor, torch.tensor([[0.9, 0.1]]), negative)
        bad = behaviour_contrastive_loss(anchor, torch.tensor([[-0.9, 0.1]]), negative)
        self.assertLess(float(good), float(bad))

    def test_model_can_return_quantized_latent(self):
        from types import SimpleNamespace
        config = SimpleNamespace(
            hidden_sizes=[8], codebook_num=2, codebook_size=(8, 8),
            codebook_dim=4, dropout=0.0, beta=0.25, vq_type="vq",
            ema_decay=0.99, sk_epsilon=-1.0, sk_iters=-1, fix_code_embs=False,
        )
        model = RQVAE(config, in_dim=6)
        output = model(torch.randn(3, 6), return_quantized=True)
        self.assertEqual(len(output), 5)
        quantized = output[4]
        self.assertEqual(tuple(quantized.shape), (3, 4))
        self.assertEqual(tuple(output[3].shape), (3, 2))

    def test_behaviour_loss_reaches_real_encoder(self):
        from types import SimpleNamespace
        config = SimpleNamespace(
            hidden_sizes=[8], codebook_num=2, codebook_size=(8, 8),
            codebook_dim=4, dropout=0.0, beta=0.25, vq_type="vq",
            ema_decay=0.99, sk_epsilon=-1.0, sk_iters=-1, fix_code_embs=False,
        )
        model = RQVAE(config, in_dim=6)
        model.zero_grad(set_to_none=True)

        def encode(x: torch.Tensor) -> torch.Tensor:
            return model(x, return_quantized=True)[4]

        anchors = encode(torch.randn(2, 6))
        positives = encode(torch.randn(2, 6))
        negatives = encode(torch.randn(2 * 3, 6)).view(2, 3, 4)
        loss = behaviour_contrastive_loss(anchors, positives, negatives)
        loss.backward()
        encoder_norm = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.encoder.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(encoder_norm, 0.0)


if __name__ == "__main__":
    unittest.main()
