"""The exact-quota curvature vote assignment.

Two properties carry the mechanism and are pinned here: the usage multiset is
the Euclidean one EXACTLY (a transposition cannot change it), and the behaviour
term enters the cost ADDITIVELY, so it can move the top-1 choice.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import eqvote                              # noqa: E402


class SuccessorTests(unittest.TestCase):
    def test_the_successor_lists_come_from_the_training_sequences(self):
        import pandas as pd

        frame = pd.DataFrame({"history": [[1, 2, 3], [1, 2, 4]], "target": [4, 5]})
        lists = eqvote.successor_lists(frame)
        self.assertEqual(sorted(lists[1].tolist()), [2])
        self.assertEqual(sorted(lists[2].tolist()), [3, 4])
        self.assertNotIn(9, lists)


class VoteTests(unittest.TestCase):
    def test_the_share_is_the_kernel_weighted_vote_of_the_successors(self):
        item_ids = torch.tensor([7])
        successors = {7: np.asarray([1, 2], dtype=np.int64)}
        codes = torch.zeros(4, 3, dtype=torch.long)
        codes[1, 0] = 1
        codes[2, 0] = 2
        kernel = torch.tensor([[1.0, 0.5, 0.25],
                               [0.5, 1.0, 0.5],
                               [0.25, 0.5, 1.0]])
        shares = eqvote.vote_shares(item_ids, successors, codes, 0, kernel,
                                    torch.device("cpu"))
        # the share of a code is the sum of the kernels of the votes at it,
        # normalised over all the codes
        # the share of a code is the sum of the kernels of its votes, normalised
        # over all the codes and all the votes
        total = (0.5 + 0.25) + (1.0 + 0.5) + (0.5 + 1.0)
        self.assertAlmostEqual(float(shares[0, 1]), (1.0 + 0.5) / total, places=6)
        self.assertAlmostEqual(float(shares[0, 2]), (0.5 + 1.0) / total, places=6)
        self.assertAlmostEqual(float(shares[0].sum()), 1.0, places=5)

    def test_the_behaviour_switch_zeroes_the_votes(self):
        item_ids = torch.tensor([7])
        successors = {7: np.asarray([1], dtype=np.int64)}
        codes = torch.zeros(4, 3, dtype=torch.long)
        kernel = torch.eye(3)
        original = eqvote.BEHAVIOUR_OFF
        try:
            eqvote.BEHAVIOUR_OFF = True
            shares = eqvote.vote_shares(item_ids, successors, codes, 0, kernel,
                                        torch.device("cpu"))
        finally:
            eqvote.BEHAVIOUR_OFF = original
        self.assertEqual(float(shares.abs().sum()), 0.0)


class TranspositionTests(unittest.TestCase):
    def test_the_usage_multiset_survives_the_transpositions(self):
        """The resolution guarantee: a transposition cannot change the counts."""
        torch.manual_seed(0)
        cost = torch.randn(64, 8)
        labels = torch.randint(0, 8, (64,))
        improved = eqvote.accept_transpositions(cost, labels)
        before = torch.bincount(labels, minlength=8)
        after = torch.bincount(improved, minlength=8)
        self.assertTrue(torch.equal(before, after),
                        "the usage moved: %s vs %s" % (before.tolist(), after.tolist()))

    def test_the_transpositions_only_lower_the_cost(self):
        torch.manual_seed(1)
        cost = torch.randn(48, 6)
        labels = torch.randint(0, 6, (48,))
        improved = eqvote.accept_transpositions(cost, labels)

        def total(values):
            return float(cost[torch.arange(values.shape[0]), values].sum())

        self.assertLessEqual(total(improved), total(labels) + 1e-6)

    def test_a_transposition_is_accepted_when_it_lowers_the_cost(self):
        cost = torch.tensor([[5.0, 0.0], [0.0, 5.0]])
        labels = torch.tensor([0, 1])
        improved = eqvote.accept_transpositions(cost, labels)
        self.assertEqual(sorted(improved.tolist()), [0, 1])
        self.assertEqual(improved.tolist(), [1, 0],
                         "the swap that lowers the cost was not accepted")


class CostTests(unittest.TestCase):
    def test_the_vote_term_is_additive(self):
        euclidean = torch.tensor([[1.0, 2.0]])
        shares = torch.tensor([[0.0, 1.0]])
        cost = eqvote.additive_cost(euclidean, shares, 0.5)
        self.assertAlmostEqual(float(cost[0, 0]), 1.0, places=6)
        self.assertAlmostEqual(float(cost[0, 1]), 1.5, places=6)

    def test_the_switches_leave_the_cost_alone(self):
        euclidean = torch.tensor([[1.0, 2.0]])
        shares = torch.tensor([[0.0, 1.0]])
        for name in ("BEHAVIOUR_OFF", "GEOMETRY_OFF"):
            original = getattr(eqvote, name)
            try:
                setattr(eqvote, name, True)
                cost = eqvote.additive_cost(euclidean, shares, 0.5)
            finally:
                setattr(eqvote, name, original)
            self.assertTrue(torch.equal(cost, euclidean), name + " changed the cost")


class ProfileTests(unittest.TestCase):
    def test_the_profile_carries_a_curvature_and_a_delta_per_level(self):
        import json

        path = TREE / "curvature_profile.json"
        self.assertTrue(path.is_file(), "the curvature profile is missing")
        profile = json.loads(path.read_text())
        self.assertEqual(len(profile["curvature"]), 3)
        self.assertEqual(len(profile["delta"]), 3)
        for value in profile["curvature"]:
            self.assertGreater(value, 0.0)


class MechanismTests(unittest.TestCase):
    def _model(self):
        import train_rqvae as trainer
        from model.model import RQVAE

        model = RQVAE(trainer._tokenizer_config(), in_dim=768)
        model.train()
        with torch.no_grad():
            embedded = torch.randn(512, 768)
            model.rq.init_codebook(model.encoder(embedded), torch.device("cpu"))
        return model, embedded

    def test_the_gradient_reaches_the_codebook_through_the_mechanism(self):
        model, embedded = self._model()
        codes = torch.zeros(512, 3, dtype=torch.long)
        successors = {int(item): np.asarray([(item + 1) % 512], dtype=np.int64)
                      for item in range(512)}
        model.set_mechanism_state(successors, codes, [1.0, 1.0, 1.0], [0.0, 0.0, 1.0])
        out = model(embedded[:64], torch.arange(64))
        loss, _ = model.compute_loss(embedded[:64], out[0], out[1])
        loss.backward()
        for level, layer in enumerate(model.rq.vq_layers):
            grad = layer.get_code_embs().grad
            self.assertIsNotNone(grad, "level %d codebook has no gradient" % level)
            self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_a_level_without_weight_is_the_euclidean_pass(self):
        model, embedded = self._model()
        codes = torch.zeros(512, 3, dtype=torch.long)
        successors = {int(item): np.asarray([(item + 1) % 512], dtype=np.int64)
                      for item in range(512)}
        model.set_mechanism_state(successors, codes, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
        layer = model.rq.vq_layers[0]
        with torch.no_grad():
            latent = model.encoder(embedded[:64])
            chosen = layer._vote_labels(latent, torch.arange(64))
            euclidean = layer._distances(latent).argmin(dim=-1)
        self.assertTrue(torch.equal(chosen, euclidean),
                        "a level without weight changed the assignment")


if __name__ == "__main__":
    unittest.main()
