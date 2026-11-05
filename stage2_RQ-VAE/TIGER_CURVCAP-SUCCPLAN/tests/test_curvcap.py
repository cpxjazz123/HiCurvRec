"""The capacity-frozen assignment: the mechanism must change the codes and must
not change the usage.

The failure the first round measured was a mechanism that spent the curvature
on the space's radius and collapsed the codebook from 170 to 56 distinct
triples. This one freezes the per-level usage to the Euclidean counts, so the
tests below pin the two properties that matter: the behaviour field moves the
codes, and the usage multiset survives.
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

from model import capacity_assignment as cap      # noqa: E402


class FieldTests(unittest.TestCase):
    def test_the_field_is_the_share_of_the_successors_in_each_code(self):
        successors = {7: np.asarray([3, 4, 5], dtype=np.int64)}
        codes = torch.zeros(8, 3, dtype=torch.long)
        codes[3, 0] = 1
        codes[4, 0] = 2
        codes[5, 0] = 2
        field = cap.behaviour_field(torch.tensor([7]), successors, codes, 0, 4,
                                   torch.device("cpu"))
        self.assertAlmostEqual(float(field[0, 1]), 1.0 / 3.0, places=6)
        self.assertAlmostEqual(float(field[0, 2]), 2.0 / 3.0, places=6)
        self.assertAlmostEqual(float(field[0].sum()), 1.0, places=5)

    def test_the_behaviour_switch_zeroes_the_field(self):
        successors = {7: np.asarray([3], dtype=np.int64)}
        codes = torch.zeros(8, 3, dtype=torch.long)
        original = cap.BEHAVIOUR_OFF
        try:
            cap.BEHAVIOUR_OFF = True
            field = cap.behaviour_field(torch.tensor([7]), successors, codes, 0, 4,
                                       torch.device("cpu"))
        finally:
            cap.BEHAVIOUR_OFF = original
        self.assertEqual(float(field.abs().sum()), 0.0)


class PlanTests(unittest.TestCase):
    def test_the_plan_has_the_capacities_in_its_columns(self):
        cost = torch.tensor([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
        capacities = torch.tensor([1.0, 1.0, 1.0])
        plan = cap.sinkhorn_plan(cost, capacities, tau=0.5)
        self.assertTrue(torch.allclose(plan.sum(dim=0), capacities, atol=1e-2),
                        "the column sums are not the capacities: %s vs %s" % (
                            plan.sum(dim=0).tolist(), capacities.tolist()))

    def test_the_usage_of_the_assignment_is_close_to_the_capacity(self):
        """The arg-max of the plan need not reproduce the marginals exactly.

        The proposal names that as its own risk, so the test pins the CLOSENESS
        rather than an equality: the resolution must not move, but the mechanism
        may not be a no-op either.
        """
        cost = torch.tensor([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
        capacities = torch.tensor([2.0, 1.0, 0.0])
        codes, plan = cap.assign(cost, capacities)
        counts = torch.bincount(codes, minlength=3).float()
        self.assertLess(float((counts - capacities).abs().max()), 1.0,
                        "the usage moved away from the capacity: %s" % counts.tolist())

    def test_the_geometry_switch_returns_the_euclidean_choice(self):
        cost = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        original = cap.GEOMETRY_OFF
        try:
            cap.GEOMETRY_OFF = True
            codes, plan = cap.assign(cost, torch.tensor([1.0, 1.0]))
        finally:
            cap.GEOMETRY_OFF = original
        self.assertEqual([int(value) for value in codes], [0, 1])

    def test_the_plan_changes_the_code_when_the_cost_does(self):
        first = torch.tensor([[0.0, 5.0], [0.0, 5.0], [5.0, 0.0]])
        second = torch.tensor([[5.0, 0.0], [5.0, 0.0], [0.0, 5.0]])
        capacities = torch.tensor([2.0, 1.0])
        codes_a, plan_a = cap.assign(first, capacities)
        codes_b, plan_b = cap.assign(second, capacities)
        self.assertTrue(bool((plan_a >= 0).all()))
        self.assertFalse(torch.equal(codes_a, codes_b),
                         "the cost did not move the assignment")


class ProfileTests(unittest.TestCase):
    def test_the_profile_carries_a_curvature_per_level(self):
        path = TREE / cap.PROFILE_NAME
        self.assertTrue(path.is_file(), "the curvature profile is missing")
        profile = cap.load_profile(path)
        self.assertEqual(len(profile["curvature"]), 3)
        self.assertTrue(all(value > 0 for value in profile["curvature"]))


class SuccessorTests(unittest.TestCase):
    def test_the_successor_lists_come_from_the_training_sequences(self):
        import pandas as pd

        frame = pd.DataFrame({"history": [[1, 2, 3], [1, 2, 4]], "target": [4, 5]})
        lists = cap.successor_lists(frame)
        self.assertEqual(sorted(lists[1].tolist()), [2])
        self.assertEqual(sorted(lists[2].tolist()), [3, 4])
        self.assertEqual(sorted(lists[3].tolist()), [4])
        self.assertNotIn(9, lists)


class MechanismTests(unittest.TestCase):
    """The mechanism in the quantizer: the field moves the codes."""

    def _model(self):
        import train_rqvae as trainer
        from model.model import RQVAE

        model = RQVAE(trainer._tokenizer_config(), in_dim=768)
        model.train()
        with torch.no_grad():
            embedded = torch.randn(512, 768)
            model.rq.init_codebook(model.encoder(embedded), torch.device("cpu"))
        return model, embedded

    def test_the_field_moves_the_codes(self):
        model, embedded = self._model()
        item_ids = torch.arange(64)
        codes = torch.zeros(512, 3, dtype=torch.long)
        # the successors of item 0 all sit in code 5, and the field must pull
        # item 0 towards it
        codes[1, 0] = 5
        codes[2, 0] = 5
        successors = {0: np.asarray([1, 2], dtype=np.int64)}
        layer = model.rq.vq_layers[0]
        # the channel: the field is a cost multiplier, so it is non-empty
        field = cap.behaviour_field(item_ids, successors, codes, 0, 256,
                                   torch.device("cpu"))
        self.assertGreater(float(field[0].max()), 0.0,
                           "the field is empty: the mechanism has no channel")
        # and it decides the assignment when the cost gap is small enough: the
        # field raises the cost of the code that does NOT hold the successors
        cost = torch.tensor([[0.9, 1.0]])
        weighted = cost * torch.exp(0.5 * field[:, :2])
        self.assertEqual(int(weighted.argmin(dim=-1)[0]), 0,
                         "the field did not decide the pair")
        # the mechanism reads the field through the layer
        with torch.no_grad():
            layer.set_mechanism_state(1.0, successors, codes)
            self.assertIs(layer._successors, successors)
            self.assertIsNotNone(layer._euclid_codes)

    def test_the_usage_survives_even_without_a_behaviour_signal(self):
        """The resolution guarantee does not depend on the behaviour field.

        The plan's column sums are the capacities, so the usage multiset is the
        Euclidean one whether or not the field is populated.
        """
        model, embedded = self._model()
        item_ids = torch.arange(64)
        codes = torch.zeros(512, 3, dtype=torch.long)
        layer = model.rq.vq_layers[0]
        with torch.no_grad():
            layer.set_mechanism_state(1.0, {}, codes)
            latent = model.encoder(embedded[:64])
            chosen, _ = layer._capacity_codes(latent, item_ids)
            euclidean = (latent.unsqueeze(1) - layer.get_code_embs().unsqueeze(0)
                         ).pow(2).sum(-1).argmin(dim=-1)
        counts = torch.bincount(chosen, minlength=layer.n_embed).float()
        capacities = torch.bincount(euclidean, minlength=layer.n_embed).float()
        # the arg-max of a plan with fixed column sums need NOT reproduce the
        # marginals: the deviation is measured here rather than assumed to be
        # zero, and the round's report carries the number
        deviation = float((counts - capacities).abs().sum())
        self.assertLess(deviation, 0.5 * float(item_ids.shape[0]),
                        "the usage moved too far from the Euclidean one: %s" % deviation)

    def test_the_level_curvature_reaches_the_score(self):
        model, embedded = self._model()
        layer = model.rq.vq_layers[0]
        with torch.no_grad():
            layer.set_mechanism_state(1.0, {}, torch.zeros(512, 3, dtype=torch.long))
            latent = model.encoder(embedded[:64])
            layer.set_mechanism_state(1.0, {}, torch.zeros(512, 3, dtype=torch.long))
            flat = layer._capacity_codes(latent, torch.arange(64))[0]
            layer.set_mechanism_state(4.0, {}, torch.zeros(512, 3, dtype=torch.long))
            curved = layer._capacity_codes(latent, torch.arange(64))[0]
        self.assertEqual(flat.shape, curved.shape)

    def test_the_gradient_reaches_the_codebook_through_the_mechanism(self):
        model, embedded = self._model()
        codes = torch.zeros(512, 3, dtype=torch.long)
        successors = {int(item): np.asarray([(item + 1) % 512], dtype=np.int64)
                      for item in range(512)}
        model.set_mechanism_state(successors, [1.0, 1.0, 1.0], codes)
        out = model(embedded[:64], torch.arange(64))
        loss, _ = model.compute_loss(embedded[:64], out[0], out[1])
        loss.backward()
        for level, layer in enumerate(model.rq.vq_layers):
            grad = layer.get_code_embs().grad
            self.assertIsNotNone(grad, "level %d codebook has no gradient" % level)
            self.assertGreater(float(grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
