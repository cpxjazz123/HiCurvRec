"""The mechanism as the model actually runs it, not as the module defines it.

The frozen equations can be right and the wiring can still be dead: the cost
computed and never used, the plan's gradient cancelled by a residual chain that
re-adds it, the switch flipped and the baseline path taken.  Each test here
asserts one of the properties that only the assembled model can show.

One of them is a regression test for a defect found by measurement, not by
reading: the level-1 plan's gradient was cancelled exactly.  The quantised
vector entered the decoder with ``+1`` and the residual with ``-1``, and the
levels above re-add the residual with derivative one, so ``quant`` contributed
``+1 - 1 = 0`` to every parameter of the quantiser - the stratum parameters and
the level-1 codebook received zero gradient from the reconstruction while the
isolated plan received 0.57 and the codebook 8262.  The residual chain is now a
value chain (as the baseline's own straight-through estimator already made it),
which keeps the plan's gradient.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

import train_rqvae as trainer
from model import RQVAE, stratum
from model.layers import RQLayer


def small_config(stratified: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        hidden_sizes=(16,), codebook_num=3, codebook_size=(8, 8, 8),
        codebook_dim=4, dropout=0.0, beta=0.25, vq_type="vq", ema_decay=0.99,
        fix_code_embs=False, sk_epsilon=0.003, sk_iters=5,
        stratified_enabled=stratified)


def small_model(seed: int = 5, stratified: bool = True) -> RQVAE:
    torch.manual_seed(seed)
    model = RQVAE(small_config(stratified), in_dim=8)
    with torch.no_grad():
        model.rq.vq_layers[0].embed.weight.normal_(0, 0.3)
    return model


def batch(rows: int = 128, scale: float = 0.002):
    """A warm start: the costs are comparable to the plan's temperature."""
    torch.manual_seed(11)
    embeddings = torch.randn(rows, 8) * scale
    ids = torch.arange(rows)
    ranks = torch.linspace(0.0, 1.0, rows)
    return embeddings, ids, ranks


class TheSwitchIsTheConfiguration(unittest.TestCase):
    def test_the_mechanism_brings_its_one_parameter(self):
        model = small_model(stratified=True)
        self.assertTrue(model.rq.stratified_enabled)
        self.assertEqual(tuple(model.rq.stratum_alpha.shape), (8,))
        expected = stratum.stratum_init(8)
        self.assertLess(float((model.rq.stratum_alpha.detach() - expected).abs().max()), 1e-6)

    def test_without_the_switch_the_model_has_no_mechanism(self):
        model = small_model(stratified=False)
        self.assertFalse(model.rq.stratified_enabled)
        self.assertFalse(hasattr(model.rq, "stratum_alpha"))

    def test_without_the_switch_the_level_one_index_is_the_baseline_argmin(self):
        # The switch must be the baseline, not a variant of it: the level-1 index
        # is the Euclidean argmin of the baseline's own quantizer.
        model = small_model(stratified=False)
        embeddings, ids, _ = batch()
        with torch.no_grad():
            _, _, _, tokens = model(embeddings, ids)
            latent = model.encoder(embeddings)
            layer = model.rq.vq_layers[0]
            expected = layer._indices(layer._distances(latent.view(-1, layer.dim)), False)
        self.assertTrue(torch.equal(tokens[:, 0], expected))


class TheBehaviourMovesTheAssignment(unittest.TestCase):
    def test_the_offset_pulls_the_codeword_towards_the_item_s_rank(self):
        model = small_model()
        embeddings, ids, ranks = batch()
        model.set_stratum_table(ranks, ids)
        model.set_stratum_beta(0.0)
        with torch.no_grad():
            _, _, _, without = model(embeddings, ids)
            beta = float(model.level1_chosen_distance().median()) / stratum.RADIUS
        model.set_stratum_beta(beta)
        with torch.no_grad():
            _, _, _, with_offset = model(embeddings, ids)
        self.assertGreater(beta, 0.0)
        changed = float((with_offset[:, 0] != without[:, 0]).to(torch.float64).mean())
        self.assertGreater(changed, 0.5)
        strata = torch.sigmoid(model.rq.stratum_alpha.detach())
        gap_without = float((strata[without[:, 0]] - ranks).abs().mean())
        gap_with = float((strata[with_offset[:, 0]] - ranks).abs().mean())
        # Measured on this configuration: 0.365 -> 0.031.
        self.assertLess(gap_with, 0.5 * gap_without)

    def test_the_behaviour_switch_reproduces_the_plain_assignment(self):
        model = small_model()
        embeddings, ids, ranks = batch()
        model.set_stratum_table(ranks, ids)
        model.set_stratum_beta(0.0)
        with torch.no_grad():
            _, _, _, plain = model(embeddings, ids)
            model.set_stratum_beta(float(model.level1_chosen_distance().median()))
            _, _, _, with_offset = model(embeddings, ids)
        original = stratum.BEHAVIOUR_OFF
        try:
            stratum.BEHAVIOUR_OFF = True
            with torch.no_grad():
                _, _, _, off = model(embeddings, ids)
        finally:
            stratum.BEHAVIOUR_OFF = original
        # With the scale set, the switch is what decides: off is the plain
        # association, not a weakened one.
        self.assertFalse(torch.equal(off[:, 0], with_offset[:, 0]))
        self.assertTrue(torch.equal(off[:, 0], plain[:, 0]))

    def test_the_readout_reports_what_the_switch_did(self):
        model = small_model()
        embeddings, ids, ranks = batch()
        model.set_stratum_table(ranks, ids)
        model.set_stratum_beta(0.0)
        with torch.no_grad():
            model(embeddings, ids)
        model.set_stratum_beta(float(model.level1_chosen_distance().median()))
        with torch.no_grad():
            model.rq.observe_level1 = True
            try:
                model(embeddings, ids)
            finally:
                model.rq.observe_level1 = False
        readout = model.rq.level1_readout
        self.assertIsNotNone(readout)
        self.assertGreater(readout["offset_abs_mean"], 0.0)
        self.assertGreater(readout["changed_fraction"], 0.0)
        self.assertLessEqual(readout["changed_fraction"], 1.0)


class TheGeometrySwitchIsTheDistance(unittest.TestCase):
    def test_the_model_s_plan_is_the_module_s_cost_under_both_geometries(self):
        embeddings, ids, ranks = batch()
        for geometry_off in (False, True):
            original = stratum.GEOMETRY_OFF
            try:
                stratum.GEOMETRY_OFF = geometry_off
                model = small_model()
                model.set_stratum_table(ranks, ids)
                model.set_stratum_beta(0.0)
                with torch.no_grad():
                    _, _, _, tokens = model(embeddings, ids)
                with torch.no_grad():
                    latent = model.encoder(embeddings)
                    codebook = model.rq.vq_layers[0].get_code_embs()
                    rho = model.rq._gather_rank(ids, len(embeddings), embeddings.device)
                    cost = stratum.level1_cost(latent, codebook, stratum.curvature(), rho,
                                               torch.sigmoid(model.rq.stratum_alpha), 0.0)
                    expected = stratum.assignment(stratum.plan(cost, 0.003, 5))
                self.assertTrue(torch.equal(tokens[:, 0], expected))
                self.assertEqual(stratum.curvature(), 0.0 if geometry_off else 1.0)
            finally:
                stratum.GEOMETRY_OFF = original


class TheReconstructionReachesTheMechanism(unittest.TestCase):
    def test_the_stratum_parameters_and_the_codebook_receive_gradient(self):
        # The regression test for the cancellation: before the residual chain
        # became a value chain, all three of these were exactly zero.
        model = small_model()
        embeddings, ids, ranks = batch()
        model.set_stratum_table(ranks, ids)
        model.set_stratum_beta(0.0)
        with torch.no_grad():
            model(embeddings, ids)
        model.set_stratum_beta(float(model.level1_chosen_distance().median()))
        model.zero_grad(set_to_none=True)
        reconstructed, _, _, _ = model(embeddings, ids)
        torch.nn.functional.mse_loss(reconstructed, embeddings).backward()
        self.assertGreater(float(model.rq.stratum_alpha.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.rq.vq_layers[0].embed.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.encoder.mlp[1].weight.grad.abs().sum()), 0.0)

    def test_the_quantised_value_is_the_chosen_codeword(self):
        model = small_model()
        embeddings, ids, ranks = batch()
        model.set_stratum_table(ranks, ids)
        model.set_stratum_beta(0.5)
        with torch.no_grad():
            latent = model.encoder(embeddings)
            codebook = model.rq.vq_layers[0].get_code_embs()
            quant, _, _, indices = model.rq._stratified_level(
                latent, model.rq.vq_layers[0], ids)
        chosen = torch.nn.functional.embedding(indices, codebook)
        # The forward value is the chosen codeword.  It is not bit-identical:
        # ``one_hot + P - P.detach()`` rounds ``one_hot + P`` to float32 before
        # subtracting ``P``, which leaves one ulp (measured 5.96e-08 here).  The
        # exact form would need the association ``one_hot + (P - P.detach())``,
        # which is the same function and the same gradient; the frozen
        # specification writes the left-to-right form, so that is what runs.
        max_difference = float((quant.detach() - chosen.detach()).abs().max())
        self.assertLess(max_difference, 1e-6)
        self.assertGreaterEqual(max_difference, 0.0)


class TheRankTable(unittest.TestCase):
    def test_an_item_outside_the_table_is_neutral(self):
        model = small_model()
        ranks = torch.tensor([0.1, 0.4, 0.9])
        model.set_stratum_table(ranks, torch.tensor([3, 7, 11]))
        gathered = model.rq._gather_rank(torch.tensor([3, 7, 11, 2]), 4, torch.device("cpu"))
        self.assertAlmostEqual(float(gathered[0]), 0.1, places=6)
        self.assertAlmostEqual(float(gathered[2]), 0.9, places=6)
        self.assertAlmostEqual(float(gathered[3]), stratum.NEUTRAL_RANK, places=6)

    def test_no_table_at_all_is_neutral_everywhere(self):
        model = small_model()
        gathered = model.rq._gather_rank(torch.arange(5), 5, torch.device("cpu"))
        self.assertTrue(bool((gathered == stratum.NEUTRAL_RANK).all()))

    def test_the_table_refuses_what_it_cannot_index(self):
        model = small_model()
        with self.assertRaises(ValueError):
            model.set_stratum_table(torch.tensor([0.1, 0.2]), torch.tensor([5, 4]))
        with self.assertRaises(ValueError):
            model.set_stratum_table(torch.tensor([0.1, 1.7]), torch.tensor([4, 5]))
        with self.assertRaises(ValueError):
            model.set_stratum_table(torch.tensor([0.1]), torch.tensor([4, 5]))

    def test_the_scale_must_be_a_finite_distance(self):
        model = small_model()
        with self.assertRaises(ValueError):
            model.set_stratum_beta(-1.0)
        with self.assertRaises(ValueError):
            model.set_stratum_beta(float("nan"))
        model.set_stratum_beta(0.25)
        self.assertAlmostEqual(float(model.rq.stratum_beta), 0.25, places=6)

    def test_a_switched_off_model_has_no_scale_to_set(self):
        # A switched-off model must carry no mechanism state at all: an extra
        # buffer would add a key to the state dict and the checkpoint would no
        # longer be the baseline's (measured: "unexpected_keys=['rq.stratum_beta']").
        model = small_model(stratified=False)
        self.assertFalse(hasattr(model.rq, "stratum_beta"))
        model.set_stratum_beta(9.0)
        self.assertFalse(hasattr(model.rq, "stratum_beta"))
        self.assertNotIn("rq.stratum_beta", model.state_dict())


class TheMeasurement(unittest.TestCase):
    def test_beta_is_the_median_chosen_distance_over_the_radius(self):
        model = small_model()
        distances = torch.tensor([1.0, 0.5, 2.0, 0.25])
        model.rq.level1_chosen_distance = distances
        beta = trainer._measure_stratum_beta(model, torch.device("cpu"), 1)
        expected = float(distances.to(torch.float64).median()) / stratum.RADIUS
        self.assertAlmostEqual(beta, expected, places=9)
        self.assertAlmostEqual(float(model.rq.stratum_beta), expected, places=6)

    def test_an_unmeasured_pass_reports_nothing(self):
        model = small_model()
        model.rq.level1_chosen_distance = None
        self.assertIsNone(trainer._measure_stratum_beta(model, torch.device("cpu"), 1))


class TheDrift(unittest.TestCase):
    def test_the_per_codeword_mean_rank_uses_only_the_assigned_items(self):
        tokens = np.zeros((4, 3), dtype=np.int64)
        tokens[:, 0] = [0, 0, 1, 1]
        train_ids = np.arange(4)
        ranks = torch.tensor([0.1, 0.3, 0.9, 0.7])
        alpha = stratum.stratum_init(8)
        report = trainer._stratum_drift(tokens, train_ids, alpha, ranks)
        strata = torch.sigmoid(alpha)
        expected = float(((torch.tensor([0.2, 0.8]) - strata[:2]).abs()).mean())
        self.assertEqual(report["stratum_used_codes"], 2)
        self.assertAlmostEqual(report["stratum_drift"], expected, places=6)
        self.assertAlmostEqual(report["stratum_largest_code_share"], 0.5, places=6)


class TheExportIsTheTrainingAssignment(unittest.TestCase):
    def test_the_exported_index_is_what_the_training_pass_produced(self):
        model = small_model()
        embeddings, ids, ranks = batch()
        model.set_stratum_table(ranks, ids)
        model.set_stratum_beta(0.7)
        model.eval()
        with torch.no_grad():
            _, _, _, trained = model(embeddings, ids)
            exported = model.get_indices(embeddings, item_ids=ids)
        self.assertTrue(torch.equal(trained, exported))


if __name__ == "__main__":
    unittest.main()
