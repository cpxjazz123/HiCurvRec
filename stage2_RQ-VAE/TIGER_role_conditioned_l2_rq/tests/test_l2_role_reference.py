"""R05 acceptance: level 2 consumes the role mixture, levels 1 and 3 the gate.

The production routing is compared against ``tests/reference_l2_routing.py``,
an independent scalar implementation, and against hand-computed answers that
are written literally in this file.  The tests fail on the R04 tree (role at
level index 2) and pass only when the mixture moves to level index 1.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))

SPEC = importlib.util.spec_from_file_location(
    'reference_l2_routing', TREE / 'tests' / 'reference_l2_routing.py')
REFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFERENCE)


class KnownAnswers(unittest.TestCase):
    """Hand-computed values, written before the production code is consulted."""

    def test_softmax_rows_hand_answer(self):
        rows = REFERENCE.softmax_rows([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        thirds = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        concentrated = [0.9094429985127419, 0.04527850074362907, 0.04527850074362907]
        for got, want in zip(rows[0], thirds):
            self.assertAlmostEqual(got, want, places=12)
        for got, want in zip(rows[1], concentrated):
            self.assertAlmostEqual(got, want, places=12)

    def test_routing_hand_answer(self):
        # rows: role 0, role 1, uncovered, global
        logits = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        gate = [[[0.1, 0.2, 0.7], [0.2, 0.2, 0.6]],
                [[0.3, 0.3, 0.4], [0.5, 0.25, 0.25]]]
        routed = REFERENCE.routing([0, 1], logits, gate)
        # The second level is the role mixture; the first and third copy the gate.
        self.assertEqual(routed[0][0], [0.1, 0.2, 0.7])
        self.assertEqual(routed[0][2], [0.2, 0.2, 0.6])
        self.assertEqual(routed[1][0], [0.3, 0.3, 0.4])
        self.assertEqual(routed[1][2], [0.5, 0.25, 0.25])
        for got, want in zip(routed[0][1], [1.0 / 3.0] * 3):
            self.assertAlmostEqual(got, want, places=12)
        for got, want in zip(routed[1][1],
                             [0.9094429985127419, 0.04527850074362907, 0.04527850074362907]):
            self.assertAlmostEqual(got, want, places=12)

    def test_routing_switch_and_uncovered_row_hand_answer(self):
        logits = [[30.0, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 30.0], [0.0, 0.0, 0.0]]
        gate = [[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]]
        # role -1 -> uncovered row 2; behaviour off -> global row 3.
        uncovered = REFERENCE.routing([-1], logits, gate)[0][1]
        alone = REFERENCE.routing([-1], logits, gate, behavior_on=False)[0][1]
        for got, want in zip(uncovered, [0.0, 0.0, 1.0]):
            self.assertAlmostEqual(got, want, places=12)
        for got in alone:
            self.assertAlmostEqual(got, 1.0 / 3.0, places=12)
        self.assertEqual(REFERENCE.routing([-1], logits, gate)[0][0], [0.0, 0.0, 1.0])
        self.assertEqual(REFERENCE.routing([-1], logits, gate)[0][2], [1.0, 0.0, 0.0])

    def test_euclidean_cost_hand_answer(self):
        cost = REFERENCE.euclidean_cost([[1.0, 0.0]], [[0.0, 0.0], [3.0, 0.0]],
                                        [[0.2, 0.3, 0.5]], 1.0)
        self.assertAlmostEqual(cost[0][0], 1.0, places=12)
        self.assertAlmostEqual(cost[0][1], 4.0, places=12)


class ProductionAgainstReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.root = next(p for p in TREE.parents if (p / 'assets').is_dir())
        cls.embeddings = torch.from_numpy(
            np.load(cls.root / 'stage1_GeneEmbedding/output/sentence_t5.npy').astype('float32'))
        cls.initial = torch.load(cls.root / 'results/stage2_RQ-VAE/TIGER_RQ-VAE/rqvae_best.pth',
                                 map_location='cpu', weights_only=False)['state_dict']

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def model(self):
        from model.mechanism import RQVAE
        torch.manual_seed(42)
        config = SimpleNamespace(hidden_sizes=[512, 256, 128], codebook_dim=32, dropout=0.,
                                  codebook_num=3, codebook_size=[256, 256, 256], vq_type='vq',
                                  beta=.25, sk_epsilon=.003, sk_iters=50)
        model = RQVAE(config, in_dim=768)
        model.load_state_dict(self.initial, strict=False)
        model.freeze_tangent_scale(self.embeddings)
        return model

    def test_gate_has_two_rows_mapped_to_levels_one_and_three(self):
        from model.layer_gate import LEVELS
        self.assertEqual(LEVELS, 2, 'row 0 serves level 1 and row 1 serves level 3')
        model = self.model()
        self.assertEqual(tuple(model.structure.attention.query.shape)[0], 2)

    def test_level_routing_matches_the_independent_reference(self):
        model = self.model()
        ids = np.array([1234, 20000, 5])
        roles = model.role_mix.table.roles_for(ids).tolist()
        rows = model.role_mix.rows_for(ids, torch.device('cpu'))
        torch.manual_seed(7)
        own = torch.randn(3, 2, 3)
        with torch.no_grad():
            model.role_mix.logits[rows[0]] = torch.tensor([2.0, 0.0, -1.0])
            model.role_mix.logits[rows[1]] = torch.tensor([0.0, 3.0, 0.0])
            model.role_mix.logits[rows[2]] = torch.tensor([-1.0, 0.0, 2.0])
        reference = REFERENCE.routing(roles, model.role_mix.logits.detach().tolist(),
                                      own.tolist(), global_index=model.role_mix.global_row)
        for level in range(3):
            production = model.level_weights(own, level, rows).detach().tolist()
            for item in range(3):
                for expert in range(3):
                    self.assertAlmostEqual(production[item][expert],
                                           reference[item][level][expert], places=6,
                                           msg='level %d item %d' % (level, item))

    def test_behaviour_switch_matches_the_reference_and_leaves_other_levels_alone(self):
        model = self.model()
        ids = np.array([1234, 20000])
        roles = model.role_mix.table.roles_for(ids).tolist()
        rows = model.role_mix.rows_for(ids, torch.device('cpu'))
        torch.manual_seed(11)
        own = torch.randn(2, 2, 3)
        with torch.no_grad():
            model.role_mix.logits[rows[0]] = torch.tensor([3.0, 0.0, 0.0])
            model.role_mix.logits[rows[1]] = torch.tensor([0.0, 3.0, 0.0])
            model.role_mix.logits[model.role_mix.global_row] = torch.tensor([0.0, 0.0, 3.0])
        before = [model.level_weights(own, level, rows).detach().clone() for level in range(3)]
        model.role_mixture_on = False
        after = [model.level_weights(own, level, rows).detach().clone() for level in range(3)]
        reference = REFERENCE.routing(roles, model.role_mix.logits.detach().tolist(),
                                      own.tolist(), behavior_on=False,
                                      global_index=model.role_mix.global_row)
        for item in range(2):
            for expert in range(3):
                self.assertAlmostEqual(float(after[1][item][expert]),
                                       reference[item][1][expert], places=6)
        torch.testing.assert_close(before[0], after[0], atol=0, rtol=0)
        torch.testing.assert_close(before[2], after[2], atol=0, rtol=0)
        self.assertFalse(torch.allclose(before[1], after[1]),
                         'the behaviour switch must change the level-2 cost')

    def test_curvature_off_level_two_cost_matches_the_scalar_euclidean_reference(self):
        model = self.model()
        ids = np.array([1234, 20000])
        rows = model.role_mix.rows_for(ids, torch.device('cpu'))
        torch.manual_seed(13)
        own = torch.randn(2, 2, 3)
        residual = torch.randn(2, 32)
        codewords = torch.randn(8, 32)
        weights = model.level_weights(own, 1, rows)
        model.curvature_on = False
        with torch.no_grad():
            production = model.assignment_cost(residual, codewords, weights).tolist()
        reference = REFERENCE.euclidean_cost(residual.tolist(), codewords.tolist(),
                                             weights.tolist(), float(model._tangent_scale_value))
        for item in range(2):
            for code in range(8):
                expected = reference[item][code]
                self.assertAlmostEqual(production[item][code], expected,
                                       delta=1e-4 * max(1.0, abs(expected)))

    def test_level_two_differs_from_the_other_levels_when_the_role_row_differs(self):
        model = self.model()
        ids = np.array([1234])
        rows = model.role_mix.rows_for(ids, torch.device('cpu'))
        own = torch.zeros(1, 2, 3)
        own[0, 0] = torch.tensor([0.0, 0.0, 1.0])
        own[0, 1] = torch.tensor([1.0, 0.0, 0.0])
        with torch.no_grad():
            model.role_mix.logits[rows[0]] = torch.tensor([3.0, 0.0, 0.0])
        second = model.level_weights(own, 1, rows)
        first = model.level_weights(own, 0, rows)
        third = model.level_weights(own, 2, rows)
        torch.testing.assert_close(first, own[:, 0, :], atol=0, rtol=0)
        torch.testing.assert_close(third, own[:, 1, :], atol=0, rtol=0)
        self.assertFalse(torch.allclose(first, third))
        self.assertFalse(torch.allclose(second, first),
                         'level 2 must read the role table while levels 1 and 3 read the gate')


if __name__ == '__main__':
    unittest.main()
