"""The level-1 curvature mixture must come from the graphlet role, not the item.

These are the R03 acceptance tests: the role label is a training-only
behaviour signal, the level-1 cost is measured in that role's geometry, the two
switches are wired, and the level-1 cost of an item never depends on which
other items share its batch.
"""
from pathlib import Path
import json
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))


class RoleMixtureTests(unittest.TestCase):
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

    def test_role_label_is_training_only_and_carries_its_provenance(self):
        from model.role_mix import RoleTable
        table = RoleTable()
        table.check_training_only()
        self.assertEqual(table.n_items, 24587)
        self.assertGreater(table.n_roles, 1)
        roles = table.roles_for(np.arange(24587))
        self.assertEqual(roles.shape, (24587,))
        self.assertTrue((roles >= -1).all())
        self.assertTrue((roles < table.n_roles).any())
        payload = json.loads((TREE / 'role_table.json').read_text(encoding='utf-8'))
        self.assertEqual(payload['source']['sha256'], table.source_sha256)
        self.assertIn('training', payload['split_rule'])

    def test_role_mixture_has_one_normalised_row_per_role(self):
        from model.role_mix import RoleMixture
        head = RoleMixture()
        weights = head.weights()
        # One row per role, one for the items without a role and one global row.
        self.assertEqual(tuple(weights.shape), (head.table.n_roles + 2, 3))
        torch.testing.assert_close(weights.sum(-1), torch.ones(weights.shape[0]), atol=1e-6, rtol=0)
        self.assertTrue((weights >= 0).all())

    def test_level_one_uses_the_role_mixture_and_the_other_levels_the_item_mixture(self):
        model = self.model()
        ids = np.array([1234, 20000, 5])
        rows = model.role_mix.rows_for(ids, torch.device('cpu'))
        own = torch.randn(3, 2, 3)
        level_one = model.level_weights(own, 0, rows)
        torch.testing.assert_close(level_one, model.role_mix.weights()[rows])
        for level in (1, 2):
            torch.testing.assert_close(model.level_weights(own, level, rows), own[:, level - 1, :])

    def test_items_of_one_role_share_the_level_one_mixture(self):
        model = self.model()
        table = model.role_mix.table
        roles = table.roles_for(np.arange(24587))
        first = int(np.flatnonzero(roles == roles[0])[0])
        ids = np.array([0, first])
        rows = model.role_mix.rows_for(ids, torch.device('cpu'))
        own = torch.randn(2, 2, 3)
        level_one = model.level_weights(own, 0, rows)
        torch.testing.assert_close(level_one[0], level_one[1])

    def test_behaviour_switch_off_uses_one_global_mixture(self):
        model = self.model()
        ids = np.array([1234, 20000])
        rows = model.role_mix.rows_for(ids, torch.device('cpu'))
        own = torch.randn(2, 2, 3)
        model.role_mixture_on = False
        level_one = model.level_weights(own, 0, rows)
        torch.testing.assert_close(level_one[0], level_one[1])
        torch.testing.assert_close(level_one[0], model.role_mix.global_weights())

    def test_curve_switch_off_makes_the_level_one_cost_euclidean(self):
        model = self.model()
        ids = np.array([1234, 20000])
        rows = model.role_mix.rows_for(ids, torch.device('cpu'))
        own = torch.randn(2, 2, 3)
        residual = torch.randn(2, 32)
        codewords = torch.randn(256, 32)
        with torch.no_grad():
            model.role_mix.logits[rows[0]] = torch.tensor([3.0, 0.0, 0.0])
            model.role_mix.logits[rows[1]] = torch.tensor([0.0, 3.0, 0.0])
            mixed = model.assignment_cost(residual, codewords, model.level_weights(own, 0, rows))
            model.curvature_on = False
            plain = model.assignment_cost(residual, codewords, model.level_weights(own, 0, rows))
        self.assertFalse(torch.allclose(mixed, plain))
        # With the curvature off, every expert is the plain Euclidean distance
        # in the common tangent scale, so the mixture cannot change the cost.
        scale = float(model._tangent_scale_value)
        euclidean = torch.cdist(residual / scale, codewords / scale).square()
        torch.testing.assert_close(plain, euclidean, atol=1e-5, rtol=1e-5)

    def test_level_one_cost_of_an_item_does_not_depend_on_its_batch_companions(self):
        model = self.model().eval()
        first, second = 1234, 20000
        with torch.no_grad():
            rows = model.role_mix.rows_for(np.array([first]), torch.device('cpu'))
            residual = model.encoder(self.embeddings[[first]])[0]
            codewords = model.rq.vq_layers[0].embed.weight
            alone = model.assignment_cost(residual, codewords,
                                          model.level_weights(torch.zeros(1, 3, 3), 0, rows))
            rows_batch = model.role_mix.rows_for(np.array([first, second]), torch.device('cpu'))
            weights_batch = torch.zeros(2, 2, 3)
            weights_batch[1] = 1.0
            batch = model.assignment_cost(residual, codewords,
                                          model.level_weights(weights_batch, 0, rows_batch)[0:1])
        torch.testing.assert_close(alone, batch, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
