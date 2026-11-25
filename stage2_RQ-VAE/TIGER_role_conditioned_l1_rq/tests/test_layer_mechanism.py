"""Mechanism-level contracts for the layer-wise gate and the prefix supervision.

The prefix identities, the per-level routing and the gradient isolation are
checked on the production path with the real codebooks and the real gate.
"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))


class LayerMechanismTests(unittest.TestCase):
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
        # The layer-wise gate is new in this tree, so the archived single-level
        # attention weights are not part of its state.
        state = {key: value for key, value in self.initial.items()
                 if not key.startswith('structure.attention.')}
        model.load_state_dict(state, strict=False)
        model.freeze_tangent_scale(self.embeddings)
        return model

    def test_four_token_types_are_produced_and_typed(self):
        model = self.model().eval()
        ids = torch.tensor([1234, 20000])
        model(self.embeddings[ids], item_ids=ids)
        tokens = model.structure.last_tokens
        self.assertEqual(tuple(tokens.shape), (2, 4, 32))
        mask = model.structure.last_token_mask
        self.assertEqual(tuple(mask.shape), (2, 5))
        # The four token slots differ: the fixed type encoding separates them.
        differences = (tokens[0, 0] - tokens[0, 1]).abs().sum()
        self.assertGreater(float(differences), 0.0)

    def test_hard_prefix_equals_the_selected_codewords_bitwise(self):
        model = self.model().eval()
        ids = torch.tensor([1234, 20000, 5])
        model(self.embeddings[ids], item_ids=ids)
        prefixes = model.last_prefixes
        self.assertEqual(tuple(prefixes.shape), (3, 3, 32))
        running = torch.zeros_like(prefixes[0])
        for level, layer in enumerate(model.rq.vq_layers):
            selected = layer.embed.weight[model.last_tokens[:, level]]
            running = running + selected
            self.assertTrue(torch.equal(prefixes[level], running),
                            'the cumulative prefix must be the exact running sum')

    def test_earlier_prefix_cannot_depend_on_later_codebooks(self):
        model = self.model().eval()
        ids = torch.tensor([1234, 20000, 5])
        model(self.embeddings[ids], item_ids=ids)
        prefixes = model.last_prefixes
        for later in (1, 2):
            book = model.rq.vq_layers[later].embed.weight
            gradient = torch.autograd.grad(prefixes[0].sum(), book, allow_unused=True,
                                           retain_graph=True)[0]
            self.assertTrue(gradient is None or float(gradient.abs().sum()) == 0.0,
                            'an early prefix must not receive gradient from a later codebook')
        book = model.rq.vq_layers[2].embed.weight
        gradient = torch.autograd.grad(prefixes[2].sum(), book, allow_unused=True,
                                       retain_graph=True)[0]
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)


    def test_level_weights_change_only_their_own_cost(self):
        model = self.model().eval()
        ids = torch.tensor([1234, 20000])
        model(self.embeddings[ids], item_ids=ids)
        weights = model.last_weights
        # R03: the gate owns the two learned levels; the first level's mixture
        # is the item's graphlet role.
        self.assertEqual(tuple(weights.shape), (2, 2, 3))
        costs = [cost.detach().clone() for cost in model.last_costs]
        with torch.no_grad():
            model.structure.attention.query[0] += 0.75
        model(self.embeddings[ids], item_ids=ids)
        new_costs = model.last_costs
        self.assertTrue(torch.allclose(costs[0], new_costs[0].detach(), atol=1e-6),
                        'the first level cost is the role mixture and cannot follow the gate')
        self.assertFalse(torch.allclose(costs[1], new_costs[1].detach()),
                         'the second level cost must follow the second level weights')

    def test_two_identical_instances_agree_on_the_loss(self):
        first = self.model().eval()
        second = self.model().eval()
        ids = torch.tensor([1234, 20000])
        first_out = first(self.embeddings[ids], item_ids=ids)
        second_out = second(self.embeddings[ids], item_ids=ids)
        torch.testing.assert_close(first_out[0], second_out[0], atol=0, rtol=0)
        torch.testing.assert_close(first.last_weights, second.last_weights, atol=0, rtol=0)
        torch.testing.assert_close(first.last_prefixes, second.last_prefixes, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
