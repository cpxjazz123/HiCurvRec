import importlib.util
from pathlib import Path
import unittest
import numpy as np
import torch

SPEC = importlib.util.spec_from_file_location('structure251', Path(__file__).resolve().parents[1] / 'model/structure.py')
structure = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(structure)

FIXTURES_SPEC = importlib.util.spec_from_file_location(
    'structure255_fixtures', Path(__file__).with_name('test_static_means.py'))
fixtures = importlib.util.module_from_spec(FIXTURES_SPEC)
FIXTURES_SPEC.loader.exec_module(fixtures)


def toy():
    # Reciprocal 0<->1, an induced neighbour-neighbour edge 2->3, and isolate 4.
    return fixtures.TinyStore(n=5, edges=((0, 1), (1, 0), (1, 2), (1, 3), (2, 3)))


class StructureTests(unittest.TestCase):
    def test_separate_boundaries_duplicates_chunks_isolate_empty(self):
        torch.manual_seed(7)
        model = structure.SubgraphTokens().double()
        store = toy()
        fixtures.four_token_fixture(structure, model, store)
        a, valid = model.tokens_for_items(store, [0, 4, 0, 1], 'cpu', chunk=1)
        b, _ = model.tokens_for_items(store, [1, 0, 4], 'cpu', chunk=3)
        torch.testing.assert_close(a, b[[1, 2, 1, 0]])
        self.assertEqual(tuple(a.shape), (4, 4, 32))
        self.assertEqual(tuple(valid.shape), (4, 4))
        self.assertTrue(valid.all())
        encoding = fixtures.TYPE_ENCODING.to(a)
        torch.testing.assert_close(model.type_encoding, encoding, atol=0, rtol=0)
        self.assertEqual(structure.TOKEN_TYPES,
                         ('one_hop', 'two_hop', 'in_only', 'out_only'))
        self.assertNotIn('type_encoding', dict(model.named_parameters()))
        raw = a - encoding
        self.assertFalse(torch.allclose(raw[0, 0], raw[0, 1]))
        # An isolate has four identical raw root readouts, not identical types.
        for token in range(4):
            torch.testing.assert_close(raw[1, token], raw[1, 0])
        torch.testing.assert_close(a[1] - a[1, 0], encoding - encoding[0])
        # Root 2 distinguishes the full radius-two ego from both directions.
        populations = ([1, 2, 3], [0, 1, 2, 3], [1, 2], [2, 3])
        edge_sets = ({(1, 2), (1, 3), (2, 3)},
                     {(0, 1), (1, 0), (1, 2), (1, 3), (2, 3)},
                     {(1, 2)}, {(2, 3)})
        for hops, nodes, expected_edges in zip(fixtures.TOKEN_POPULATIONS, populations, edge_sets):
            batch = store.collate([2], hops=hops)
            self.assertEqual(batch['nodes'].tolist(), nodes)
            self.assertEqual(nodes.count(2), 1)
            self.assertEqual({tuple(batch['nodes'][edge]) for edge in batch['edges']}, expected_edges)
            isolated = store.collate([4], hops=hops)
            self.assertEqual(isolated['nodes'].tolist(), [4])
            self.assertEqual(isolated['edges'].shape, (0, 2))
        for hops in ('in_only', 'out_only'):
            reciprocal = store.collate([0], hops=hops)
            self.assertEqual(reciprocal['nodes'].tolist(), [0, 1])
            self.assertEqual({tuple(reciprocal['nodes'][edge]) for edge in reciprocal['edges']},
                             {(0, 1), (1, 0)})
        empty, mask = model.tokens_for_items(store, [], 'cpu')
        self.assertEqual(tuple(empty.shape), (0, 4, 32))
        self.assertEqual(tuple(mask.shape), (0, 4))

    def test_directed_hand_readout_and_padding(self):
        model = structure.SubgraphTokens().double()
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
            model.input.weight[0, 0] = 1
            for layer in model.layers:
                layer.incoming.weight[0, 0] = 1
        batch = {'items': [9], 'features': [[1.,0,0,0,0,0], [2.,0,0,0,0,0], [999.,0,0,0,0,0]], 'nodes': [9, 10, -1], 'batch_index': [0,0,0], 'node_mask': [True,True,False], 'edges': [[0,1],[1,0],[2,0]], 'edge_mask': [True,False,True]}
        # First incoming layer: [0,tanh(1)]; second: [0,0].
        torch.testing.assert_close(model(batch), torch.zeros(1,32,dtype=torch.double))
        batch['edges'] = [[0,1],[1,1],[2,0]]
        batch['edge_mask'] = [True,True,True]
        expected = torch.tanh(torch.tanh(torch.tensor(1.5,dtype=torch.double)) / 2) / 2
        self.assertAlmostEqual(float(model(batch)[0,0]), float(expected), places=12)

    def test_initial_feature_cache_real_root(self):
        import tempfile
        import json
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root = Path(directory)
            features = np.arange(30).reshape(5,6).astype(float)
            np.savez(root / 'node_features.npz', features=features)
            (root / 'node_features_report.json').write_text(json.dumps({'standardization': {'mean': [1]*6, 'std': [2]*6}}))
            store = toy()
            store.root = root
            store.node_features = lambda: (_ for _ in ()).throw(AssertionError('refit'))
            store.standardization = store.node_features
            model = structure.SubgraphTokens()
            fixtures.four_token_fixture(structure, model, store)
            model.tokens_for_items(store, [0], 'cpu')
            (root / 'node_features.npz').unlink()
            model.tokens_for_items(store, [0], 'cpu')


if __name__ == '__main__':
    unittest.main()
