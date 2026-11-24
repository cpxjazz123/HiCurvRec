import importlib.util
from pathlib import Path
import unittest
import numpy as np
import torch
from tools.graph_infra.ego_gnn import EgoStore

SPEC = importlib.util.spec_from_file_location('structure251', Path(__file__).resolve().parents[1] / 'model/structure.py')
structure = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(structure)


def toy():
    from scipy.sparse import csr_matrix
    s = EgoStore.__new__(EgoStore)
    s.n = 5
    s.forward = csr_matrix(([1., 1., 1.], ([0, 1, 1], [1, 2, 3])), shape=(5, 5))
    s.reverse = s.forward.T.tocsr()
    s.undirected = (s.forward + s.reverse).tocsr()
    s._in_degree = np.diff(s.reverse.indptr)
    s._out_degree = np.diff(s.forward.indptr)
    s._reciprocal = np.zeros(5)
    blocks = []
    for i in range(5):
        nodes = np.union1d([i], s._neighbours(i))
        local = {int(g): j for j, g in enumerate(nodes)}
        edges = np.array([(local[a], local[b]) for a, b in zip(*s.forward.nonzero()) if a in local and b in local], dtype=np.int64).reshape(-1, 2)
        blocks.append((nodes, edges, local[i]))
    s._frozen_ego = {'nodes': np.concatenate([b[0] for b in blocks]), 'edges': np.concatenate([b[1] for b in blocks]), 'node_offsets': np.r_[0, np.cumsum([len(b[0]) for b in blocks])], 'edge_offsets': np.r_[0, np.cumsum([len(b[1]) for b in blocks])], 'root_local': np.array([b[2] for b in blocks])}
    return s


class StructureTests(unittest.TestCase):
    def test_separate_boundaries_duplicates_chunks_isolate_empty(self):
        torch.manual_seed(7)
        model = structure.SubgraphTokens().double()
        store = toy()
        a, valid = model.tokens_for_items(store, [0, 4, 0, 1], 'cpu', chunk=1)
        b, _ = model.tokens_for_items(store, [1, 0, 4], 'cpu', chunk=3)
        torch.testing.assert_close(a, b[[1, 2, 1, 0]])
        self.assertTrue(valid.all())
        self.assertFalse(torch.allclose(a[0, 0], a[0, 1]))
        torch.testing.assert_close(a[1, 0], a[1, 1])
        empty, mask = model.tokens_for_items(store, [], 'cpu')
        self.assertEqual(tuple(empty.shape), (0, 2, 32))
        self.assertEqual(tuple(mask.shape), (0, 2))

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
            model.tokens_for_items(store, [0], 'cpu')
            (root / 'node_features.npz').unlink()
            model.tokens_for_items(store, [0], 'cpu')


if __name__ == '__main__':
    unittest.main()
