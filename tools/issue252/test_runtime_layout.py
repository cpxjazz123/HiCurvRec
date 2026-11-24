"""Hand-computed lossless layout contracts; no generated asset dependency."""
import unittest
import numpy as np


class RuntimeLayoutTests(unittest.TestCase):
    def test_directed_bits_preserve_direction_and_isolation(self):
        from tools.issue252.runtime_layout import pack_directed
        directed=np.array([[0,1,0],[0,0,0],[1,0,0]],dtype=np.int64)
        packed=pack_directed(directed)
        np.testing.assert_array_equal(packed,np.array([[2],[0],[1]],dtype=np.uint8))

    def test_valid_targets_are_lossless_and_duplicate_preserving(self):
        from tools.issue252.runtime_layout import pack_targets
        shard={'centre':np.array([0,0,0,2]),'b':np.array([1,1,2,0]),
               'c':np.array([2,2,1,1]),'valid':np.array([True,True,False,True]),
               'curvature':np.array([-.5,-.5,99.,2.25],dtype=np.float64)}
        rows,offsets=pack_targets([shard],3)
        np.testing.assert_array_equal(offsets,np.array([0,2,2,3],dtype=np.int64))
        np.testing.assert_array_equal(rows['b'],[1,1,0])
        np.testing.assert_array_equal(rows['c'],[2,2,1])
        np.testing.assert_array_equal(rows['target'],[-.5,-.5,2.25])
        self.assertEqual(rows.dtype.names,('b','c','target'))
        self.assertEqual(rows.dtype['b'],np.dtype('int32'))
        self.assertEqual(rows.dtype['target'],np.dtype('float64'))

    def test_byte_boundaries_empty_and_sparse_support(self):
        from scipy.sparse import csr_matrix
        from tools.issue252.runtime_layout import pack_directed
        matrix = np.zeros((10, 10), dtype=np.int64)
        matrix[0, [0, 7, 8, 9]] = [3, -2, 4, 1]
        matrix[9, 0] = 7
        expected = np.zeros((10, 2), dtype=np.uint8)
        expected[0] = [129, 3]
        expected[9] = [1, 0]
        for source in (matrix, csr_matrix(matrix)):
            np.testing.assert_array_equal(pack_directed(source), expected)
            np.testing.assert_array_equal(
                np.unpackbits(pack_directed(source), axis=1, bitorder='little')[:, :10],
                matrix != 0)
        self.assertEqual(pack_directed(np.empty((0, 0))).shape, (0, 0))
        with self.assertRaises(ValueError):
            pack_directed(np.zeros((2, 3)))
        duplicates = csr_matrix((np.array([1, -1, 0, 2]), np.array([1, 1, 2, 0]),
                                 np.array([0, 3, 4, 4])), shape=(3, 3))
        original_data = duplicates.data.copy()
        np.testing.assert_array_equal(pack_directed(duplicates),
                                      np.array([[0], [1], [0]], dtype=np.uint8))
        np.testing.assert_array_equal(duplicates.data, original_data)

    def test_stable_centre_order_and_float64_bits_across_shards(self):
        from tools.issue252.runtime_layout import pack_targets
        values = np.array([0x8000000000000000, 0x3ff0000000000001,
                           0x7ff800000000002a], dtype=np.uint64).view(np.float64)
        first = {'centre': np.array([2, 0, -100]), 'b': np.array([1, 2, -1]),
                 'c': np.array([0, 1, 999]), 'valid': np.array([True, True, False]),
                 'curvature': np.array([values[0], values[1], 99.])}
        second = {'centre': np.array([2]), 'b': np.array([1]), 'c': np.array([0]),
                  'valid': np.array([True]), 'curvature': values[2:]}
        rows, offsets = pack_targets(iter([first, second]), 4)
        np.testing.assert_array_equal(offsets, [0, 1, 1, 3, 3])
        np.testing.assert_array_equal(rows['b'], [2, 1, 1])
        np.testing.assert_array_equal(rows['target'].view(np.uint64),
                                      values[[1, 0, 2]].view(np.uint64))
        empty, offsets = pack_targets([], 0)
        self.assertEqual(len(empty), 0)
        np.testing.assert_array_equal(offsets, [0])
        first['valid'][:] = False
        empty, offsets = pack_targets([first], 4)
        self.assertEqual(len(empty), 0)
        np.testing.assert_array_equal(offsets, np.zeros(5, dtype=np.int64))

    def test_invalid_input_rejected_not_silently_truncated(self):
        from tools.issue252.runtime_layout import pack_targets
        shard = {'centre': np.array([0]), 'b': np.array([1]), 'c': np.array([0]),
                 'valid': np.array([True]), 'curvature': np.array([1.])}
        for n in (-1, 1.5):
            with self.assertRaises(ValueError):
                pack_targets([], n)
        with self.assertRaises(ValueError):
            pack_targets([shard], 1)
        shard['b'] = np.array([0.5])
        with self.assertRaises(ValueError):
            pack_targets([shard], 2)
        shard['b'] = np.array([0])
        shard['curvature'] = np.array([])
        with self.assertRaises(ValueError):
            pack_targets([shard], 2)

    def test_two_hop_nodes_match_egostore_with_isolated_root(self):
        from scipy.sparse import csr_matrix
        from tools.graph_infra.ego_gnn import EgoStore
        from tools.issue252.runtime_layout import closed_two_hop_nodes
        store = EgoStore.__new__(EgoStore)
        store.forward = csr_matrix(([1, 1, 1], ([0, 2, 2], [1, 1, 3])), shape=(5, 5))
        store.reverse = store.forward.T.tocsr()
        store.undirected = (store.forward + store.reverse).tocsr()
        store.n = 5
        store._node_degree = np.diff(store.undirected.indptr)
        store._in_degree = np.diff(store.reverse.indptr)
        store._out_degree = np.diff(store.forward.indptr)
        store._reciprocal = np.zeros(5, dtype=np.int64)
        populations = [np.unique(np.r_[i, store._neighbours(i)]) for i in range(5)]
        store._frozen_ego = {
            'nodes': np.concatenate(populations),
            'node_offsets': np.r_[0, np.cumsum([len(x) for x in populations])]}
        expected = ([0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3], [1, 2, 3], [4])
        for root in range(5):
            nodes, local = closed_two_hop_nodes(store, root)
            reference = store.ego(root, hops=2)
            np.testing.assert_array_equal(nodes, expected[root])
            np.testing.assert_array_equal(nodes, reference['nodes'])
            self.assertEqual(local, reference['root_local'])
            self.assertEqual(nodes[local], root)
        with self.assertRaises(IndexError):
            closed_two_hop_nodes(store, 5)

    def test_streamed_memmap_preserves_every_original_valid_row(self):
        import tempfile
        from pathlib import Path
        from tools.issue252.runtime_layout import _write_target_layout, pack_targets
        shards = [
            {'centre': np.array([2, 0, 2]), 'b': np.array([0, 1, 0]),
             'c': np.array([1, 2, 1]), 'valid': np.array([True, False, True]),
             'curvature': np.array([-0., 99., 1.0000000000000002])},
            {'centre': np.array([1, 2]), 'b': np.array([0, 1]),
             'c': np.array([2, 0]), 'valid': np.array([True, True]),
             'curvature': np.array([2.25, -3.5])}]
        visits = []
        def source():
            for i, shard in enumerate(shards):
                visits.append(i)
                yield shard
        repo = next(p for p in Path(__file__).resolve().parents
                    if (p / 'stage2_RQ-VAE').is_dir())
        with tempfile.TemporaryDirectory(dir=repo / 'results/research/issue252') as folder:
            path = Path(folder)
            _write_target_layout(source, 4, path)
            actual = np.load(path / 'targets.npy', mmap_mode='r')
            offsets = np.load(path / 'target_offsets.npy', mmap_mode='r')
            expected, expected_offsets = pack_targets(shards, 4)
            self.assertIsInstance(actual, np.memmap)
            np.testing.assert_array_equal(offsets, expected_offsets)
            self.assertEqual(actual.tobytes(), expected.tobytes())
            self.assertEqual(visits, [0, 1, 0, 1])

    def test_reused_npz_metadata_never_unpickles_object_labels(self):
        import json
        from pathlib import Path
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        from tools.issue252 import runtime_layout as layout
        with tempfile.TemporaryDirectory(dir=layout.REPO/'results/research/issue252') as folder:
            repo=Path(folder)
            np.savez(repo/'features.npz',features=np.zeros((1,6)),names=np.array(['degree'],dtype=object))
            (repo/'report.json').write_text(json.dumps({'standardization':{}}))
            entries={}
            for name,filename in (('ego_full_1hop_visible','features.npz'),
                                  ('node_features','features.npz'),('node_features_report','report.json'),
                                  ('item_ids','features.npz')):
                path=repo/filename
                entries[name]={'path':filename,'schema':path.suffix[1:],
                               'sha256':layout.sha256(path),'bytes':path.stat().st_size}
            with patch.object(layout,'REPO',repo),patch.object(layout,'OUTPUT_FILES',()):
                result=layout._metadata(entries,{}, {},SimpleNamespace(n=1))
            feature=next(entry for entry in result['assets'] if entry['name']=='node_features')
            self.assertEqual(feature['content']['features']['shape'],[1,6])
            self.assertEqual(feature['content']['names']['dtype'],'|O')
            self.assertIsNone(feature['content']['names']['data_bytes'])


if __name__=='__main__':unittest.main()
