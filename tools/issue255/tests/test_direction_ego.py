"""Hand-computed and reconstruction tests for the frozen direction ego asset.

The tiny graph answers are written out by hand; the real asset is checked
against an independent brute-force reconstruction from the frozen CSR.
"""
import json
import unittest
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from tools.issue255.build_direction_ego import (
    ARRAY_NAMES, VARIANTS, audit_variant, build_variant, load_csr, local_csr, sha256,
)

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'stage2_RQ-VAE').is_dir())
BASE = ROOT / 'assets/v10_direction_ego'
GRAPH = ROOT / 'assets/train/visible_forward.npz'
ITEMS = ROOT / 'assets/train/item_ids.npy'


def tiny_graph():
    """0<->1, 0->2, 1->2, 2->3; item 4 is isolated."""
    rows = [0, 0, 1, 1, 2]
    cols = [1, 2, 0, 2, 3]
    data = [2.0, 3.0, 5.0, 7.0, 11.0]
    return csr_matrix((data, (rows, cols)), shape=(5, 5))


def brute_force_token(graph, item, incoming):
    """Recompute one token from the CSR without the builder's own helpers."""
    reverse = graph.T.tocsr()
    source = reverse if incoming else graph
    neighbours = sorted(int(node) for node in
                        source.indices[source.indptr[item]:source.indptr[item + 1]])
    nodes = [item] + neighbours
    position = {node: index for index, node in enumerate(nodes)}
    edges = set()
    for node in nodes:
        for target in graph.indices[graph.indptr[node]:graph.indptr[node + 1]]:
            if int(target) in position:
                edges.add((position[node], position[int(target)]))
    return nodes, edges


class DirectionEgoTests(unittest.TestCase):
    def test_hand_computed_in_only_population(self):
        graph = tiny_graph()
        reverse = graph.T.tocsr()
        arrays, counters = build_variant(graph, reverse, incoming=True)
        offsets = arrays['offsets']
        # Item 0: root plus its in-neighbour 1 -> nodes {0,1}; the induced
        # edges are the reciprocal pair 0->1 and 1->0.
        self.assertEqual(arrays['nodes'][offsets[0]:offsets[1]].tolist(), [0, 1])
        self.assertEqual(int(arrays['out_edge_counts'][0]), 2)
        # Item 1: root plus its in-neighbour 0 -> the root first, then 0.
        self.assertEqual(arrays['nodes'][offsets[1]:offsets[2]].tolist(), [1, 0])
        # Item 2: root plus its in-neighbours 0 and 1 -> the root first, then
        # the sorted neighbours; the induced edges are 0->1, 0->2, 1->0, 1->2.
        self.assertEqual(arrays['nodes'][offsets[2]:offsets[3]].tolist(), [2, 0, 1])
        self.assertEqual(int(arrays['out_edge_counts'][2]), 4)
        # Item 4 is isolated: the token holds the root only and has no edge.
        self.assertEqual(arrays['nodes'][offsets[4]:offsets[5]].tolist(), [4])
        self.assertEqual(int(arrays['in_edge_counts'][4]), 0)
        self.assertEqual(int(arrays['out_edge_counts'][4]), 0)
        self.assertEqual(counters['empty_neighbourhood_items'], 1)

    def test_hand_computed_out_only_population_and_reciprocity(self):
        graph = tiny_graph()
        reverse = graph.T.tocsr()
        arrays, counters = build_variant(graph, reverse, incoming=False)
        offsets = arrays['offsets']
        # Item 0: root plus its out-neighbours 1 and 2 -> nodes {0,1,2}.
        self.assertEqual(arrays['nodes'][offsets[0]:offsets[1]].tolist(), [0, 1, 2])
        self.assertEqual(int(arrays['out_edge_counts'][0]), 4)
        # Item 1 is both an in- and an out-neighbour of item 0: the two token
        # populations are not mutually exclusive.
        in_only, _ = build_variant(graph, reverse, incoming=True)
        self.assertEqual(in_only['nodes'][in_only['offsets'][1]:in_only['offsets'][2]].tolist(), [1, 0])
        self.assertEqual(counters['overlap_items'], 2)
        self.assertEqual(counters['empty_neighbourhood_items'], 2)

    def test_hand_computed_reciprocal_pair_and_overlap_statistics(self):
        """0<->1 is a reciprocal pair inside the induced sets of 0 and 1."""
        graph = tiny_graph()
        reverse = graph.T.tocsr()
        in_only, in_counters = build_variant(graph, reverse, incoming=True)
        out_only, out_counters = build_variant(graph, reverse, incoming=False)
        # In-only node sets: {0,1}, {1,0}, {2,0,1}, {3,2}, {4}. The mutual
        # pair 0<->1 lies inside the first three of them, so three tokens hold
        # one reciprocal pair each.
        self.assertEqual(in_counters['reciprocal_pairs_in_induced_sets'], 3)
        self.assertEqual(in_counters['tokens_with_reciprocal_pairs'], 3)
        # Out-only node sets: {0,1,2}, {1,0,2}, {2,3}, {3}, {4}. The mutual
        # pair 0<->1 lies inside the first two of them only.
        self.assertEqual(out_counters['reciprocal_pairs_in_induced_sets'], 2)
        self.assertEqual(out_counters['tokens_with_reciprocal_pairs'], 2)
        # Items 0 and 1 are each both an in- and an out-neighbour of the
        # other, so two items have a non-empty in/out neighbourhood overlap.
        self.assertEqual(in_counters['overlap_items'], 2)
        self.assertEqual(in_counters['overlap_incidences'], 2)
        self.assertEqual(out_counters['overlap_items'], 2)
        self.assertEqual(out_counters['overlap_incidences'], 2)

    def test_hand_computed_chain_neighbourhood(self):
        """2->3 is the chain tail: item 3 has exactly one in-neighbour."""
        graph = tiny_graph()
        reverse = graph.T.tocsr()
        arrays, _ = build_variant(graph, reverse, incoming=True)
        offsets = arrays['offsets']
        self.assertEqual(arrays['nodes'][offsets[3]:offsets[4]].tolist(), [3, 2])
        self.assertEqual(int(arrays['in_edge_counts'][3]), 1)
        self.assertEqual(int(arrays['out_edge_counts'][3]), 1)

    def test_local_csr_swaps_the_row_and_column_for_the_incoming_direction(self):
        edges = np.array([[0, 1], [2, 1]])
        out_ptr, out_col = local_csr(3, edges, incoming=False)
        in_ptr, in_col = local_csr(3, edges, incoming=True)
        self.assertEqual(out_col.tolist(), [1, 1])
        self.assertEqual(in_col.tolist(), [0, 2])
        self.assertEqual(out_ptr.tolist(), [0, 1, 1, 2])
        self.assertEqual(in_ptr.tolist(), [0, 0, 2, 2])

    def test_local_csr_rejects_a_duplicated_directed_edge(self):
        edges = np.array([[0, 1], [0, 1]])
        with self.assertRaises(ValueError):
            local_csr(2, edges, incoming=False)

    def test_real_asset_matches_an_independent_reconstruction(self):
        graph = load_csr(GRAPH)
        reverse = graph.T.tocsr()
        for variant in VARIANTS:
            directory = BASE / variant
            arrays = {}
            for name in ARRAY_NAMES:
                arrays[name] = np.load(directory / (name + '.npy'), allow_pickle=False)
            report = audit_variant(variant, arrays, graph, reverse)
            self.assertEqual(report['verdict'], 'PASS', variant)
            self.assertEqual(report['mismatches'], 0, variant)
            self.assertEqual(len(arrays['nodes']), int(arrays['offsets'][-1]))
            self.assertTrue(np.array_equal(np.diff(arrays['offsets']), arrays['node_counts']))

    def test_real_asset_sample_matches_a_brute_force_reconstruction(self):
        """A deterministic sample of items is rebuilt from the CSR by hand."""
        graph = load_csr(GRAPH)
        reverse = graph.T.tocsr()
        for variant in VARIANTS:
            directory = BASE / variant
            offsets = np.load(directory / 'offsets.npy', allow_pickle=False)
            nodes = np.load(directory / 'nodes.npy', allow_pickle=False)
            in_ptr = np.load(directory / 'in_ptr.npy', allow_pickle=False)
            in_col = np.load(directory / 'in_col.npy', allow_pickle=False)
            out_ptr = np.load(directory / 'out_ptr.npy', allow_pickle=False)
            out_col = np.load(directory / 'out_col.npy', allow_pickle=False)
            for item in (0, 1, 7, 101, 999, 5000, 12345, 24586):
                incoming = variant == 'in_only'
                source = reverse if incoming else graph
                neighbours = sorted(int(node) for node in
                                    source.indices[source.indptr[item]:source.indptr[item + 1]])
                expected_nodes = [item] + neighbours
                base = int(offsets[item])
                self.assertEqual(nodes[base:base + len(expected_nodes)].tolist(),
                                 expected_nodes, (variant, item))
                position = {node: index for index, node in enumerate(expected_nodes)}
                edges = set()
                for node in expected_nodes:
                    for target in graph.indices[graph.indptr[node]:graph.indptr[node + 1]]:
                        if int(target) in position:
                            edges.add((position[node], position[int(target)]))
                stored_in, stored_out = set(), set()
                for row in range(len(expected_nodes)):
                    for column in in_col[in_ptr[base + row]:in_ptr[base + row + 1]]:
                        stored_in.add((row, int(column)))
                    for column in out_col[out_ptr[base + row]:out_ptr[base + row + 1]]:
                        stored_out.add((row, int(column)))
                self.assertEqual(stored_out, edges, (variant, item))
                self.assertEqual(stored_in, {(column, row) for row, column in edges},
                                 (variant, item))

    def test_real_asset_global_counts_are_consistent(self):
        for variant in VARIANTS:
            directory = BASE / variant
            offsets = np.load(directory / 'offsets.npy', allow_pickle=False)
            counts = np.load(directory / 'node_counts.npy', allow_pickle=False)
            in_ptr = np.load(directory / 'in_ptr.npy', allow_pickle=False)
            out_ptr = np.load(directory / 'out_ptr.npy', allow_pickle=False)
            in_col = np.load(directory / 'in_col.npy', allow_pickle=False)
            out_col = np.load(directory / 'out_col.npy', allow_pickle=False)
            in_edges = np.load(directory / 'in_edge_counts.npy', allow_pickle=False)
            out_edges = np.load(directory / 'out_edge_counts.npy', allow_pickle=False)
            self.assertEqual(offsets.shape, (24588,))
            self.assertEqual(counts.shape, (24587,))
            self.assertEqual(int(offsets[0]), 0)
            self.assertEqual(int(offsets[-1]), counts.sum())
            self.assertTrue(np.array_equal(np.diff(offsets), counts))
            self.assertEqual(in_ptr.shape, (int(offsets[-1]) + 1,))
            self.assertEqual(out_ptr.shape, in_ptr.shape)
            self.assertTrue(np.array_equal(in_edges, in_ptr[offsets[1:]] - in_ptr[offsets[:-1]]))
            self.assertTrue(np.array_equal(out_edges, out_ptr[offsets[1:]] - out_ptr[offsets[:-1]]))
            self.assertEqual(int(in_edges.sum()), len(in_col))
            self.assertEqual(int(out_edges.sum()), len(out_col))
            # Every token keeps its root, so the node count is at least one.
            self.assertTrue((counts >= 1).all())

    def test_asset_metadata_and_manifest_registration(self):
        metadata = json.loads((BASE / 'metadata.json').read_text())
        self.assertEqual(metadata['items'], 24587)
        self.assertEqual(metadata['graph']['sha256'],
                         sha256(ROOT / 'assets/train/visible_forward.npz'))
        self.assertEqual(metadata['item_ids']['sha256'], sha256(ITEMS))
        for variant in VARIANTS:
            for name in ARRAY_NAMES:
                record = metadata['arrays'][variant][name]
                path = ROOT / record['path']
                self.assertEqual(record['sha256'], sha256(path), path)
        manifest = json.loads((ROOT / 'assets/manifest.json').read_text())
        registered = {entry['path']: entry for entry in manifest['assets']}
        for variant in VARIANTS:
            for name in ARRAY_NAMES:
                path = 'assets/v10_direction_ego/%s/%s.npy' % (variant, name)
                self.assertIn(path, registered)
                self.assertEqual(registered[path]['sha256'],
                                 sha256(ROOT / path), path)
        for name in ('metadata', 'coverage', 'audit'):
            path = 'assets/v10_direction_ego/%s.json' % name
            self.assertIn(path, registered)
            self.assertEqual(registered[path]['sha256'],
                             sha256(ROOT / path), path)

    def test_audit_and_coverage_report_a_pass_verdict(self):
        audit = json.loads((BASE / 'audit.json').read_text())
        self.assertEqual(audit['verdict'], 'PASS')
        for check in audit['checks']:
            self.assertEqual(check['verdict'], 'PASS', check['variant'])
            self.assertEqual(check['mismatches'], 0, check['variant'])
            self.assertEqual(check['items_checked'], 24587)
        coverage = json.loads((BASE / 'coverage.json').read_text())
        self.assertEqual(coverage['items'], 24587)
        self.assertIn('exact and complete', coverage['population'])
        for variant in VARIANTS:
            self.assertEqual(coverage['variants'][variant]['nodes'], 285719)
            self.assertEqual(coverage['variants'][variant]['in_edges'], 695377)
            self.assertEqual(coverage['variants'][variant]['out_edges'], 695377)


if __name__ == '__main__':
    unittest.main()
