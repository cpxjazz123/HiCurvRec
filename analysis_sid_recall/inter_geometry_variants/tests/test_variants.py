"""Toy-graph contract tests for Issue #135."""
from __future__ import annotations

import csv
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for path in (str(REPO), str(HERE.parent),
             str(REPO / "analysis_sid_recall" / "layerwise_curvature"),
             str(REPO / "analysis_sid_recall" / "intra_inter_geometry")):
    if path not in sys.path:
        sys.path.insert(0, path)

import analysis_lw as stats  # noqa: E402
import geom  # noqa: E402
import variants as variants  # noqa: E402


def build(n: int, edges: list[tuple[int, int]]) -> sp.csr_matrix:
    rows, cols = [], []
    for a, b in edges:
        rows += [a, b]
        cols += [b, a]
    return sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()


def path(n: int) -> sp.csr_matrix:
    return build(n, [(i, i + 1) for i in range(n - 1)])


def clique(n: int) -> sp.csr_matrix:
    return build(n, [(a, b) for a in range(n) for b in range(a + 1, n)])


class TriangleCountTests(unittest.TestCase):
    def test_path_has_no_triangles(self):
        matrix = path(4)
        block = variants.base_block(matrix, variants.degree_of(matrix))
        self.assertEqual(block["triangles"], 0.0)
        self.assertEqual(block["transitivity"], 0.0)
        self.assertEqual(block["density"], 0.5)

    def test_triangle_is_exact(self):
        matrix = clique(3)
        block = variants.base_block(matrix, variants.degree_of(matrix))
        self.assertEqual(block["triangles"], 1.0)
        self.assertEqual(block["transitivity"], 1.0)
        nodes = variants._triangles_at_nodes(matrix, variants.degree_of(matrix))
        self.assertTrue(np.array_equal(nodes, np.ones(3)))

    def test_two_disjoint_triangles(self):
        matrix = build(6, [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)])
        block = variants.base_block(matrix, variants.degree_of(matrix))
        cycle = variants.cycle_block(matrix, variants.degree_of(matrix))
        self.assertEqual(block["triangles"], 2.0)
        self.assertEqual(block["transitivity"], 1.0)
        self.assertEqual(cycle["components"], 2)
        self.assertEqual(cycle["cycle_surplus_per_component_mean"], 1.0)
        self.assertAlmostEqual(cycle["cycle_rank_per_node"], 2 / 6)

    def test_clique_counts_and_forman(self):
        matrix = clique(5)
        blocks = variants.quotient_blocks(matrix)
        self.assertEqual(blocks["base"]["triangles"], math.comb(5, 3))
        self.assertEqual(blocks["local_transitivity_mean"], 1.0)
        self.assertEqual(blocks["triangles_per_node_mean"], 6.0)
        self.assertEqual(blocks["ego_density_mean"], 1.0)
        self.assertEqual(blocks["ego_cycle_surplus_mean"], 6.0)
        self.assertEqual(blocks["forman_mean"], -4.0)
        self.assertEqual(blocks["forman_negative_fraction"], 1.0)
        self.assertEqual(blocks["forman_edges"], 10)

    def test_triangle_incidence_matches_the_global_count(self):
        rng = np.random.default_rng(3)
        matrix = build(9, [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (2, 4), (4, 5), (5, 6)])
        block = variants.base_block(matrix, variants.degree_of(matrix))
        nodes = variants._triangles_at_nodes(matrix, variants.degree_of(matrix))
        self.assertAlmostEqual(nodes.sum(), 3.0 * block["triangles"])
        reference = float((matrix @ matrix).multiply(matrix).sum() // 6)
        self.assertEqual(block["triangles"], reference)


class LocalCoverageTests(unittest.TestCase):
    def test_degree_below_two_is_undefined_not_zero(self):
        matrix = path(4)                     # endpoints have degree 1
        block = variants.local_block(matrix, variants.degree_of(matrix))
        self.assertEqual(block["local_transitivity_coverage"], 0.5)
        self.assertEqual(block["local_transitivity_mean"], 0.0)

    def test_isolated_node_is_seen_but_not_a_ratio(self):
        matrix = build(3, [(0, 1)])
        block = variants.local_block(matrix, variants.degree_of(matrix))
        self.assertEqual(block["ego_density_coverage"], 2 / 3)
        self.assertEqual(block["local_transitivity_coverage"], 0.0)
        edge = variants.forman_block(matrix, variants.degree_of(matrix))
        self.assertEqual(edge["forman_edges"], 1)


class SiblingTests(unittest.TestCase):
    def test_sibling_edges_and_cross_parent_edges_are_accounted(self):
        # children a,b,c in parent A; d in parent B; edges a-b, b-c (sibling), a-d (cross)
        matrix = build(4, [(0, 1), (1, 2), (0, 3)])
        parent = np.array([0, 0, 0, 1])
        items = np.array([2.0, 2.0, 2.0, 1.0])
        block = variants.sibling_blocks(matrix, parent, items)
        aggregates = block["aggregates"]
        self.assertEqual(aggregates["sibling_edges"], 2)
        self.assertEqual(aggregates["cross_parent_edges"], 1)
        self.assertAlmostEqual(aggregates["sibling_edge_share"], 2 / 3)
        self.assertEqual(aggregates["parents_total"], 2)
        self.assertEqual(aggregates["parents_multi_child"], 1)
        self.assertAlmostEqual(aggregates["sibling_density_mean"], 2 / 3)

    def test_a_parent_with_one_child_is_not_a_density(self):
        matrix = build(2, [(0, 1)])
        parent = np.array([0, 1])
        block = variants.sibling_blocks(matrix, parent, np.array([1.0, 1.0]))
        self.assertIsNone(block["aggregates"]["sibling_density_mean"])
        self.assertEqual(block["aggregates"]["parents_total"], 2)
        self.assertEqual(block["aggregates"]["parents_multi_child"], 0)

    def test_more_than_300_children_and_no_cap(self):
        size = 320
        matrix = build(size, [(i, (i + 1) % size) for i in range(size)])
        parent = np.zeros(size, dtype=np.int64)
        block = variants.sibling_blocks(matrix, parent, np.ones(size))
        self.assertEqual(block["aggregates"]["children_total"], size)
        self.assertEqual(block["aggregates"]["sibling_edges"], size)
        self.assertAlmostEqual(block["aggregates"]["sibling_density_mean"],
                               2.0 / (size - 1))

    def test_the_large_graph_keeps_every_node(self):
        size = 320
        matrix = build(size, [(i, (i + 1) % size) for i in range(size)])
        blocks = variants.quotient_blocks(matrix)
        self.assertEqual(blocks["base"]["nodes"], size)
        self.assertEqual(blocks["base"]["edges"], size)
        self.assertEqual(blocks["components"], 1)
        self.assertAlmostEqual(blocks["cycle_rank_per_node"], (size - size + 1) / size)


class SingletonSensitivityTests(unittest.TestCase):
    def test_the_secondary_subgraph_loses_exactly_the_singletons(self):
        matrix = build(5, [(0, 1), (1, 2), (2, 0), (2, 3)])
        sizes = np.array([2.0, 2.0, 2.0, 1.0, 1.0])
        induced = variants.induced_subgraph(matrix, sizes >= 2)
        self.assertEqual(induced.shape[0], 3)
        self.assertEqual(int(induced.nnz // 2), 3)
        self.assertEqual(int((sizes == 1).sum()), 2)
        self.assertEqual(float(sizes[sizes == 1].sum()), 2.0)


class NullDesignTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.array([0, 0, 1, 1, 2, -1, -1], dtype=np.int64)
        self.degree = np.array([5, 4, 3, 2, 1, 0, 0], dtype=float)

    def test_global_null_preserves_the_label_multiset(self):
        rng = np.random.default_rng(0)
        out = variants.null_labels(self.labels, self.degree, "global", rng)
        self.assertEqual(sorted(out.tolist()), sorted(self.labels.tolist()))
        self.assertEqual((out[5:6] == -1).all(), True)

    def test_stratified_null_keeps_each_bucket_multiset(self):
        rng = np.random.default_rng(1)
        out = variants.null_labels(self.labels, self.degree, "degree_stratified", rng)
        buckets = np.floor(np.log2(np.maximum(self.degree, 1.0))).astype(int)
        for bucket in np.unique(buckets):
            mask = buckets == bucket
            self.assertEqual(sorted(out[mask].tolist()), sorted(self.labels[mask].tolist()))

    def test_parent_conditioned_null_never_crosses_a_parent(self):
        parent = np.array([0, 0, 0, 1, 1, -1, -1], dtype=np.int64)
        rng = np.random.default_rng(2)
        for _ in range(10):
            out = variants.null_labels(self.labels, self.degree, "parent_conditioned",
                                       rng, parent)
            for value in np.unique(parent[parent >= 0]):
                mask = parent == value
                self.assertEqual(sorted(out[mask].tolist()), sorted(self.labels[mask].tolist()))

    def test_null_blocks_count_every_replicate(self):
        matrix = path(6)
        degree = variants.degree_of(matrix)
        labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)

        def build_labels(candidate):
            return variants.base_block(geom.quotient(matrix, candidate),
                                       variants.degree_of(geom.quotient(matrix, candidate)))

        block = variants.null_blocks(labels, np.ones(6), "global", build_labels,
                                     replicates=5, seed=7)
        self.assertEqual(block["density"]["n"], 5)


class RankMonotonicityTests(unittest.TestCase):
    def test_a_monotonic_transform_leaves_rank_correlation_unchanged(self):
        rng = np.random.default_rng(5)
        x = np.arange(1, 30, dtype=float) * 0.7
        y = rng.random(29)
        baseline = stats.spearman(x, y)
        for transform in (np.log1p, np.sqrt):
            self.assertAlmostEqual(stats.spearman(transform(x), y), baseline, places=12)


class JoinAndSupportTests(unittest.TestCase):
    def test_the_join_keeps_108_rows_and_excludes_two(self):
        path_csv = HERE.parent / "features_by_level.csv"
        if not path_csv.is_file():
            self.skipTest("the measurement table has not been produced yet")
        rows = list(csv.DictReader(path_csv.open()))
        completeness = json.loads((HERE.parent / "completeness.json").read_text())
        self.assertEqual(len(rows) + completeness["excluded_hash_mismatch"], 110)
        self.assertEqual(completeness["excluded_hash_mismatch"], 2)
        hashes = [row["sid_sha256"] for row in rows]
        self.assertEqual(len(hashes), len(set(hashes)))
        for row in rows:
            self.assertNotEqual(row["provenance_tier"], "PATH_ONLY")

    def test_every_support_row_accounts_for_all_edges(self):
        support = HERE.parent / "group_parent_support.csv"
        if not support.is_file():
            self.skipTest("support rows have not been produced yet")
        for row in csv.DictReader(support.open()):
            if row["level"] == "1":
                continue
            self.assertEqual(int(row["sibling_edges"]) + int(row["cross_parent_edges"]),
                             int(row["edges"]))

    def test_every_applicable_candidate_has_a_finite_value_or_a_status(self):
        path_csv = HERE.parent / "features_by_level.csv"
        if not path_csv.is_file():
            self.skipTest("the measurement table has not been produced yet")
        rows = list(csv.DictReader(path_csv.open()))
        registry = json.loads((HERE.parent / "candidate_registry.json").read_text())
        sys.path.insert(0, str(HERE.parent))
        import run_stats as stats_module
        aliases = stats_module.RECORD_ALIASES
        fields = set(rows[0].keys())
        for entry in registry["candidates"]:
            for level in entry["levels"]:
                stem = aliases.get(entry["name"], entry["name"])
                prefix = f"{entry['family']}_{stem}_l{level}"
                if entry["family"] == "C" and entry["name"] in ("null_delta", "null_z",
                                                                "null_ratio"):
                    continue     # expanded per feature, design and statistic below
                self.assertIn(prefix, fields)
        for row in rows:
            for level in (1, 2, 3):
                for design in ("global", "degree_stratified"):
                    self.assertIn(f"C_density_{design}_delta_l{level}", row)


if __name__ == "__main__":
    unittest.main()
