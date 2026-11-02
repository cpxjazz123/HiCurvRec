"""Row-level and artifact-level invariants for the Issue #137 audit."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for path in (str(REPO), str(HERE.parent),
             str(REPO / "analysis_sid_recall" / "augmented_forman_gap")):
    if path not in sys.path:
        sys.path.insert(0, path)

import af3  # noqa: E402


def build(n: int, edges) -> sp.csr_matrix:
    rows, cols = [], []
    for a, b in edges:
        rows += [a, b]
        cols += [b, a]
    return sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()


class GraphInvariants(unittest.TestCase):
    def test_symmetry_and_af3_identity(self):
        matrix = build(6, [(0, 1), (1, 2), (0, 2), (1, 3), (1, 4), (1, 5)])
        self.assertTrue(np.array_equal(matrix.toarray(), matrix.toarray().T))
        curvature = af3.edge_curvatures(matrix)
        self.assertTrue(np.allclose(curvature["af3"] - curvature["forman"],
                                    3.0 * curvature["triangles"]))
        self.assertEqual(curvature["rows"].size, 6)

    def test_label_shuffle_preserves_histogram_and_degree_buckets(self):
        labels = np.array([0, 0, 1, 1, 2, -1, -1], dtype=np.int64)
        degree = np.array([1.0, 128.0, 4.0, 4.0, 64.0, 0.0, 0.0])
        rng = np.random.default_rng(3)
        for design in ("global", "degree_stratified"):
            for _ in range(20):
                out = af3.shuffle_labels(labels, degree, design, rng)
                self.assertEqual(sorted(out[out >= 0].tolist()),
                                 sorted(labels[labels >= 0].tolist()))
                if design == "degree_stratified":
                    buckets = np.floor(np.log2(np.maximum(degree, 1.0))).astype(int)
                    for bucket in np.unique(buckets[labels >= 0]):
                        mask = (buckets == bucket) & (labels >= 0)
                        self.assertEqual(sorted(out[mask].tolist()),
                                         sorted(labels[mask].tolist()))


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.features = HERE.parent / "features_by_level.csv"
        self.support = HERE.parent / "support.csv"
        if not self.features.is_file():
            self.skipTest("the audit measurement has not been produced yet")

    def test_join_keeps_108_and_two_exclusions(self):
        rows = list(csv.DictReader(self.features.open()))
        completeness = json.loads((HERE.parent / "completeness.json").read_text())
        self.assertEqual(len(rows), 108)
        self.assertEqual(completeness["excluded_hash_mismatch"], 2)
        self.assertEqual(completeness["tiers"]["HASH_PROVEN"], 0)
        self.assertEqual(completeness["tiers"]["RECORD_HASH_MATCH"], 108)
        self.assertEqual(len({row["sid_sha256"] for row in rows}), 108)
        for row in rows:
            for name in ("test_recall@5", "test_recall@10", "test_ndcg@5", "test_ndcg@10"):
                self.assertTrue(row.get(name) not in (None, ""))
            self.assertEqual(row["provenance_tier"], "RECORD_HASH_MATCH")

    def test_edges_are_partitioned_once(self):
        for row in csv.DictReader(self.support.open()):
            self.assertEqual(
                int(row["gap_edges_within"]) + int(row["gap_edges_cross"])
                + int(row["gap_edges_unlabelled"]), int(row["gap_edges_total"]))
            self.assertEqual(row["edge_counts_consistent"], "True")

    def test_semantics_are_distinct_fields(self):
        rows = list(csv.DictReader((HERE.parent / "feature_semantics_audit.csv").open()))
        self.assertTrue(rows)
        differing = [row for row in rows
                     if row["group_mean_negative_fraction"] not in ("", "None")
                     and row["edge_negative_fraction"] not in ("", "None")
                     and abs(float(row["group_mean_negative_fraction"])
                             - float(row["edge_negative_fraction"])) > 1e-9]
        self.assertTrue(differing, "the two negative-fraction measures collapsed")
        for row in rows:
            self.assertIn(row["level"], ("1", "2", "3"))

    def test_primary_family_is_528_potential(self):
        statistics = HERE.parent / "statistics.json"
        if not statistics.is_file():
            self.skipTest("statistics.json has not been produced yet")
        data = json.loads(statistics.read_text())
        primary = data["primary"]
        self.assertEqual(primary["potential"], 528)
        self.assertEqual(primary["defined_tests"], 2 * primary["defined_entries"])
        self.assertEqual(primary["potential"] - primary["defined_tests"],
                         primary["undefined_tests"])
        for entry, block in primary["results"].items():
            if block.get("status") == "DEFINED":
                self.assertIn("p_pearson", block)
                self.assertIn("p_spearman", block)
                self.assertIn("holm_pearson", block)
                self.assertIn("holm_spearman", block)
                self.assertIn("ci_pearson", block)
                self.assertIn("ci_spearman", block)

    def test_the_old_and_new_inference_are_both_shown(self):
        statistics = HERE.parent / "statistics.json"
        if not statistics.is_file():
            self.skipTest("statistics.json has not been produced yet")
        data = json.loads(statistics.read_text())
        self.assertTrue(data["new_vs_old"])
        for entry in data["new_vs_old"]:
            self.assertIn("old_holm", entry)
            self.assertIn("new_holm_pearson", entry)
        report = (HERE.parent / "REPORT.md").read_text()
        self.assertIn("SUPERSEDED", report)
        self.assertIn("bijection", report)
        self.assertIn("with-replacement", report)

    def test_restart_produces_identical_rows(self):
        script = HERE.parent / "run_measure.py"
        outputs = []
        for attempt in range(2):
            directory = Path(tempfile.mkdtemp(prefix=f"audit_resume_{attempt}_"))
            subprocess.run([sys.executable, str(script), "--limit", "1",
                            "--out", str(directory)],
                           check=True, capture_output=True, timeout=600)
            rows = json.loads((directory / "features_by_level.json").read_text())
            outputs.append(rows[0])
        keys = sorted(outputs[0])
        for key in keys:
            self.assertEqual(outputs[0][key], outputs[1][key], msg=key)


if __name__ == "__main__":
    unittest.main()
