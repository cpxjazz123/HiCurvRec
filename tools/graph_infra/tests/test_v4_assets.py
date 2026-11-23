"""Issue #248: asset isolation, manifest tamper detection and query-interface tests.

These tests use the persisted assets when they exist; the ones that need the full build are
skipped with an explicit reason when the asset directory has not been built yet.
"""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

from tools.graph_infra.assets import sha256
from tools.graph_infra.interfaces_v4 import (ManifestError, TrainingView, EvaluationView,
                                             load_manifest, verify_manifest)

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / 'results/research/issue248/assets'
HAVE_BUILD = (ROOT / 'manifest.json').is_file()


class ManifestTests(unittest.TestCase):
    def test_tampered_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=REPO / 'results/research/issue248') as temp:
            root = Path(temp)
            payload = root / 'data.bin'
            payload.write_bytes(b'original')
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'assets': [
                {'name': 'x', 'path': str(payload), 'sha256': sha256(payload), 'use': 'training'}]}))
            loaded = load_manifest(REPO, root)
            verify_manifest(loaded, REPO)
            payload.write_bytes(b'tampered')
            with self.assertRaisesRegex(ManifestError, 'hash mismatch'):
                verify_manifest(load_manifest(REPO, root), REPO)

    def test_missing_file_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=REPO / 'results/research/issue248') as temp:
            root = Path(temp)
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'assets': [
                {'name': 'x', 'path': 'results/research/issue248/does_not_exist.bin',
                 'sha256': '0' * 64, 'use': 'training'}]}))
            with self.assertRaisesRegex(ManifestError, 'missing asset'):
                verify_manifest(load_manifest(REPO, root), REPO)


@unittest.skipUnless(HAVE_BUILD, 'the v4 assets have not been built yet')
class BuiltAssetTests(unittest.TestCase):
    def test_training_view_refuses_evaluation_assets(self):
        training = TrainingView(REPO)
        with self.assertRaisesRegex(ManifestError, 'declared "evaluation"'):
            training.load('held_out_support')
        with self.assertRaisesRegex(ManifestError, 'declared "evaluation"'):
            training.entry('split_users')

    def test_evaluation_view_refuses_training_assets(self):
        evaluation = EvaluationView(REPO)
        with self.assertRaisesRegex(ManifestError, 'not a strict evaluation bank'):
            evaluation.entry('motif_counts')
        with self.assertRaisesRegex(ManifestError, 'not a strict evaluation bank'):
            evaluation.load('visible_forward')

    def test_manifest_hashes_verify(self):
        payload = load_manifest(REPO)
        report = verify_manifest(payload, REPO)
        self.assertGreater(report['verified'], 0)

    def test_full_enumeration_is_complete(self):
        training = TrainingView(REPO)
        table = training.curvature_table()
        degrees = np.diff(training.undirected_projection().indptr)
        expected = degrees * (degrees - 1) // 2
        self.assertEqual(len(table.item_count), len(degrees))
        np.testing.assert_array_equal(table.item_count, expected)
        self.assertEqual(table.total_rows(), int(expected.sum()))

    def test_item_triples_are_contiguous_and_queryable(self):
        training = TrainingView(REPO)
        table = training.curvature_table()
        for item in (0, 5, 1234, 20000):
            rows = table.item_triples(item)
            self.assertEqual(len(rows['b']), int(table.item_count[item]))
            if len(rows['b']):
                self.assertTrue((rows['centre'] == item).all())
                self.assertTrue((rows['b'] < rows['c']).all())
                found = table.triple(item, int(rows['b'][0]), int(rows['c'][0]))
                self.assertAlmostEqual(found['curvature'], float(rows['curvature'][0]), places=12)
                with self.assertRaises(KeyError):
                    table.triple(item, int(rows['c'][0]) + 10 ** 6, 10 ** 6 + 1)

    def test_structure_class_counts_sum_to_the_total(self):
        training = TrainingView(REPO)
        table = training.curvature_table()
        counts = table.structure_class_counts()
        self.assertEqual(sum(counts.values()), table.total_rows())


if __name__ == '__main__':
    unittest.main()
