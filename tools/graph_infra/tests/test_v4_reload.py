"""Issue #248 C11: reload from the persisted assets only, then scan, query and merge.

The test never regenerates anything: it opens the saved files and checks that a fresh process
finds the same rows, the same hashes and the same answers.
"""
import json
import subprocess
import sys
import unittest
from pathlib import Path
import numpy as np

from tools.graph_infra.assets import sha256
from tools.graph_infra.full_assets import FullAssets
from tools.graph_infra.interfaces_v4 import TrainingView, EvaluationView, load_manifest, verify_manifest

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / 'results/research/issue248/assets'
HAVE_BUILD = (ROOT / 'manifest.json').is_file()


@unittest.skipUnless(HAVE_BUILD, 'the v4 assets have not been built yet')
class ReloadTests(unittest.TestCase):
    def test_fresh_process_sees_the_same_hashes(self):
        script = ('from tools.graph_infra.interfaces_v4 import load_manifest, verify_manifest;'
                  'from pathlib import Path;'
                  "p = load_manifest(Path('.'));"
                  "print(verify_manifest(p, Path('.'))['verified'])")
        done = subprocess.run([sys.executable, '-c', script], cwd=REPO,
                              capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        expected = len(load_manifest(REPO)['assets'])
        self.assertEqual(int(done.stdout.strip().splitlines()[-1]), expected)

    def test_shard_merge_reproduces_the_global_counts(self):
        assets = FullAssets(REPO)
        rows = 0
        per_class = {}
        for name in assets.shard_names():
            blob = assets.shard(name)
            rows += len(blob['b'])
            for class_id, count in zip(*np.unique(blob['structure_class'], return_counts=True)):
                per_class[int(class_id)] = per_class.get(int(class_id), 0) + int(count)
        self.assertEqual(rows, assets.total_rows())
        published = assets.structure_class_counts()
        merged = {int(name.split('_')[1]): count for name, count in published.items()
                  if count > 0}
        self.assertEqual(per_class, merged)

    def test_random_access_is_consistent_with_the_full_scan(self):
        assets = FullAssets(REPO)
        rng = np.random.default_rng(248009)
        for _ in range(200):
            item = int(rng.integers(0, len(assets.item_count)))
            rows = assets.item_triples(item)
            self.assertEqual(len(rows['b']), int(assets.item_count[item]))
            if len(rows['b']):
                position = int(rng.integers(0, len(rows['b'])))
                found = assets.triple(item, int(rows['b'][position]), int(rows['c'][position]))
                self.assertAlmostEqual(found['curvature'], float(rows['curvature'][position]),
                                       places=12)
                self.assertEqual(found['shard'] if 'shard' in found else rows['shard'], rows['shard'])

    def test_structure_class_rows_sum_to_the_class_count(self):
        assets = FullAssets(REPO)
        counts = assets.structure_class_counts()
        for name, count in counts.items():
            class_id = int(name.split('_')[1])
            rows = assets.rows_of_structure_class(class_id)
            self.assertEqual(len(rows['b']), count)

    def test_manifest_declares_use_and_origin_for_every_asset(self):
        payload = load_manifest(REPO)
        for entry in payload['assets']:
            self.assertIn(entry['use'], ('training', 'evaluation', 'description'))
            self.assertIn(entry['origin'], ('built by issue #248', 'reused by immutable hash'))
            self.assertGreater(entry['bytes'], 0)
            self.assertEqual(len(entry['sha256']), 64)

    def test_training_and_evaluation_views_are_separated(self):
        training = TrainingView(REPO)
        evaluation = EvaluationView(REPO)
        strict = [name for name in training.payload['by_name'] if name.startswith('strict_')]
        self.assertTrue(strict, 'the strict banks are referenced')
        for name in strict:
            with self.assertRaises(Exception):
                training.load(name)
            evaluation.load(name)  # the evaluation view may read them


if __name__ == '__main__':
    unittest.main()
