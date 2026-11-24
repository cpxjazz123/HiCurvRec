"""Final metrics use the pre-training manifest, not the pre-optimization intake."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools.issue252.archive_audit import sha256
from tools.issue252.final_evaluation import _frozen_inputs

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'tools/research/stage2/contract.py').is_file())


class FinalInputGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / 'results/research/issue252')
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name)
        self.assets = self.repo / 'assets'
        self.assets.mkdir()
        self.train = self.repo / 'train.parquet'
        self.embeddings = self.repo / 'embeddings.npy'
        self.train.write_bytes(b'fixed training interactions')
        self.embeddings.write_bytes(b'fixed embeddings')
        self.leaf = self.assets / 'registered.npy'
        self.leaf.write_bytes(b'fixed registered data')
        self.manifest = self.assets / 'manifest.json'
        entry = {'path': 'assets/registered.npy', 'sha256': sha256(self.leaf)}
        self.manifest.write_text(json.dumps({'assets': [entry]}))
        self.intake_digest = sha256(self.manifest)
        acceptance = self.repo / 'results/research/issue252/ACCEPTANCE.json'
        acceptance.parent.mkdir(parents=True)
        acceptance.write_text(json.dumps({'inputs': {
            'train': {'path': 'train.parquet', 'sha256': sha256(self.train)},
            'embeddings': {'path': 'embeddings.npy', 'sha256': sha256(self.embeddings)},
            'assets_manifest': {'path': 'assets/manifest.json', 'sha256': self.intake_digest}}}))
        # Registered metadata evolved during authorized engineering, before training.
        self.manifest.write_text(json.dumps({'assets': [entry], 'version': 2}))
        self.provenance = {'asset_manifest_sha256': sha256(self.manifest),
                           'training_inputs': {'train.parquet': sha256(self.train)},
                           'assets': {'assets/registered.npy': sha256(self.leaf)}}

    def load(self, provenance=None):
        return _frozen_inputs(self.repo, self.assets, self.embeddings,
                              self.provenance if provenance is None else provenance)

    def test_registered_generation_bound_before_training_is_reported_explicitly(self):
        report, manifest = self.load()
        self.assertEqual(manifest['version'], 2)
        self.assertEqual(report['inputs']['assets_manifest']['sha256'], sha256(self.manifest))
        self.assertEqual(report['intake_asset_manifest_sha256'], self.intake_digest)
        self.assertEqual(report['asset_manifest_authority'], 'checkpoint_pre_training_provenance')

    def test_raw_training_or_embedding_changes_still_fail(self):
        for path in (self.train, self.embeddings):
            original = path.read_bytes()
            path.write_bytes(original + b'changed')
            try:
                with self.subTest(path=path.name), self.assertRaisesRegex(ValueError, 'frozen.*input hash'):
                    self.load()
            finally:
                path.write_bytes(original)

    def test_manifest_changed_after_training_or_missing_binding_still_fails(self):
        original = self.manifest.read_text()
        self.manifest.write_text(original + '\n')
        with self.assertRaisesRegex(ValueError, 'manifest'):
            self.load()
        self.manifest.write_text(original)
        provenance = copy.deepcopy(self.provenance)
        del provenance['asset_manifest_sha256']
        with self.assertRaisesRegex(ValueError, 'manifest'):
            self.load(provenance)

    def test_registered_leaf_changed_after_training_still_fails(self):
        self.leaf.write_bytes(b'changed asset')
        with self.assertRaisesRegex(ValueError, 'asset'):
            self.load()


if __name__ == '__main__':
    unittest.main()
