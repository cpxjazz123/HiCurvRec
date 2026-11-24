"""The export manifest must describe the table that is on disk now.

A trainer's in-memory table may differ from the table its checkpoint exports;
the framework rewrites the file and keeps the trainer's copy. After that
rewrite the exported table differs from the checkpoint on no item, while the
original difference stays recorded as evidence.
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.research.stage2 import export_check
from tools.issue252.archive_audit import _check_export


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ExportAgreementTests(unittest.TestCase):
    def test_agreement_counts_the_current_table_against_the_checkpoint(self):
        import numpy as np
        tokens = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        expected = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 9]])
        same_l1, same_l3, differing = export_check.agreement(tokens, expected)
        self.assertEqual(differing, 1)
        self.assertAlmostEqual(same_l3, 2/3)
        self.assertAlmostEqual(same_l1, 1.0)
        # The rewritten table IS the checkpoint's export: it differs on no item.
        same_l1, same_l3, differing = export_check.agreement(tokens, tokens[:, :3])
        self.assertEqual(differing, 0)
        self.assertEqual(same_l3, 1.0)


class BindingAuditTests(unittest.TestCase):
    def _workspace(self, root):
        tree = root/'tree'
        tree.mkdir()
        checkpoint = root/'rqvae_best.pth'
        checkpoint.write_bytes(b'checkpoint-bytes')
        sid = root/'item_sids.json'
        sid.write_text('{}', encoding='utf-8')
        manifest = {'schema': 'genrec.stage2.export_manifest/1',
                    'tree': str(tree.relative_to(root)),
                    'checkpoint': str(checkpoint.relative_to(root)),
                    'checkpoint_sha256': _sha256(checkpoint),
                    'sid': str(sid.relative_to(root)),
                    'sid_sha256': _sha256(sid),
                    'tokens_match': True, 'same_l3': 1.0, 'differing_items': 0,
                    'original_differing_items': 303, 'rewritten_from_checkpoint': True}
        return tree, checkpoint, sid, manifest

    def test_rewritten_manifest_passes_and_a_mismatch_still_fails(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            tree, checkpoint, sid, manifest = self._workspace(root)
            # A table rewritten from the checkpoint differs on no item: accepted.
            _check_export(manifest, tree, checkpoint, sid, root)
            manifest['differing_items'] = 1
            with self.assertRaises(ValueError):
                _check_export(manifest, tree, checkpoint, sid, root)
            manifest['differing_items'] = 0
            manifest['tokens_match'] = False
            with self.assertRaises(ValueError):
                _check_export(manifest, tree, checkpoint, sid, root)

    def test_manifest_keeps_the_original_difference_as_evidence(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            tree, checkpoint, sid, manifest = self._workspace(root)
            text = json.dumps(manifest)
            self.assertIn('"original_differing_items": 303', text)


if __name__ == '__main__':
    unittest.main()
