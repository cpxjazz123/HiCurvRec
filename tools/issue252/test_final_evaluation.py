"""Final evaluation must use explicit global identity and a completed run."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'stage2_RQ-VAE').is_dir())


class FinalEvaluationTests(unittest.TestCase):
    def test_json_key_order_is_not_catalogue_order(self):
        from tools.issue252.final_evaluation import load_sid
        with tempfile.TemporaryDirectory(dir=ROOT/'results/research/issue252') as directory:
            path=Path(directory)/'item_sids.json'
            path.write_text(json.dumps({'1':[2,1,0,0],'0':[0,1,2,0]}))
            np.testing.assert_array_equal(load_sid(path,2,[3,3,3]),[[0,1,2,0],[2,1,0,0]])
            path.write_text(json.dumps({'0':[0,1,2,0],'2':[2,1,0,0]}))
            with self.assertRaisesRegex(ValueError,'global|catalogue|identit'):
                load_sid(path,2,[3,3,3])

    def test_float_and_out_of_range_codes_are_not_silently_cast(self):
        from tools.issue252.final_evaluation import load_sid
        with tempfile.TemporaryDirectory(dir=ROOT/'results/research/issue252') as directory:
            path=Path(directory)/'item_sids.json'
            for row in ([.9,1,2,0],[True,1,2,0],[3,1,2,0]):
                path.write_text(json.dumps({'0':row}))
                with self.assertRaises(ValueError):load_sid(path,1,[3,3,3])

    def test_microstep_is_not_formal_completion(self):
        from tools.issue252.final_evaluation import validate_final_checkpoint
        with self.assertRaisesRegex(ValueError,'3000|formal|complet'):
            validate_final_checkpoint({'epoch':1,'checkpoint_selection':'final_epoch','run_id':'micro'})


class FinalCheckpointTests(unittest.TestCase):
    def setUp(self):
        import torch
        from tools.issue252 import archive_audit
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT/'results/research/issue252')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.tree = self.root/'source'
        (self.tree/'model').mkdir(parents=True)
        (self.tree/'train_rqvae.py').write_text('from model import VALUE\n')
        (self.tree/'model/__init__.py').write_text('VALUE=1\n')
        self.sid = self.root/'item_sids.json'
        self.sid.write_text('{"0":[0,0,0,0]}')
        self.checkpoint = self.root/'rqvae_best.pth'
        self.payload = {
            'epoch': 3000, 'selection': 'final_epoch', 'run_id': 'formal-run',
            'source_tree': str(self.tree),
            'state_dict': {'value': torch.tensor([1.])},
            'provenance': {'candidate_sources': archive_audit.source_hashes(self.tree),
                           'infrastructure_sources': {}},
        }
        torch.save(self.payload, self.checkpoint)
        self.export = self.root/'export_manifest.json'
        self.export.write_text(json.dumps({
            'schema': 'genrec.stage2.export_manifest/1',
            'tree': str(self.tree), 'checkpoint': str(self.checkpoint),
            'checkpoint_sha256': archive_audit.sha256(self.checkpoint),
            'sid': str(self.sid), 'sid_sha256': archive_audit.sha256(self.sid),
            'tokens_match': True, 'same_l3': 1., 'differing_items': 0,
        }))
        self.binding = self.root/'archive_binding.json'
        archive_audit.create_binding(self.tree, self.checkpoint, self.sid,
                                     self.export, output=self.binding, repo=self.root)

    def load(self, **overrides):
        from tools.issue252.final_evaluation import load_bound_checkpoint
        arguments = dict(checkpoint=self.checkpoint, sid=self.sid, tree=self.tree,
                         binding=self.binding, export_manifest=self.export)
        arguments.update(overrides)
        return load_bound_checkpoint(**arguments)

    def test_complete_generation_loads_without_model_or_dataset(self):
        payload, receipt = self.load()
        self.assertEqual(payload['epoch'], 3000)
        self.assertEqual(payload['run_id'], receipt['run_id'])

    def test_missing_final_identity_and_selection_are_rejected(self):
        from tools.issue252.final_evaluation import validate_final_checkpoint
        for key in ('run_id', 'source_tree', 'provenance', 'selection'):
            payload = copy.deepcopy(self.payload)
            del payload[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_final_checkpoint(payload)
        for changes in ({'epoch': 3000.}, {'epoch': True}, {'epoch': 2999},
                        {'epoch': 3001}, {'selection': 'best_collision'},
                        {'checkpoint_selection': 'smoke'}, {'run_id': ' '},
                        {'source_tree': ''}, {'provenance': {'candidate_sources': {}}}):
            payload = copy.deepcopy(self.payload)
            payload.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_final_checkpoint(payload)

    def test_receipt_cannot_supply_another_run_identity(self):
        receipt = json.loads(self.binding.read_text())
        receipt['run_id'] = 'another-run'
        self.binding.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, 'run'):
            self.load()

    def test_crossed_or_missing_checkpoint_sid_source_fail_closed(self):
        other = self.root/'another.pth'
        other.write_bytes(self.checkpoint.read_bytes())
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            self.load(checkpoint=other)
        for path, message in ((self.sid, 'SID'), (self.checkpoint, 'checkpoint')):
            content = path.read_bytes()
            path.write_bytes(content+b' ')
            try:
                with self.subTest(path=path.name), self.assertRaisesRegex(ValueError, message):
                    self.load()
            finally:
                path.write_bytes(content)
        (self.tree/'model/__init__.py').write_text('VALUE=2\n')
        with self.assertRaisesRegex(ValueError, 'source'):
            self.load()

    def test_missing_artifacts_and_wrong_explicit_tree_fail_closed(self):
        for field in ('checkpoint', 'sid', 'binding', 'export_manifest', 'tree'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.load(**{field: self.root/'missing'})

    def test_exact_export_path_is_required_not_an_equivalent_copy(self):
        copy_path = self.root/'another-export.json'
        copy_path.write_bytes(self.export.read_bytes())
        with self.assertRaisesRegex(ValueError, 'export'):
            self.load(export_manifest=copy_path)


class SidInputTests(unittest.TestCase):
    def test_exact_width_counter_and_identity_contract(self):
        from tools.issue252.final_evaluation import load_sid
        with tempfile.TemporaryDirectory(dir=ROOT/'results/research/issue252') as directory:
            path = Path(directory)/'item_sids.json'
            for value in ({'0': [0, 1, 2]}, {'0': [0, 1, 2, 0, 0]},
                          {'0': [0, 1, 2, -1]}, {'0': [0, 1, 2, False]},
                          {'0': [0, 1, 2, .0]}, {'0': [-1, 1, 2, 0]},
                          {'00': [0, 1, 2, 0]}, [[0, 1, 2, 0]],
                          {'0': [0, 1, 2, 0], '1': [0, 1, 2, 0]}):
                path.write_text(json.dumps(value))
                with self.subTest(value=value), self.assertRaises(ValueError):
                    load_sid(path, 1, [3, 3, 3])
            path.write_text('{"0":[0,1,2,0],"0":[1,1,2,0]}')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                load_sid(path, 1, [3, 3, 3])


if __name__=='__main__':unittest.main()
