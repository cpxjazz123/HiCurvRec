"""Stage3 must find the exact SID's manual layer record after §94 removal."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'tools/research/stage2/contract.py').is_file())
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location('layer_record_server', ROOT / 'tools/genrec-mcp/server.py')
server = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = server
SPEC.loader.exec_module(server)


class LayerRecordTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / 'results/research/issue252')
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name)
        self.arm = 'MetricFixture'
        self.result = self.repo / 'results/stage2_RQ-VAE' / self.arm
        self.result.mkdir(parents=True)
        self.sid = self.result / 'item_sids.json'
        self.sid.write_text('{"0":[0,0,0,0]}')
        self.digest = hashlib.sha256(self.sid.read_bytes()).hexdigest()
        self.record = self.result / 'stage2_metrics' / (self.digest + '.json')
        self.report = {'sid': {'path': str(self.sid), 'sha256': self.digest},
                       'arm': self.arm, 'run_id': 'metric-fixture',
                       'catalogue': {'items': 1, 'l3_collision': 0.},
                       'validation': {'conditional_g3': {'status': 'MISSING'}}}
        self.addCleanup(patch.stopall)
        patch.object(server, 'REPO', self.repo).start()

    def save(self):
        self.record.parent.mkdir(parents=True, exist_ok=True)
        self.record.write_text(json.dumps(self.report))

    def test_flat_record_is_reused_without_changing_bytes(self):
        self.save()
        before = self.record.read_bytes()
        result = server.record_sid_layers(self.sid)
        self.assertTrue(result['passed'])
        self.assertEqual(Path(result['report']), self.record)
        self.assertEqual(self.record.read_bytes(), before)

    def test_immutable_run_resolves_canonical_sha_keyed_record(self):
        from tools.issue252.run_workspace import freeze_run
        tree = self.repo / 'stage2_RQ-VAE' / self.arm
        (tree / 'model').mkdir(parents=True)
        (tree / 'train_rqvae.py').write_text('from model import VALUE\n')
        (tree / 'model/__init__.py').write_text('VALUE=1\n')
        run = freeze_run(self.repo, tree, 'metric-fixture', self.arm)
        self.sid = Path(run['result']) / 'item_sids.json'
        self.sid.write_text('{"0":[0,0,0,0]}')
        self.report['sid']['path'] = str(self.sid)
        self.save()
        removed = subprocess.CompletedProcess([], 1, '', 'removed automatic measurement')
        with patch.object(server.subprocess, 'run', return_value=removed):
            result = server.record_sid_layers(self.sid)
        self.assertTrue(result['passed'], result)
        self.assertEqual(Path(result['report']), self.record)
        self.report['run_id'] = 'another-run'
        self.save()
        self.assertFalse(server.record_sid_layers(self.sid)['passed'])

    def test_another_sid_hash_or_path_cannot_supply_the_record(self):
        for field, value in (('sha256', '0' * 64), ('path', str(self.sid.with_name('other.json')))):
            with self.subTest(field=field):
                self.report['sid'] = {'path': str(self.sid), 'sha256': self.digest}
                self.report['sid'][field] = value
                self.save()
                self.assertFalse(server.record_sid_layers(self.sid)['passed'])

    def test_invalid_json_is_not_reported_as_measured(self):
        self.save()
        self.record.write_text('{')
        self.assertFalse(server.record_sid_layers(self.sid)['passed'])

    def test_missing_manual_record_reports_gap_without_removed_module(self):
        removed = subprocess.CompletedProcess([], 1, '', 'removed automatic measurement')
        with patch.object(server.subprocess, 'run', return_value=removed):
            result = server.record_sid_layers(self.sid)
        self.assertFalse(result['passed'])
        self.assertIn('manual', result['reason'].lower())
        self.assertEqual(Path(result['report']), self.record)


if __name__ == '__main__':
    unittest.main()
