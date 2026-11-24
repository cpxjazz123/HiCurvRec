"""A growing JSONL tail cannot erase the immutable identity in its first event."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'tools/research/stage2/contract.py').is_file())
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location('metric_identity_server', ROOT / 'tools/genrec-mcp/server.py')
server = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = server
SPEC.loader.exec_module(server)


class Stage3MetricIdentityTests(unittest.TestCase):
    def test_identity_survives_more_than_64k_of_training_records(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'results/research/issue252') as temporary:
            root = Path(temporary)
            sid = root / 'item_sids.json'
            sid.write_text('{"0":[0,0,0,0]}')
            log = root / 'variant/run/training_metrics.jsonl'
            log.parent.mkdir(parents=True)
            first = {'event': 'train_start', 'code_path': str(sid), 'world_size': 4}
            rows = [first] + [{'event': 'train', 'epoch': i, 'padding': 'x' * 1024}
                              for i in range(1, 151)]
            log.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
            with patch.object(server, 'RESULT3_ROOT', root):
                metrics, final = server.stage3_bound_files('variant', sid, 'formal-run')
            self.assertEqual(metrics, log)
            self.assertIsNone(final)
            self.assertEqual(server._last_record(log, 'train')['epoch'], 150)

    def test_other_sid_stream_cannot_replace_the_long_matching_stream(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'results/research/issue252') as temporary:
            root = Path(temporary)
            sid = root / 'wanted.json'
            sid.write_text('{}')
            wanted = root / 'variant/old/training_metrics.jsonl'
            other = root / 'variant/new/training_metrics.jsonl'
            for path, code in ((wanted, sid), (other, root / 'different.json')):
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({'event': 'train_start', 'code_path': str(code)}) + '\n'
                                + ('{"event":"train","epoch":100,"pad":"' + 'x' * 1024 + '"}\n') * 100)
            with patch.object(server, 'RESULT3_ROOT', root):
                metrics, _ = server.stage3_bound_files('variant', sid, 'formal-run')
            self.assertEqual(metrics, wanted)


if __name__ == '__main__':
    unittest.main()
