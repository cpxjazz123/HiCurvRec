"""A sanctioned launcher changes cwd; the diagnostic must resolve its own repo."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'tools/research/stage2/contract.py').is_file())
ENTRY = ROOT / 'stage2_RQ-VAE/TIGER_gnn_attention_sectional_rq/steady_profile_entry.py'


class DriverRootTests(unittest.TestCase):
    def test_actual_standalone_driver_tracks_epochs_from_result_cwd(self):
        code = '''
import json, runpy, sys
from types import SimpleNamespace
module = runpy.run_path(sys.argv[1], run_name='import_probe')
tracker = object.__new__(module['SteadyTracker'])
tracker.training = SimpleNamespace(_record=lambda *args, **kwargs: None)
tracker.boundaries = []
tracker.completed_epochs = 0
tracker.diagnostic_end = None
calls = []
tracker.start_window = lambda: calls.append('start')
tracker.stop_window = lambda: calls.append('stop')
tracker.install_record_hook()
for epoch in (1, 2, 3):
    tracker.training._record(3, 'epoch_boundary', epoch=epoch, start=epoch, end=epoch+.5)
print(json.dumps({'events': calls, 'epochs': tracker.completed_epochs}))
'''
        env = dict(os.environ)
        env.pop('PYTHONPATH', None)
        with tempfile.TemporaryDirectory(dir=ROOT / 'results/research/issue252') as cwd:
            result = subprocess.run([sys.executable, '-c', code, str(ENTRY)], cwd=cwd,
                                    env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {'events': ['start', 'stop'], 'epochs': 3})


if __name__ == '__main__':
    unittest.main()
