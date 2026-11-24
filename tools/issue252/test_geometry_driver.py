"""CPU-only startup, report, and real-event resource-gate contracts."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'stage2_RQ-VAE').is_dir()
            and (p / 'tools/research/stage2/contract.py').is_file())
ENTRY = ROOT / 'stage2_RQ-VAE/TIGER_gnn_attention_sectional_rq/geometry_validation_entry.py'
DRIVER = ROOT / 'tools/issue252/launch_geometry_validation.py'


class GeometryDriverTests(unittest.TestCase):
    def probe(self, code, path):
        base = ROOT / 'results/research/issue252/geometry_validation'
        base.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.pop('PYTHONPATH', None)
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        with tempfile.TemporaryDirectory(dir=base) as cwd:
            result = subprocess.run([sys.executable, '-c', code, str(path)],
                                    cwd=cwd, env=env, capture_output=True,
                                    text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_launcher_resolves_repo_before_tools_import_from_result_cwd(self):
        value = self.probe('''
import contextlib, io, json, runpy, sys
m = runpy.run_path(sys.argv[1], run_name='import_probe')
calls = []
m['main'].__globals__['launch'] = lambda *a, **kw: calls.append((a, kw)) or 17
with contextlib.redirect_stdout(io.StringIO()):
    assert m['main']() == 17
args, kw = calls[0]
print(json.dumps({'root': str(m['ROOT']), 'entry': str(args[0]),
                  'output': str(args[1]), 'world_size': kw['world_size'],
                  'port': kw['port'], 'python': kw['python']}))
''', DRIVER)
        self.assertEqual(value['root'], str(ROOT))
        self.assertEqual(value['entry'], str(ENTRY))
        self.assertTrue(Path(value['output']).is_relative_to(
            ROOT / 'results/research/issue252/geometry_validation'))
        self.assertEqual((value['world_size'], value['port']), (4, 50202))
        self.assertEqual(value['python'],
                         '/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9')

    def test_entry_import_is_cpu_safe_and_report_schema_is_complete(self):
        value = self.probe('''
import json, runpy, sys
m = runpy.run_path(sys.argv[1], run_name='import_probe')
assert 'torch' not in sys.modules
r = m['new_report'](2, 'schema_probe')
print(json.dumps(r))
''', ENTRY)
        self.assertEqual(value['rank'], 2)
        self.assertEqual(value['run_id'], 'schema_probe')
        self.assertFalse(value['formal_training'])
        self.assertFalse(value['optimizer_created'])
        self.assertEqual(value['world_size'], 4)
        self.assertEqual(value['port'], 50202)
        self.assertEqual(value['profile_shape'], [1024, 256, 32])
        self.assertEqual(value['tolerances']['float64'], {'atol': 1e-9, 'rtol': 1e-8})
        self.assertEqual(value['tolerances']['float32'], {'atol': 1e-5, 'rtol': 1e-4})
        for key in ('source_before', 'source_after', 'command', 'device', 'floor_checks',
                    'worker_pid', 'numeric_checks', 'selection_checks', 'profiles', 'error'):
            self.assertIn(key, value)

    def test_kernel_gate_counts_actual_cuda_events_not_compile_flags(self):
        value = self.probe('''
import json, runpy, sys
m = runpy.run_path(sys.argv[1], run_name='import_probe')
small = {'cuda_kernel_count': 99, 'cuda_kernel_duration_us': 1.0}
m['assert_resource'](small)
failures = []
for count in (0, 100, 380):
    try:
        m['assert_resource']({'cuda_kernel_count': count, 'cuda_kernel_duration_us': 1.0})
    except AssertionError:
        failures.append(count)
assert m['is_compute_kernel']('vectorized_elementwise_kernel')
assert not m['is_compute_kernel']('Memcpy HtoD (Pageable -> Device)')
assert not m['is_compute_kernel']('[CUDA memset]')
print(json.dumps({'rejected': failures}))
''', ENTRY)
        self.assertEqual(value['rejected'], [0, 100, 380])

    def test_source_inventory_detects_imported_compiler_policy_drift(self):
        value = self.probe('''
import json, pathlib, runpy, sys, tempfile
m = runpy.run_path(sys.argv[1], run_name='import_probe')
with tempfile.TemporaryDirectory(dir=pathlib.Path.cwd()) as temporary:
    root = pathlib.Path(temporary)
    tree = root / 'stage2_RQ-VAE/test_arm'
    files = [
        'stage2_RQ-VAE/test_arm/model/geometry.py',
        'stage2_RQ-VAE/test_arm/tests/reference.py',
        'stage2_RQ-VAE/test_arm/config/train_config.json',
        'stage2_RQ-VAE/test_arm/geometry_validation_entry.py',
        'stage2_RQ-VAE/test_arm/assignment_floor.py',
        'tools/issue252/launch_geometry_validation.py',
        'tools/issue252/launch_validation.py',
        'tools/issue252/performance.py',
        'tools/research/stage2/contract.py',
        'tools/research/stage2/anticollapse/floor.py',
        'assets/manifest.json',
        'results/research/issue252/CONTRACT.md',
        'results/research/issue252/EXECUTION_CONTRACT.md',
        'results/research/issue252/profiling_baseline/source_snapshot.tar.gz',
    ]
    for name in files:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}' if path.suffix == '.json' else '')
    (tree / 'train_rqvae.py').write_text('import tools.issue252.cuda_math\\n')
    helper = root / 'tools/issue252/cuda_math.py'
    helper.write_text('VALUE = 1\\n')
    inventory = m['source_inventory']
    inventory.__globals__.update(ROOT=root, TREE=tree,
        __file__=str(tree / 'geometry_validation_entry.py'),
        FLOOR_SNAPSHOT=root / files[-1])
    before = inventory()
    helper.write_text('VALUE = 2\\n')
    after = inventory()
    print(json.dumps({'before': before, 'after': after}))
''', ENTRY)
        self.assertNotEqual(value['before'], value['after'],
                            'Imported compiler policy changes must invalidate GPU proof')
        helper = 'tools/issue252/cuda_math.py'
        self.assertIn(helper, value['before'])
        self.assertNotEqual(value['before'][helper], value['after'][helper])


if __name__ == '__main__':
    unittest.main()
