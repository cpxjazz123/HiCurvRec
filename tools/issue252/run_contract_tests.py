"""Execute the frozen mathematical suites and persist real unittest receipts.

No training, model selection or recipe overrides. Invoke with the project Stage2
interpreter after production source freezes; any source drift invalidates all
receipts. All normal/edge/degenerate, reference and mutation tests run once.
"""
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest

from tools.issue252.training_admission import source_binding

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'tools/research/stage2/contract.py').is_file())
TREE = ROOT / 'stage2_RQ-VAE/TIGER_gnn_attention_sectional_rq'
OUTPUT = ROOT / 'results/research/issue252/contract_tests'
COMMAND = [sys.executable, '-m', 'tools.issue252.run_contract_tests']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def suite(patterns):
    result = unittest.TestSuite()
    for directory, pattern in patterns:
        result.addTests(unittest.TestLoader().discover(str(directory), pattern=pattern))
    return result


def main():
    # New reports never overwrite an earlier failed generation.
    import time
    import torch
    # Match the sanctioned workers' CPU budget; do not oversubscribe tiny tests.
    torch.set_num_threads(1)
    directory = OUTPUT / str(time.time_ns())
    directory.mkdir(parents=True)
    initial = source_binding(ROOT, TREE, 'validation')
    tests = TREE / 'tests'
    names = sorted(path.name for path in tests.glob('test_*.py')
                   if path.name not in ('test_reference.py', 'test_mutations.py'))
    framework_names = sorted(path.name for path in (ROOT / 'tools/issue252').glob('test_*.py')
                             if path.name != 'test_estimator_mutations.py')
    groups = {
        'deterministic': ([(tests, name) for name in names]
                          + [(ROOT / 'tools/issue252', name) for name in framework_names]),
        'reference': [(tests, 'test_reference.py')],
        'mutation': [(tests, 'test_mutations.py'),
                     (ROOT / 'tools/issue252', 'test_estimator_mutations.py')],
    }
    results = {}
    passed = True
    for kind, patterns in groups.items():
        buffer = io.StringIO()
        result = unittest.TextTestRunner(stream=buffer, verbosity=2).run(suite(patterns))
        text = buffer.getvalue()
        print(text, end='')
        log = directory / (kind + '.log')
        log.write_text(text)
        unchanged = initial == source_binding(ROOT, TREE, 'validation')
        okay = result.wasSuccessful() and result.testsRun > 0 and not result.skipped and unchanged
        report = {
            'schema': 'genrec.issue252.tests/1', 'kind': kind,
            'status': 'pass' if okay else 'fail', 'tests_run': result.testsRun,
            'failures': len(result.failures), 'errors': len(result.errors),
            'skipped': len(result.skipped), 'source_hashes': initial,
            'command': COMMAND, 'exit_code': 0 if okay else 1,
            'output': {'path': str(log.relative_to(ROOT)), 'sha256': sha256(log)},
            'producer': {'path': str(Path(__file__).relative_to(ROOT)), 'sha256': sha256(__file__)},
            'source_unchanged': unchanged,
        }
        if kind == 'mutation':
            mutations = []
            for name in ('MUTATIONS.json', 'ESTIMATOR_MUTATIONS.json'):
                for row in json.loads((ROOT / 'results/research/issue252' / name).read_text()):
                    mutations.append({**row, 'caught': row.get('caught', row.get('caught_by_assertion', False))})
            report['mutations'] = mutations
        path = directory / (kind + '.json')
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        results[kind] = str(path.relative_to(ROOT))
        passed &= okay
    print(json.dumps({'passed': passed, 'test_reports': results}, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
