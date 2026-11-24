"""Run the candidate's own test suites and the asset tests, then write the receipt.

The receipt binds every test count to the SHA-256 of the exact source files
that produced it, so a later source change invalidates it automatically.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'tools/research/stage2/contract.py').is_file()
            and (p / 'stage2_RQ-VAE').is_dir())
TREE = ROOT / 'stage2_RQ-VAE/TIGER_prefix_curvature_gate_rq'
PYTHON = '/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9'
OUTPUT = ROOT / 'results/research/issue255'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def test_source_hashes():
    sys.path.insert(0, str(ROOT))
    from tools.issue252.archive_audit import imported_repo_sources
    sources = imported_repo_sources(TREE, ROOT)
    for directory in (ROOT / 'tools/issue254', ROOT / 'tools/issue255', TREE):
        for path in sorted(directory.rglob('*')):
            if (path.is_file() and path.suffix in ('.py', '.json')
                    and '__pycache__' not in path.parts and 'logs' not in path.parts):
                sources[str(path.relative_to(ROOT))] = sha256(path)
    for path in (Path(__file__), OUTPUT / 'CONTRACT.md', OUTPUT / 'ACCEPTANCE.json', OUTPUT / 'MUTATIONS.md'):
        sources[str(path.relative_to(ROOT))] = sha256(path)
    return sources


def main() -> int:
    started = time.time()
    before = test_source_hashes()
    suites = {
        'candidate': (TREE / 'tests', TREE),
        'assets': (ROOT / 'tools/issue255/tests', ROOT),
    }
    results = {}
    failures = []
    for name, (directory, cwd) in suites.items():
        command = [PYTHON, '-c',
                   'import torch, pytest; torch.set_num_threads(1); raise SystemExit(pytest.main())',
                   str(directory), '-q', '--basetemp=' + str(OUTPUT / ('pytest_' + name))]
        completed = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True)
        tail = completed.stdout.strip().splitlines()[-12:]
        results[name] = {'command': command, 'cwd': str(cwd.relative_to(ROOT)),
                         'exit_code': completed.returncode, 'summary': tail}
        if completed.returncode != 0:
            failures.append(name)
        (OUTPUT / ('TEST_%s.log' % name.upper())).write_text(
            completed.stdout + '\n' + completed.stderr, encoding='utf-8')
    sources = test_source_hashes()
    if sources != before:
        failures.append('source_changed')
    record = {'schema': 'genrec.issue255.test_results/1', 'issue': 255,
              'wall_seconds': round(time.time() - started, 3),
              'suites': results, 'failed_suites': failures,
              'passed': not failures, 'sources_sha256': sources}
    (OUTPUT / 'TEST_RESULTS.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n',
                                               encoding='utf-8')
    print(json.dumps({'passed': record['passed'], 'failed_suites': failures,
                      'suites': {name: value['summary'][-1] for name, value in results.items()}},
                     indent=2))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
