"""Run candidate and relevant inherited asset/framework tests, binding exact sources."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import time

from tools.issue256.r02.identity import ROOT, SOURCE, EVIDENCE, PYTHON
from tools.issue252.archive_audit import imported_repo_sources


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_inventory():
    files = imported_repo_sources(SOURCE, ROOT)
    for directory in (SOURCE, ROOT / 'tools/issue256/r02/r02', ROOT / 'tools/issue255/tests'):
        for path in sorted(directory.rglob('*.py')):
            if '__pycache__' not in path.parts:
                files[str(path.relative_to(ROOT))] = digest(path)
    for path in list((SOURCE / 'config').glob('*.json')) + [
            EVIDENCE / 'CONTRACT.md', EVIDENCE / 'MUTATIONS.md', EVIDENCE / 'reference.py',
            ROOT / 'results/research/issue256/ACCEPTANCE.json',
            ROOT / 'results/research/issue256/EVALUATION.md']:
        files[str(path.relative_to(ROOT))] = digest(path)
    return files


def main():
    before = source_inventory()
    started = time.monotonic()
    results, failed = {}, []
    for name, directory, cwd in (
            ('candidate', SOURCE / 'tests', SOURCE),
            ('assets', ROOT / 'tools/issue255/tests', ROOT)):
        command = [PYTHON, '-c',
                   'import torch, pytest; torch.set_num_threads(1); raise SystemExit(pytest.main())',
                   str(directory), '-q', '--basetemp=' + str(EVIDENCE / ('pytest_' + name))]
        done = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True)
        log = EVIDENCE / ('TEST_' + name.upper() + '.log')
        log.write_text(done.stdout + '\n' + done.stderr)
        results[name] = {'command': command, 'cwd': str(cwd), 'exit_code': done.returncode,
                         'summary': done.stdout.strip().splitlines()[-12:],
                         'log': {'path': str(log), 'sha256': digest(log)}}
        if done.returncode:
            failed.append(name)
        print(name, done.returncode, results[name]['summary'][-1:], flush=True)
    after = source_inventory()
    if before != after:
        failed.append('source_changed')
    report = {'schema': 'genrec.issue256.test_results/1', 'passed': not failed,
              'failed_suites': failed, 'suites': results, 'sources_sha256': after,
              'wall_seconds': time.monotonic() - started}
    (EVIDENCE / 'TEST_RESULTS.json').write_text(json.dumps(report, indent=2) + '\n')
    return bool(failed)


if __name__ == '__main__':
    raise SystemExit(main())
