"""Shared helpers for the iteration pipelines tests (not collected)."""
from __future__ import annotations

import json
from pathlib import Path

from tools.iteration import depgraph


def evidence(root, unit='unit', *, log_text='ok\n', output_text='{"ok": true}\n'):
    root = Path(root)
    log = root / 'logs' / (unit + '.log')
    out = root / 'outputs' / (unit + '.json')
    log.parent.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(log_text)
    out.write_text(output_text)
    return {'command': ['python', '-m', 'pytest', 'tests'], 'exit_code': 0,
            'log': str(log), 'output': str(out), 'output_sha256': depgraph.file_sha(out)}


def record_valid(cache, spec, root, *, unit=None, repo=None, status='PASS', ev=None):
    repo = Path(repo or root)
    unit = unit or spec.test_id
    fp = spec.fingerprint(repo=repo)
    cache.record(unit, fp['digest'], status, evidence=ev or evidence(root, unit),
                 declared=spec.declared(), fingerprint=fp)
    return fp


def write_tree(root, files):
    root = Path(root)
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root
