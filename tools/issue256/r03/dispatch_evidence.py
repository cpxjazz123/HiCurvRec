"""Refuse formal dispatch until current-source tests and real credentials pass."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path, label):
    if not Path(path).is_file():
        raise ValueError('missing ' + label + ': ' + str(path))
    return json.loads(Path(path).read_text())


def verify(repo, source_files):
    repo = Path(repo).resolve()
    evidence = repo / 'results/research/issue256/rounds/r03'
    if not source_files:
        raise ValueError('empty source identity')
    for relative, digest in source_files.items():
        path = repo / relative
        if not path.is_file() or sha256(path) != digest:
            raise ValueError('source changed: ' + relative)
    tests = _read(evidence / 'TEST_RESULTS.json', 'test receipt')
    if not tests.get('passed') or tests.get('failed_suites'):
        raise ValueError('test suites did not pass')
    for name in ('candidate', 'assets', 'tools'):
        suite = tests.get('suites', {}).get(name, {})
        summary = '\n'.join(suite.get('summary', []))
        if (suite.get('exit_code') != 0 or not re.search(r'[1-9][0-9]* passed', summary)
                or re.search(r'\b[1-9][0-9]* skipped\b', summary)):
            raise ValueError('missing/zero/skipped test suite: ' + name)
    for relative, digest in tests.get('sources_sha256', {}).items():
        path = repo / relative
        if not path.is_file() or sha256(path) != digest:
            raise ValueError('tested source changed: ' + relative)
    records = {'TEST_RESULTS.json': tests}
    for name in ('ADMISSION.json', 'VALIDATION.json', 'PERFORMANCE.json', 'PERIODIC_VALIDATION.json'):
        report = _read(evidence / name, name)
        if report.get('source_hashes', {}).get('files') != source_files:
            raise ValueError('credential source identity differs: ' + name)
        records[name] = report
    admission, validation, performance = (records[name] for name in
                                        ('ADMISSION.json', 'VALIDATION.json', 'PERFORMANCE.json'))
    if not admission.get('passed') or {c['name']: c['status'] for c in admission.get('checks', [])} != {
            'entry': 'PASS', 'behavior_graph': 'PASS'}:
        raise ValueError('machine admission did not pass')
    if not validation.get('passed'):
        raise ValueError('real validation did not pass')
    if (not performance.get('ceiling_met') or performance.get('mean_seconds', float('inf')) > .8
            or performance.get('window', {}).get('epochs', 0) < 30):
        raise ValueError('inclusive performance ceiling did not pass')
    rank_evidence = {}
    for role in ('VALIDATION.json', 'PERFORMANCE.json', 'PERIODIC_VALIDATION.json'):
        directory = Path(records[role]['source_run_directory'])
        if not directory.is_absolute():
            directory = repo / directory
        for rank in range(4):
            path = directory / ('rank_%d.json' % rank)
            report = _read(path, role + ' rank ' + str(rank))
            if (report.get('rank') != rank or str(report.get('status')).lower() != 'pass'
                    or report.get('source_hashes', {}).get('files') != source_files):
                raise ValueError('rank source/status differs: ' + str(path))
            recipe = report.get('recipe', {})
            if any(recipe.get(k) != v for k, v in
                   {'epochs': 3000, 'global_batch': 4096, 'world_size': 4, 'seed': 42}.items()):
                raise ValueError('rank recipe differs: ' + str(path))
            if role == 'PERIODIC_VALIDATION.json':
                from tools.issue256.r03.calibration_receipts import validate_diagnostic
                validate_diagnostic(report)
            rank_evidence[str(path)] = sha256(path)
    modules = _read(evidence / 'MODULE_CHECKS.json', 'real module checks')
    if (not modules.get('gradient_reached_every_module') or not modules.get('optimizer_updated_every_module')
            or Path(modules['source_run_directory']) != Path(validation['source_run_directory'])):
        raise ValueError('real module gradient/update evidence differs')
    records['MODULE_CHECKS.json'] = modules
    return {'source_files': source_files,
            'credentials': {name: sha256(evidence / name) for name in records},
            'ranks': rank_evidence}
