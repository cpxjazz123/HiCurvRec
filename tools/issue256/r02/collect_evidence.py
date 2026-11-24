"""Collect one complete current-source validation/benchmark/profile generation.

Selection is by exact source identity and four actual rank reports, never mtime
or the best metric. Multiple matching runs are ambiguous and fail closed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

from tools.issue256.r02.identity import ROOT, SOURCE, EVIDENCE, RUN_ID


def record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def select(kind, files):
    matches = []
    core_prefix = str(SOURCE.relative_to(ROOT)) + '/'
    core = {name: digest for name, digest in files.items() if name.startswith(core_prefix)}
    for directory in sorted((EVIDENCE / kind).glob('*')):
        paths = [directory / ('rank_%d.json' % i) for i in range(4)]
        if not all(path.is_file() for path in paths):
            continue
        reports = [read_json(path) for path in paths]
        if kind == 'steady_profile':
            good = all(row.get('status') == 'PROFILED' and row.get('source_unchanged')
                       and row['source_hashes_before'] == row['source_hashes_after']
                       and all(row['source_hashes_before'].get(name) == digest for name, digest in core.items())
                       for row in reports)
        else:
            good = all(str(row.get('status')).lower() == 'pass'
                       and row.get('source_hashes', {}).get('files') == files for row in reports)
        if good and all(row.get('rank') == rank for rank, row in enumerate(reports)):
            matches.append((directory, paths, reports))
    if len(matches) != 1:
        raise ValueError('%s requires one complete exact-source run, found %d' % (kind, len(matches)))
    return matches[0]


def main():
    sys.path.insert(0, str(SOURCE))
    import benchmark_entry
    from tools.issue253.collect_module_checks import collect as collect_modules
    from tools.issue253.collect_performance import collect as collect_performance

    files = benchmark_entry.source_binding('collection')['files']
    validation_dir, validation_paths, validation = select('validation', files)
    benchmark_dir, benchmark_paths, benchmark = select('benchmark', files)
    profile_dir, profile_paths, profiles = select('steady_profile', files)
    from tools.issue256.r02.calibration_receipts import validate_diagnostic
    periodic_dir, periodic_paths, periodic = select('periodic_budget', files)
    for report in periodic:
        validate_diagnostic(report)
    write_json(EVIDENCE / 'PERIODIC_VALIDATION.json', {
        'schema': 'genrec.issue256.periodic_validation/1', 'passed': True,
        'approved_run_id': RUN_ID, 'source_run_directory': str(periodic_dir.resolve()),
        'source_hashes': periodic[0]['source_hashes'],
        'ranks': [record(path) for path in periodic_paths],
        'anchors': {str(i): row['anchors'] for i, row in enumerate(periodic)}})
    for rank in validation:
        if len(rank.get('individual_layer_losses', [])) != 3:
            raise ValueError('real per-layer gradient evidence is missing')
    collect_modules(validation_dir, EVIDENCE / 'MODULE_CHECKS.json')
    modules = read_json(EVIDENCE / 'MODULE_CHECKS.json')
    modules.update(schema='genrec.issue256.module_checks/1', approved_run_id=RUN_ID,
                   timing_scope='pre-dispatch preregistered-candidate diagnostic',
                   source_hashes=validation[0]['source_hashes'],
                   individual_layer_losses={str(i): row['individual_layer_losses']
                                            for i, row in enumerate(validation)})
    modules['initial_global_module_budget'] = {
        str(i): row['initial_global_module_budget'] for i, row in enumerate(validation)}
    write_json(EVIDENCE / 'MODULE_CHECKS.json', modules)
    write_json(EVIDENCE / 'VALIDATION.json', {
        'schema': 'genrec.issue256.validation/1', 'passed': True, 'approved_run_id': RUN_ID,
        'source_run_directory': str(validation_dir.resolve()),
        'source_hashes': validation[0]['source_hashes'],
        'ranks': [record(path) for path in validation_paths],
        'module_checks': record(EVIDENCE / 'MODULE_CHECKS.json'),
        'timing_scope': 'before the one preregistered formal dispatch'})
    collect_performance(benchmark_dir, EVIDENCE / 'PERFORMANCE.json')
    performance = read_json(EVIDENCE / 'PERFORMANCE.json')
    performance.update(schema='genrec.issue256.performance/1', approved_run_id=RUN_ID,
                       source_run_directory=str(benchmark_dir.resolve()),
                       source_hashes=benchmark[0]['source_hashes'],
                       ranks=[record(path) for path in benchmark_paths])
    performance['native_block_calibrations'] = benchmark[0]['actual_calibration_records']
    performance['verdict'] = ('inclusive ceiling=%s; target=%s; expected=%s' %
                              (performance['ceiling_met'], performance['target_met'], performance['expected_met']))
    write_json(EVIDENCE / 'PERFORMANCE.json', performance)
    write_json(EVIDENCE / 'PROFILE_SUMMARY.json', {
        'schema': 'genrec.issue256.profile_summary/1', 'approved_run_id': RUN_ID,
        'source_run_directory': str(profile_dir.resolve()),
        'scope': 'instrumented epoch2, not the inclusive performance acceptance window',
        'token_names': {'GNN_1hop': 'complete one-hop', 'GNN_2hop': 'complete two-hop',
                        'GNN_3hop': 'in-only induced', 'GNN_4hop': 'out-only induced'},
        'memory_scope': 'absolute allocator high-water marks, not incremental module memory',
        'ranks': [{'rank': index, 'source': record(path),
                   'window_seconds': row['profile_window_wall_seconds'],
                   'peak_allocated_bytes': row['profile_window_peak_allocated_bytes'],
                   'peak_reserved_bytes': row['profile_window_peak_reserved_bytes'],
                   'profiled_epoch': row['profiled_epoch'],
                   'optimizer_steps': row['profiled_optimizer_steps'],
                   'phases': [{key: value for key, value in phase.items() if key != 'footprint'}
                              for phase in row['phases']]}
                  for index, (path, row) in enumerate(zip(profile_paths, profiles))]})
    if files != benchmark_entry.source_binding('collection')['files']:
        raise ValueError('source changed while collecting evidence')
    print(json.dumps({'validation': str(validation_dir), 'benchmark': str(benchmark_dir),
                      'profile': str(profile_dir), 'approved_run_id': RUN_ID,
                      'ceiling_met': performance['ceiling_met']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
