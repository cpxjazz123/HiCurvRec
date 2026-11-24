"""Stdlib-only Issue252 formal admission credentials.

create_receipt(repo, tree, *, validation_reports, benchmark_reports, test_reports,
               output=None) validates and writes TRAINING_ADMISSION.json.
The two report arguments are explicit four-path lists (one actual rank each).
The test_reports mapping has deterministic/reference/mutation paths, each a
source-bound genrec.issue252.tests/1 report with kind, status='pass', tests_run,
failures/errors/skipped, and source_hashes equal to source_binding(...,
'validation'). Mutation reports also carry caught numeric/structural/routing
mutations. check_receipt(repo, tree) rechecks every byte and returns {path,sha256}.
Test reports also include command (string/list), exit_code=0 and an output
descriptor {path,sha256} for the actual captured command output.
Neither API executes tests, imports torch, guesses latest reports or grants a
pass based on a summary flag. Call creation only after the actual proofs finish.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from tools.issue252.archive_audit import imported_repo_sources, sha256, source_hashes
from tools.issue252.performance import summarize_window

ARM = 'TIGER_gnn_attention_sectional_rq'
SCHEMA = 'genrec.issue252.admission/1'
RECEIPT = 'results/research/issue252/TRAINING_ADMISSION.json'
RECIPE = {'epochs': 3000, 'per_rank_batch': 1024, 'global_batch': 4096,
          'world_size': 4, 'seed': 42}


def required(tree):
    return Path(tree).name == ARM


def _require(condition, message):
    if not condition:
        raise ValueError('Issue252 training admission: ' + message)


def _file(repo, value):
    path = Path(value)
    path = path if path.is_absolute() else Path(repo) / path
    _require(path.resolve().is_relative_to(Path(repo).resolve()), 'evidence escapes repository')
    _require(not path.is_symlink() and path.is_file(), 'missing/redirected file: ' + str(path))
    return path.resolve()


def _record(repo, value):
    path = _file(repo, value)
    return {'path': str(path), 'sha256': sha256(path)}


def _read_record(repo, record):
    _require(isinstance(record, dict), 'missing evidence descriptor')
    path = _file(repo, record.get('path', ''))
    _require(record.get('sha256') == sha256(path), 'evidence hash changed: ' + str(path))
    payload = json.loads(path.read_text(encoding='utf-8'))
    _require(isinstance(payload, dict), 'report must be an object')
    return path, payload


def source_binding(repo, tree, kind):
    """Canonical stdlib source/test inventory for diagnostics and admission."""
    repo, tree = Path(repo).resolve(), Path(tree).resolve()
    _require(kind in ('validation', 'benchmark'), 'unknown diagnostic kind')
    paths = list(tree.rglob('*.py')) + list((tree / 'config').glob('*.json'))
    paths += [repo / 'assets/manifest.json', repo / 'results/research/issue252/CONTRACT.md',
              repo / 'results/research/issue252/reference.py',
              repo / 'results/research/issue252/EXECUTION_CONTRACT.md',
              repo / 'tools/issue252/performance.py', repo / 'tools/issue252/run_contract_tests.py',
              repo / ('tools/issue252/launch_' + kind + '.py')]
    paths += list((repo / 'tools/issue252').glob('test_*.py'))
    paths += [p for p in (repo / 'tools/research/stage2').rglob('*.py')
              if not {'tests', '__pycache__'} & set(p.parts)]
    result = {str(_file(repo, p).relative_to(repo)): sha256(p) for p in paths}
    result.update(imported_repo_sources(tree, repo))
    return dict(sorted(result.items()))


def _production_binding(repo, tree):
    result = dict(source_binding(repo, tree, 'validation'))
    result.update(source_binding(repo, tree, 'benchmark'))
    # Full directory inventories prevent adding an unbound production module,
    # even when the new file is not yet a statically visible import.
    for directory in ('tools/issue252', 'tools/graph_infra', 'tools/research/stage2',
                      'tools/research/verification', 'tools/genrec-mcp', 'tools/stage2', 'tools/stage3'):
        for path in (repo / directory).rglob('*.py'):
            if {'tests', '__pycache__'} & set(path.parts) or path.name.startswith('test_'):
                continue
            result[str(path.relative_to(repo))] = sha256(_file(repo, path))
    result.update({str((tree / name).relative_to(repo)): digest
                   for name, digest in source_hashes(tree).items()})
    return dict(sorted(result.items()))


def _asset_binding(repo):
    manifest = json.loads((repo / 'assets/manifest.json').read_text(encoding='utf-8'))
    _require(isinstance(manifest.get('assets'), list), 'missing registered asset inventory')
    result = {}
    for entry in manifest['assets']:
        path = _file(repo, entry['path'])
        digest = sha256(path)
        _require(digest == entry.get('sha256') and path.stat().st_size == entry.get('bytes'),
                 'registered asset changed: ' + str(path))
        name = str(path.relative_to(repo))
        _require(name not in result or result[name] == digest, 'conflicting registered asset hashes')
        result[name] = digest
    return dict(sorted(result.items()))


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _positive(value):
    return _finite(value) and value > 0


def _stats(values, *, positive=False):
    _require(isinstance(values, dict) and bool(values), 'missing actual module gradient/update/VJP evidence')
    expected = {'encoder', 'decoder', 'GNN.input', 'sharedKey', 'sharedValue', 'sharedScore'}
    expected.update('Query%d' % i for i in range(5))
    expected.update('codebook%d' % i for i in range(3))
    _require(expected <= values.keys() and any(k.startswith('GNN.') and k != 'GNN.input' for k in values),
             'incomplete actual module coverage')
    for name, row in values.items():
        _require(isinstance(row, dict) and row.get('finite') is True and _finite(row.get('norm')),
                 'missing/nonfinite module: ' + name)
        if positive:
            _require(_positive(row['norm']) and row.get('absent') is False
                     and row.get('tensors', 0) > 0, 'missing gradient/update: ' + name)


def _validation(repo, paths, rows):
    identities = []
    for row in rows:
        _require(row.get('actual_batch_items') == 1024 and row.get('completed_epoch') == 1,
                 'validation did not complete an actual full-batch epoch')
        losses = row.get('first_step_losses', {})
        _require(set(losses) == {'original_loss', 'structural_loss', 'combined_loss'}
                 and all(_finite(v) for v in losses.values()), 'missing/nonfinite actual losses')
        _require(row.get('completed_local_losses') and all(_finite(v) for v in row['completed_local_losses']),
                 'missing/nonfinite completed losses')
        _require(row.get('actual_structural_hard_q_decode_exact') is True
                 and row.get('first_structural_sample'), 'missing actual structural sample/decode')
        for key in ('normal_backward', 'actual_optimizer_update'):
            _stats(row.get(key), positive=True)
        forward = row.get('normal_forward')
        _require(isinstance(forward, dict) and forward and all(
            value.get('finite') is True and value.get('calls', 0) > 0 for value in forward.values()),
            'missing/nonfinite actual forward evidence')
        vjp = row.get('raw_real_batch_VJP', {})
        _stats(vjp.get('original')); _stats(vjp.get('structural'))
        _require(vjp['structural']['decoder']['norm'] == 0, 'structural VJP reaches decoder')
        _require(_positive(vjp['structural']['encoder']['norm']), 'structural VJP misses encoder')
        hashes = row.get('completed_parameter_hashes')
        _require(isinstance(hashes, list) and len(hashes) == 4 and len(set(hashes)) == 1
                 and isinstance(hashes[0], str) and len(hashes[0]) == 64
                 and row.get('completed_state_finite') is True, 'missing/divergent four-rank DDP identity')
        identities.append(hashes[0])
        switches = row.get('switches', {})
        for label in ('both_on', 'behavior_off', 'curvature_off', 'both_off'):
            switch = switches.get(label, {})
            _require(switch.get('hard_q_decode_exact') is True and all(
                isinstance(switch.get(key), str) and len(switch[key]) == 64
                for key in ('weights_sha256', 'cost_sha256', 'tokens_sha256')),
                'missing actual switch decode: ' + label)
            if label != 'both_on':
                _require(_positive(switch.get('cost_delta_norm')), 'switch did not affect actual cost')
        intervention = switches.get('query_intervention', {})
        _require(all(_positive(intervention.get(k)) for k in ('weight_delta_norm', 'cost_delta_norm')),
                 'missing actual Query intervention')
    _require(len(set(identities)) == 1, 'rank reports disagree on completed DDP parameters')
    zero = next(row for row in rows if row['rank'] == 0)
    directory = paths[rows.index(zero)].parent
    export = zero.get('export', {})
    _require(export.get('same_checkpoint_exact') is True and export.get('formal_training') is False
             and export.get('items', 0) > 0, 'missing exact diagnostic checkpoint export')
    for name, field in (('diagnostic_epoch1.pth', 'checkpoint_sha256'),
                        ('diagnostic_item_sids.json', 'sid_sha256')):
        _require(export.get(field) == sha256(_file(repo, directory / name)), 'diagnostic export hash changed')


def _benchmark(rows):
    zero = next(row for row in rows if row['rank'] == 0)
    boundaries = zero.get('epoch_boundaries', [])
    _require([row.get('epoch') for row in boundaries] == list(range(1, 61)),
             'benchmark must contain exactly 60 complete inclusive epochs')
    # Validate all raw boundaries, not only the accepted tail or its passed flag.
    summarize_window(boundaries)
    for boundary in boundaries:
        coverage = boundary.get('structural_coverage', [])
        _require(len(coverage) == 4 and all(isinstance(row, dict) and all(
            _positive(row.get(key)) for key in ('epoch_updates', 'epoch_target_occurrences',
                                               'epoch_reference_draws')) for row in coverage),
            'missing actual all-four-rank structural coverage')
    _require(50 in zero.get('periodic_evaluation_epochs', []), 'missing actual epoch50 evaluation')
    result = summarize_window([row for row in boundaries if 31 <= row['epoch'] <= 60])
    _require(result['passed'] and result['epochs'] >= 30 and result['first_epoch'] == 31
             and result['last_epoch'] == 60, 'inclusive epochs31..60 performance gate failed')
    return result


def _reports(repo, tree, descriptors, kind):
    _require(isinstance(descriptors, list) and len(descriptors) == 4, 'missing one or more ' + kind + ' ranks')
    paths, rows = [], []
    binding = source_binding(repo, tree, kind)
    for descriptor in descriptors:
        path, row = _read_record(repo, descriptor)
        _require(row.get('status') == 'pass' and row.get('formal_training') is False,
                 'missing/failing actual ' + kind + ' report')
        _require(row.get('source_hashes') == binding, 'stale ' + kind + ' source binding')
        _require(row.get('diagnostic_epochs') == (1 if kind == 'validation' else 60), 'wrong diagnostic budget')
        _require(isinstance(row.get('recipe'), dict) and all(row['recipe'].get(k) == v for k, v in RECIPE.items())
                 and row['recipe'].get('config_sha256') ==
                 sha256(_file(repo, tree / 'config/train_config.json')), 'wrong/missing frozen recipe')
        _require(_positive(row.get('worker_pid')) and _positive(row.get('controller_pid')),
                 'missing actual worker/controller identity')
        paths.append(path); rows.append(row)
    _require(len(set(paths)) == 4 and sorted(row.get('rank', -1) for row in rows) == [0, 1, 2, 3],
             'missing/duplicate rank reports')
    _require(len({p.parent for p in paths}) == 1 and len({row.get('run_id') for row in rows}) == 1
             and isinstance(rows[0].get('run_id'), str) and bool(rows[0]['run_id'])
             and len({row['controller_pid'] for row in rows}) == 1
             and len({row['worker_pid'] for row in rows}) == 4
             and all(row['recipe'] == rows[0]['recipe'] for row in rows), 'cross-run/rank recipe identity mismatch')
    if kind == 'validation':
        _validation(repo, paths, rows)
        return None
    return _benchmark(rows)


def _validate(repo, tree, receipt):
    _require(receipt.get('schema') == SCHEMA and receipt.get('tree') == str(tree), 'receipt schema/tree mismatch')
    _require(receipt.get('final_sources') == _production_binding(repo, tree), 'production/source inventory changed')
    _require(receipt.get('final_assets') == _asset_binding(repo), 'registered asset inventory changed')
    _reports(repo, tree, receipt.get('validation_reports'), 'validation')
    performance = _reports(repo, tree, receipt.get('benchmark_reports'), 'benchmark')
    tests = receipt.get('test_reports', {})
    _require(set(tests) == {'deterministic', 'reference', 'mutation'}, 'missing math/reference/mutation reports')
    binding = source_binding(repo, tree, 'validation')
    for kind, descriptor in tests.items():
        _, row = _read_record(repo, descriptor)
        _require(row.get('schema') == 'genrec.issue252.tests/1' and row.get('kind') == kind
                 and row.get('status') == 'pass' and row.get('source_hashes') == binding
                 and _positive(row.get('tests_run'))
                 and all(row.get(k) == 0 for k in ('failures', 'errors', 'skipped')),
                 'missing/failing/stale ' + kind + ' test evidence')
        _require(isinstance(row.get('command'), (str, list)) and bool(row['command'])
                 and row.get('exit_code') == 0, 'missing/failed actual test execution')
        output = row.get('output', {})
        _require(isinstance(output, dict), 'missing captured test output')
        output_path = _file(repo, output.get('path', ''))
        _require(output.get('sha256') == sha256(output_path), 'captured test output changed')
        if kind == 'mutation':
            mutations = row.get('mutations', [])
            _require(bool(mutations) and all(m.get('caught') is True for m in mutations)
                     and {'numeric', 'structural', 'routing'} <= {m.get('category') for m in mutations},
                     'missing/uncaught mutations')
    return performance


def create_receipt(repo, tree, *, validation_reports, benchmark_reports, test_reports, output=None):
    """Validate explicit raw evidence, write the sole receipt, return its descriptor."""
    repo, tree = Path(repo).resolve(), Path(tree).resolve()
    _require(tree == repo / 'stage2_RQ-VAE' / ARM, 'not the Issue252 active candidate')
    target = repo / RECEIPT
    _require(output is None or Path(output).resolve() == target, 'receipt must use its explicit policy path')
    receipt = {'schema': SCHEMA, 'tree': str(tree), 'final_sources': _production_binding(repo, tree),
               'final_assets': _asset_binding(repo),
               'validation_reports': [_record(repo, p) for p in validation_reports],
               'benchmark_reports': [_record(repo, p) for p in benchmark_reports],
               'test_reports': {kind: _record(repo, path) for kind, path in test_reports.items()}}
    receipt['performance'] = _validate(repo, tree, receipt)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.json.partial')
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(target)
    return _record(repo, target)


def check_receipt(repo, tree):
    """Recompute evidence and inventories; missing or stale evidence is never a pass."""
    repo, tree = Path(repo).resolve(), Path(tree).resolve()
    path = _file(repo, repo / RECEIPT)
    receipt = json.loads(path.read_text(encoding='utf-8'))
    _require(isinstance(receipt, dict), 'receipt must be an object')
    _validate(repo, tree, receipt)
    return _record(repo, path)
