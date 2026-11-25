"""Read-only replay of one finished run: recompute its numbers from its own bytes.

The replay never trains, never rewrites a weight and never touches the original
result directory.  It answers three questions with tolerances:

* does the SID still contain the triple structure the stored layer record reports?
* does an independent reduction of the persisted Stage3 predictions reproduce the
  reported Recall/NDCG on the complete population?
* do the artefact identities in the final record still match the files on disk?
"""
from __future__ import annotations

import json
from pathlib import Path

from tools.iteration import analysis, identity as ident


def _close(actual, expected, tolerance):
    if actual is None or expected is None:
        return False
    return abs(float(actual) - float(expected)) <= tolerance


def replay(path, *, repo=ident.REPO, tolerance=1e-10):
    block = ident.identity(path)
    out = analysis.analyse(block, repo=repo)
    checks = []

    stored = (out.get('layer_record') or {}).get('catalogue') or {}
    chain = out['sid_chain']
    pairs = [('distinct_l3_triples', 'distinct_triples', 0),
             ('distinct_l2_prefixes', 'distinct_L2', 0),
             ('l3_collision', 'collision', tolerance),
             ('l3_refinement_gain_bits', 'H_L3_given_L1L2', tolerance)]
    for stored_key, computed_key, tol in pairs:
        expected = stored.get(stored_key)
        actual = chain.get(computed_key)
        checks.append({'check': 'sid_chain:' + stored_key, 'expected': expected, 'actual': actual,
                       'tolerance': tol, 'status': 'PASS' if _close(actual, expected, tol) else 'FAIL'})

    metrics = block.get('metrics') or {}
    predictions = block.get('predictions') or []
    if predictions and metrics.get('n_eval'):
        reduced = analysis.reduce_predictions(predictions, metrics['n_eval'])
        for key in ('test_recall@5', 'test_recall@10', 'test_ndcg@5', 'test_ndcg@10'):
            checks.append({'check': 'independent_reduction:' + key,
                           'expected': metrics.get(key), 'actual': reduced.get(key),
                           'tolerance': tolerance,
                           'status': 'PASS' if _close(reduced.get(key), metrics.get(key), tolerance) else 'FAIL'})
    else:
        checks.append({'check': 'independent_reduction', 'status': 'NA',
                       'reason': 'no persisted predictions for this run'})

    artefacts = []
    for key in ('test_final', 'sid', 'stage2_checkpoint', 'stage2_export_manifest', 'archive_binding'):
        item = block.get(key)
        if not item:
            artefacts.append({'check': 'artefact:' + key, 'status': 'NA'})
            continue
        current = ident.sha256(item['path'])
        recorded = item.get('recorded_sha256', item['sha256'])
        artefacts.append({'check': 'artefact:' + key, 'path': item['path'],
                          'recorded': recorded, 'current': current,
                          'status': 'PASS' if current == recorded else 'FAIL'})
    checks += artefacts

    failed = [row for row in checks if row['status'] == 'FAIL']
    return {'schema': 'genrec.pipeline.replay/1',
            'run': {'arm': block.get('arm'), 'stage2_run_id': block.get('stage2_run_id'),
                    'stage3_run_id': block.get('stage3_run_id')},
            'checks': checks,
            'status': 'PASS' if not failed else 'FAIL',
            'failed': failed,
            'analysis': out}


def write(path, out_dir, **kwargs):
    result = replay(path, **kwargs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'REPLAY.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    lines = ['# Replay of ' + str(result['run']['arm']) + ' (' + str(result['run']['stage2_run_id']) + ')', '',
             'Status: **' + result['status'] + '**', '',
             '| check | expected | actual | tolerance | status |', '|---|---|---|---|---|']
    for row in result['checks']:
        lines.append('| %s | %s | %s | %s | %s |' % (
            row.get('check'), row.get('expected', row.get('recorded', '')), row.get('actual', row.get('current', '')),
            row.get('tolerance', ''), row.get('status')))
    (out / 'REPLAY.md').write_text('\n'.join(lines) + '\n')
    return result
