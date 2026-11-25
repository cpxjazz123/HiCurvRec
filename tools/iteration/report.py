"""One place that writes the standard artefact set of a round.

RUN_MANIFEST, TIMELINE, VALIDATION_PLAN, RECEIPTS, REPORT, VERDICT and
ROUND_RESULT are generated from the identity, the plan, the receipts and the
analysis; nothing is retyped by hand, so a later reader can rebuild the report
from the same inputs.
"""
from __future__ import annotations

import json
from pathlib import Path


def _write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return path


def write_all(out_dir, *, identity, plan, receipts, analysis, timeline=None,
              verdict=None, round_result=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    written['RUN_MANIFEST'] = _write(out / 'RUN_MANIFEST.json',
                                    {'schema': 'genrec.pipeline.run_manifest/1',
                                     'arm': identity.get('arm'),
                                     'stage2_run_id': identity.get('stage2_run_id'),
                                     'stage3_run_id': identity.get('stage3_run_id'),
                                     'artefacts': {key: identity.get(key)
                                                   for key in ('sid', 'stage2_checkpoint',
                                                               'stage2_export_manifest',
                                                               'archive_binding', 'test_final')}})
    written['VALIDATION_PLAN'] = _write(out / 'VALIDATION_PLAN.json', plan)
    written['RECEIPTS'] = _write(out / 'RECEIPTS.json',
                                 {'schema': 'genrec.pipeline.receipts/1', 'receipts': receipts})
    if timeline is not None:
        written['TIMELINE'] = _write(out / 'TIMELINE.json', timeline)
    written['ANALYSIS'] = _write(out / 'ANALYSIS.json', analysis)
    if round_result is not None:
        written['ROUND_RESULT'] = _write(out / 'ROUND_RESULT.json', round_result)
    written['REPORT'] = _report(out / 'REPORT.md', identity=identity, analysis=analysis,
                                plan=plan, timeline=timeline, verdict=verdict)
    if verdict is not None:
        written['VERDICT'] = _write(out / 'VERDICT.json', verdict)
        written['VERDICT_MD'] = _verdict_md(out / 'VERDICT.md', identity=identity, verdict=verdict,
                                            analysis=analysis)
    return {key: str(path) for key, path in written.items()}


def _report(path, *, identity, analysis, plan, timeline, verdict):
    chain = analysis.get('sid_chain') or {}
    role = analysis.get('role_conditioning') or {}
    stage3 = identity.get('metrics') or {}
    lines = ['# Round report: ' + str(identity.get('arm')), '',
             'Run: `' + str(identity.get('stage2_run_id')) + '`',
             'Stage3: `' + str(identity.get('stage3_run_id')) + '`', '',
             '## SID structure', '',
             '| quantity | value |', '|---|---|']
    for key in ('items', 'H_L1', 'H_L2_given_L1', 'H_L3_given_L1L2', 'distinct_L1',
                'distinct_L2', 'distinct_triples', 'items_per_L2', 'items_per_triple', 'collision'):
        lines.append('| %s | %s |' % (key, chain.get(key)))
    if role.get('available'):
        lines += ['', '## Role conditioning', '', '| quantity | mean | median | p90 |',
                  '|---|---|---|---|']
        for key in ('roles_per_prefix', 'items_per_prefix', 'items_per_cell', 'distinct_L3_per_cell'):
            row = role.get(key) or {}
            lines.append('| %s | %s | %s | %s |' % (key, row.get('mean'), row.get('median'), row.get('p90')))
    if stage3:
        lines += ['', '## Stage3 (complete population)', '',
                  '| metric | value |', '|---|---|']
        for key in ('n_eval', 'test_recall@5', 'test_recall@10', 'test_ndcg@5', 'test_ndcg@10'):
            lines.append('| %s | %s |' % (key, stage3.get(key)))
    if plan:
        lines += ['', '## Validation', '',
                  'reused: %d, rerun: %d' % (len(plan.get('reuse', [])), len(plan.get('rerun', [])))]
        for row in plan.get('rerun', []):
            lines.append('- rerun `%s`: %s' % (row.get('test_id'), row.get('reason')))
    if timeline:
        lines += ['', '## Timing', '',
                  'merged wall: %ss, GPU wall: %ss' % (timeline.get('merged_wall_s'), timeline.get('gpu_wall_s'))]
    if verdict:
        lines += ['', '## Verdict', '', 'status: **' + str(verdict.get('status')) + '**']
        for row in verdict.get('checks', []):
            lines.append('- %s: %s' % (row.get('check'), row.get('status')))
    Path(path).write_text('\n'.join(lines) + '\n')
    return path


def _verdict_md(path, *, identity, verdict, analysis):
    lines = ['# Verdict: ' + str(identity.get('arm')), '',
             'Status: **' + str(verdict.get('status')) + '**', '',
             '## Checks', '', '| check | expected | actual | status |', '|---|---|---|---|']
    for row in verdict.get('checks', []):
        lines.append('| %s | %s | %s | %s |' % (row.get('check'), row.get('expected', row.get('recorded', '')),
                                                row.get('actual', row.get('current', '')), row.get('status')))
    if verdict.get('limits'):
        lines += ['', '## Limits', '', verdict['limits']]
    Path(path).write_text('\n'.join(lines) + '\n')
    return path
