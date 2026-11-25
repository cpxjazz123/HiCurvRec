"""Read-only replay: a synthetic run must reproduce its own numbers, and a
corrupted number must be caught.  Expected values are literal, not recomputed.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'tools' / 'research' / 'stage2' / 'contract.py').is_file())
sys.path.insert(0, str(REPO))

from tools.issue257 import identity as ident, replay


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(tmp_path):
    root = tmp_path
    run = root / 'stage2_RQ-VAE' / '_runs' / 'run99' / 'ARM' / 'result'
    run.mkdir(parents=True)
    sid = {'0': [0, 0, 0, 0], '1': [0, 0, 1, 0], '2': [0, 1, 0, 0], '3': [1, 0, 0, 0]}
    (run / 'item_sids.json').write_text(json.dumps(sid, sort_keys=True))
    (run / 'rqvae_best.pth').write_bytes(b'weights')
    (run / 'export_manifest.json').write_text('{}')
    source = root / 'source'
    source.mkdir()
    (source / 'role_table.json').write_text(json.dumps({'table': [5, 5, 6, 7]}))
    (run / 'archive_binding.json').write_text(json.dumps({'source_tree': str(source)}))

    metrics_dir = root / 'results' / 'stage2_RQ-VAE' / 'ARM' / 'stage2_metrics'
    metrics_dir.mkdir(parents=True)
    digest = sha(run / 'item_sids.json')
    (metrics_dir / (digest + '.json')).write_text(json.dumps({
        'catalogue': {'distinct_l2_prefixes': 3, 'distinct_l3_triples': 4,
                      'l3_collision': 0.0, 'l3_refinement_gain_bits': 0.5}}))

    stage3 = root / 'stage3' / 'logs' / 'ds' / 'ts'
    (stage3 / 'predictions').mkdir(parents=True)
    rows = [
        {'target_item': 10, 'target_rank': 2, 'topk': [{'item': 1}, {'item': 2}, {'item': 10}]},
        {'target_item': 11, 'target_rank': None, 'topk': [{'item': 1}, {'item': 2}]},
    ]
    (stage3 / 'predictions' / 'rank0.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    final = {'sid_sha256': sha(run / 'item_sids.json'),
             'stage2_checkpoint_sha256': sha(run / 'rqvae_best.pth'),
             'stage2_export_manifest_sha256': sha(run / 'export_manifest.json'),
             'archive_binding_sha256': sha(run / 'archive_binding.json'),
             'sid_path': str(run / 'item_sids.json'),
             'stage2_checkpoint': str(run / 'rqvae_best.pth'),
             'stage2_export_manifest': str(run / 'export_manifest.json'),
             'archive_binding': str(run / 'archive_binding.json'),
             'stage2_run_id': 'run99', 'stage3_run_id': str(stage3),
             'predictions': str(stage3 / 'predictions' / 'rank0.jsonl'),
             'n_eval': 2,
             'test_recall@5': 0.5, 'test_recall@10': 0.5,
             'test_ndcg@5': 0.25, 'test_ndcg@10': 0.25}   # 1/log2(2+2) on one of two rows
    (stage3 / 'test_final.json').write_text(json.dumps(final))
    return root, stage3 / 'test_final.json', run


def test_replay_reproduces_every_number(tmp_path):
    root, final, run = build(tmp_path)
    result = replay.replay(final, repo=root)
    assert result['status'] == 'PASS', result['failed']
    checks = {row['check']: row for row in result['checks']}
    assert checks['sid_chain:distinct_l3_triples']['actual'] == 4
    assert checks['sid_chain:l3_refinement_gain_bits']['actual'] == pytest.approx(0.5)
    assert checks['independent_reduction:test_recall@10']['actual'] == pytest.approx(0.5)
    assert checks['artefact:sid']['status'] == 'PASS'


def test_replay_catches_a_corrupted_reported_number(tmp_path):
    root, final, run = build(tmp_path)
    payload = json.loads(final.read_text())
    payload['test_recall@10'] = 0.9
    final.write_text(json.dumps(payload))
    result = replay.replay(final, repo=root)
    assert result['status'] == 'FAIL'
    assert any(row['check'] == 'independent_reduction:test_recall@10' for row in result['failed'])


def test_replay_catches_a_changed_artefact(tmp_path):
    root, final, run = build(tmp_path)
    (run / 'item_sids.json').write_text(json.dumps({'0': [9, 9, 9, 0]}))
    result = replay.replay(final, repo=root)
    assert result['status'] == 'FAIL'
    assert any(row['check'] == 'artefact:sid' for row in result['failed'])
