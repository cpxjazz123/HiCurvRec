"""Diagnostic: which measurable SID property separates the measured Stage3 Recall.

This is an exploratory correlation over already-read final test values (the same
fixed test has been read by every earlier round); it selects nothing, trains
nothing and re-runs no model. It is reported with that bias disclosure.

Usage: python3.9 sid_quality_vs_recall.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/fs04/ar57/wenyu/GeneRec')
sys.path.insert(0, str(ROOT))

import numpy as np
import pyarrow.parquet as pq

from tools.issue252.layer_metrics import catalogue_metrics, validation_metrics

STAGE3 = ROOT / 'results/stage3_T5Train'
#: Arms whose historical Stage3 record predates the sid_path field: the SID file is
#: recovered from the arm's own Stage2 result directory and the pairing is disclosed.
SID_FALLBACK = {arm: str(ROOT / 'results/stage2_RQ-VAE' / arm / 'item_sids.json')
                for arm in ('TIGER_mixed_rq_F0', 'TIGER_mixed_rq_E', 'TIGER_mixed_rq_G',
                            'TIGER_mixed_rq_L', 'TIGER_mixed_rq_A', 'TIGER_role_geom_rq',
                            'TIGER_role_geom_global_rq', 'TIGER_role_geom_shuffled_rq',
                            'TIGER_graphlet_role_E', 'TIGER_graphlet_role_Q',
                            'TIGER_graphlet_role_QR', 'TIGER_graphlet_role_R',
                            'TIGER_graphlet_role_T', 'TIGER_struct_mixed_rq',
                            'TIGER_af3_gate_mixed_rq', 'TIGER_RQ-VAE', 'TIGER_RQ-VAE__canonical')}
ARMS = [
    'TIGER_struct_mixed_rq', 'TIGER_mixed_rq_F0', 'TIGER_mixed_rq_L', 'TIGER_mixed_rq_A',
    'TIGER_mixed_rq_G', 'TIGER_mixed_rq_E', 'TIGER_role_geom_rq', 'TIGER_role_geom_global_rq',
    'TIGER_role_geom_shuffled_rq', 'TIGER_graphlet_role_E', 'TIGER_graphlet_role_QR',
    'TIGER_graphlet_role_R', 'TIGER_graphlet_role_Q', 'TIGER_graphlet_role_T',
    'TIGER_module_budget_prefix_rq', 'TIGER_periodic_budget_prefix_rq',
    'TIGER_RQ-VAE', 'TIGER_RQ-VAE__canonical',
]


def finals(arm):
    return sorted((STAGE3 / arm).glob('logs/*/*/test_final.json'))


def load_sid(path):
    payload = json.loads(Path(path).read_text())
    rows = np.array([list(value[:3]) for value in payload.values()], dtype=np.int64)
    return rows


def main():
    examples = pq.read_table(ROOT / 'results/stage0_build_parquet/valid.parquet',
                             columns=['history', 'seen_history', 'target']).to_pylist()
    records = []
    skipped = []
    for arm in ARMS:
        for final_path in finals(arm):
            final = json.loads(final_path.read_text())
            sid_path = Path(final.get('sid_path') or SID_FALLBACK.get(arm, ''))
            if not str(sid_path) or not sid_path.is_file():
                skipped.append({'arm': arm, 'stage3_run': final_path.parent.name,
                                'reason': 'no recoverable SID file for this arm',
                                'test_recall@10': final.get('test_recall@10')})
                continue
            rows = load_sid(sid_path)
            catalogue = catalogue_metrics(rows)
            validation = validation_metrics(rows, examples)
            records.append({
                'arm': arm,
                'stage3_run': final_path.parent.name,
                'sid_path': str(sid_path),
                'sid_sha256': final.get('sid_sha256') or hashlib.sha256(sid_path.read_bytes()).hexdigest(),
                'test_recall@10': final['test_recall@10'],
                'test_recall@5': final['test_recall@5'],
                'test_ndcg@10': final['test_ndcg@10'],
                'l1_positive_hit_rate': validation['l1_positive_hit_rate'],
                'l1_negative_hit_probability': validation['l1_negative_hit_probability'],
                'l1_lift': (validation['l1_positive_hit_rate'] /
                            validation['l1_negative_hit_probability']),
                'l1_hard_er': validation['l1_hard_er'],
                'l2_positive_hit_rate': validation['positive_hit_rates_by_level'][1],
                'l3_positive_hit_rate': validation['positive_hit_rates_by_level'][2],
                'distinct_l2_prefixes': catalogue['distinct_l2_prefixes'],
                'l3_collision': catalogue['l3_collision'],
                'l3_refinement_gain_bits': catalogue['l3_refinement_gain_bits'],
                'prefix_entropy_bits': catalogue['prefix_entropy_bits'],
            })
    records.sort(key=lambda row: -row['test_recall@10'])
    out = ROOT / 'results/research/issue256/rounds/r02/analysis/sid_quality_vs_recall.json'
    out.write_text(json.dumps({'schema': 'genrec.issue256.sid_quality_vs_recall/1',
                               'bias_disclosure': 'Exploratory correlation over already-read final test values; no selection, no training, no model replay.',
                               'records': records, 'skipped': skipped}, indent=2) + '\n')
    header = ['arm', 'r@10', 'l1_pos', 'l1_neg', 'lift', 'hard_er', 'l2_pos', 'l3_pos',
              'distinct_l2', 'l3_coll', 'refine_bits']
    print('\t'.join(header))
    for row in records:
        print('\t'.join([
            row['arm'], '%.6f' % row['test_recall@10'], '%.5f' % row['l1_positive_hit_rate'],
            '%.6f' % row['l1_negative_hit_probability'], '%.2f' % row['l1_lift'],
            '%.2f' % row['l1_hard_er'], '%.5f' % row['l2_positive_hit_rate'],
            '%.6f' % row['l3_positive_hit_rate'], str(row['distinct_l2_prefixes']),
            '%.5f' % row['l3_collision'], '%.3f' % row['l3_refinement_gain_bits']]))
    if len(records) > 2:
        y = np.array([row['test_recall@10'] for row in records])
        for key in ('l1_positive_hit_rate', 'l1_lift', 'l1_hard_er', 'l2_positive_hit_rate',
                    'l3_positive_hit_rate', 'distinct_l2_prefixes', 'l3_collision',
                    'l3_refinement_gain_bits'):
            x = np.array([row[key] for row in records], dtype=float)
            if x.std() == 0:
                continue
            print('pearson %-28s %+.3f' % (key, float(np.corrcoef(x, y)[0, 1])))
    print('written', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
