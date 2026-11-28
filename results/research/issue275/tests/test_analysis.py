"""Minimal checks for the #275 analysis: metric arithmetic, label aggregation, expression identity."""
import json
import sys
from pathlib import Path
import numpy as np

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P))
from run_analysis import metrics  # noqa: E402


def test_metrics_perfect():
    m = metrics([0.05, 0.06, 0.07], [0.05, 0.06, 0.07])
    assert abs(m['mae']) < 1e-12 and abs(m['rmse']) < 1e-12
    assert abs(m['spearman'] - 1.0) < 1e-12 and m['reversals'] == 0.0


def test_metrics_reversal_counted():
    m = metrics([0.05, 0.06], [0.06, 0.05])
    assert m['reversals'] == 1.0


def test_label_aggregation_matches_inventory():
    inv = json.load(open(P / 'RUN_INVENTORY.json'))
    joint = {r['sid_sha256']: r for r in json.load(open(P / 'JOINT_TABLE.json'))['rows']}
    checked = 0
    for u in inv['units']:
        sha = u.get('sid_sha256')
        if not sha or sha not in joint:
            continue
        recs = u['records']
        ws = [r['world_size'] for r in recs if r['world_size'] is not None]
        major = max(set(ws), key=ws.count) if ws else None
        used = [r['recall@10'] for r in recs
                if (major is None or r['world_size'] == major) and r['recall@10'] is not None]
        if not used:
            continue
        assert abs(joint[sha]['label_recall@10'] - float(np.mean(used))) < 1e-12
        checked += 1
    assert checked > 40


def test_expressions_recompute_exactly():
    d = json.load(open(P / 'ANALYSIS_RESULTS.json'))
    for q in ('Q_SID', 'Q_VEC'):
        for fam, e in d.get(q, {}).get('folds', {}).items():
            assert e['expr_max_abs_diff'] is not None and e['expr_max_abs_diff'] < 1e-9, (q, fam)


if __name__ == '__main__':
    test_metrics_perfect()
    test_metrics_reversal_counted()
    test_label_aggregation_matches_inventory()
    test_expressions_recompute_exactly()
    print('all checks passed')
