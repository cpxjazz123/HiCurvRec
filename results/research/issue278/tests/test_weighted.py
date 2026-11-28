"""Known-answer checks for the weighted fidelity (CONTRACT.md section 2) and the frozen-evaluator
reproduction check on a real SID."""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path('results/research/issue278').resolve()))
import importance278 as imp  # noqa: E402


def test_hand_weighted_example():
    # SID prefix1: items 0..3 -> 1, items 4,5 -> 2
    prefix = np.array([1, 1, 1, 1, 2, 2])
    anchor = np.array([0, 4])
    target = np.array([2, 5])
    row_idx = np.array([0, 0, 1, 1])
    item_idx = np.array([0, 2, 4, 5])
    population = np.array([4, 4])
    out, hp, hn = imp.weighted_level(prefix, anchor, target, row_idx, item_idx, population,
                                     {'uniform': np.ones(2), 'w': np.array([1.0, 3.0])})
    assert list(hp) == [1.0, 1.0]
    assert list(hn) == [0.5, 0.0]
    assert abs(out['uniform']['P_w'] - 1.0) < 1e-12
    assert abs(out['uniform']['N_w'] - 0.25) < 1e-12
    assert abs(out['uniform']['ER_w'] - 4.0) < 1e-12
    assert abs(out['w']['P_w'] - 1.0) < 1e-12
    assert abs(out['w']['N_w'] - 0.125) < 1e-12
    assert abs(out['w']['ER_w'] - 8.0) < 1e-12


def test_uniform_reproduces_the_frozen_evaluator():
    table = json.load(open('results/research/issue276/SID_TABLE.json'))
    feats = None
    sid_path = None
    for g in table['groups']:
        for r in g['rows']:
            f = (r.get('features') or {}).get('_source')
            if not f or not Path(f).is_file():
                continue
            rec = json.load(open(f))
            sp = (rec.get('sid') or {}).get('path')
            if sp and Path(sp).is_file() and r['label_recall@10'] is not None:
                feats, sid_path = rec, sp
                break
        if feats:
            break
    assert feats is not None
    va = pq.read_table('results/stage0_build_parquet/valid.parquet',
                       columns=['history', 'seen_history', 'target']).to_pylist()
    n = len(va)
    anchor = np.array([r['history'][-1] for r in va], dtype=np.int64)
    target = np.array([r['target'] for r in va], dtype=np.int64)
    row_idx, item_idx, n_exc = [], [], np.zeros(n, dtype=np.int64)
    for i, r in enumerate(va):
        s = set(r['history']) | set(r.get('seen_history') or []) | {int(r['target'])}
        n_exc[i] = len(s)
        row_idx.extend([i] * len(s)); item_idx.extend(s)
    row_idx = np.asarray(row_idx); item_idx = np.asarray(item_idx)
    population = imp.N_ITEMS - n_exc
    raw = json.loads(Path(sid_path).read_text())
    p1 = np.zeros(imp.N_ITEMS, dtype=np.int64)
    for k, v in raw.items():
        p1[int(k)] = int(v[0])
    out, _, _ = imp.weighted_level(p1, anchor, target, row_idx, item_idx, population,
                                   {'uniform': np.ones(n)})
    val = feats['validation']
    assert abs(out['uniform']['P_w'] - val['l1_positive_hit_rate']) < 1e-9
    assert abs(out['uniform']['N_w'] - val['l1_negative_hit_probability']) < 1e-9
    assert abs(out['uniform']['ER_w'] - val['l1_hard_er']) < 1e-9


if __name__ == '__main__':
    test_hand_weighted_example()
    test_uniform_reproduces_the_frozen_evaluator()
    print('all weighted checks passed')
