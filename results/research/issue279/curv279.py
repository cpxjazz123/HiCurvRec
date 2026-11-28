"""Issue #279 — curvature diagnostic on a fixed 20k relation sample (CONTRACT.md §2).

For each sampled pair (a,b): common centres m in N(a)∩N(b) (undirected training graph), the asset's
triple (m; min(a,b), max(a,b)) curvature (valid rows only), mean + structure-class mode. AUC on the
covered subset; stratification by the 2-hop count. No model selection uses curvature.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy import stats

import importlib.util
spec = importlib.util.spec_from_file_location('rl', 'results/research/issue279/rellevel279.py')
rl = importlib.util.module_from_spec(spec); spec.loader.exec_module(rl)
N = rl.N
P = Path('results/research/issue279')
SEED = 42


def main() -> None:
    A, B, C, succ, pred, c_from, c_to = rl.load_graph()
    und = {k: set(v) for k, v in succ.items()}
    for k, v in pred.items():
        und.setdefault(k, set()).update(v)
    idx = np.load('assets/index.npz', allow_pickle=True)
    item_shard = idx['item_shard']; item_start = idx['item_start']; item_count = idx['item_count']
    names = [str(x) for x in idx['shard_names']]
    class_names = [str(x) for x in idx['structure_class_names']]
    shards = {}

    def shard(i):
        if i not in shards:
            d = np.load(f'assets/shards/{names[i]}', allow_pickle=True)
            shards[i] = {'centre': d['centre'], 'key': d['b'].astype(np.int64) * N + d['c'].astype(np.int64),
                         'curv': d['curvature'], 'valid': d['valid'], 'cls': d['structure_class']}
        return shards[i]

    def pair_curv(a, b):
        lo, hi = (a, b) if a < b else (b, a)
        centres = und.get(a, set()) & und.get(b, set())
        vals, classes = [], []
        for m in centres:
            i = int(item_shard[m]); s = shard(i)
            st, ct = int(item_start[m]), int(item_count[m])
            key = lo * N + hi
            pos = int(np.searchsorted(s['key'], key))
            if pos >= st + ct or s['key'][pos] != key:
                continue
            if not bool(s['valid'][pos]):
                continue
            vals.append(float(s['curv'][pos])); classes.append(int(s['cls'][pos]))
        if not vals:
            return None, None, 0
        return float(np.mean(vals)), (Counter(classes).most_common(1)[0][0] if classes else None), len(vals)

    va = pq.read_table('results/stage0_build_parquet/valid.parquet',
                       columns=['history', 'seen_history', 'target']).to_pylist()
    rng = np.random.default_rng(SEED)
    n = len(va)
    rows = rng.choice(n, min(10000, n), replace=False)
    pos_c, neg_c = [], []
    for i in rows.tolist():
        r = va[i]; a = int(r['history'][-1]); b = int(r['target'])
        pc = pair_curv(a, b)
        excl = set(r['history']) | set(r.get('seen_history') or []) | {b}
        while True:
            nb = int(rng.integers(0, N))
            if nb not in excl:
                break
        nc = pair_curv(a, nb)
        pos_c.append(pc[0]); neg_c.append(nc[0])
    pos = np.array([x if x is not None else np.nan for x in pos_c], float)
    neg = np.array([x if x is not None else np.nan for x in neg_c], float)
    cov_pos = np.isfinite(pos); cov_neg = np.isfinite(neg)
    both = cov_pos & cov_neg
    auc = rl.auc_rank(pos[both], neg[both]) if both.sum() > 10 else None
    out = {'schema': 'genrec.issue279.curvature_diag/1', 'n_sample': len(rows),
           'coverage_pos': float(cov_pos.mean()), 'coverage_neg': float(cov_neg.mean()),
           'n_pairs_with_both': int(both.sum()), 'auc_covered': auc,
           'median_pos': float(np.nanmedian(pos)), 'median_neg': float(np.nanmedian(neg)),
           'note': 'curvature is a diagnostic only; the asset covers pairs with a common centre and a '
                   'legal triple; the graph is undirected for the centre set, so it is not directional'}
    (P / 'CURVATURE_DIAG.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
