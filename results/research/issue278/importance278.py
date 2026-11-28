"""Issue #278 — training-behaviour transition importance and importance-weighted SID fidelity.

Frozen definitions live in CONTRACT.md. This script only computes them; no label is read here.
Writes IMPORTANCE.json (graph weights, coverage, strata) and WEIGHTED_TABLE.json (per-SID P_w/N_w/ER_w
for L1 and L1+L2 under uniform / w_freq / w_lift, plus per-stratum tables).
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path('.').resolve()
P = REPO / 'results/research/issue278'
TRAIN = REPO / 'results/stage0_build_parquet/train.parquet'
VALID = REPO / 'results/stage0_build_parquet/valid.parquet'
N_ITEMS = 24587
ALPHA = 1.0
STRATA_QS = (33, 66)


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def weighted_level(prefix, anchor, target, row_idx, item_idx, population, weights):
    """CONTRACT.md section 2: P_w, N_w, ER_w for every weight array. Returns (results, h+, h-)."""
    n = len(anchor)
    cnt = np.bincount(prefix, minlength=int(prefix.max()) + 1)
    hp = (prefix[anchor] == prefix[target]).astype(np.float64)
    same = (prefix[item_idx] == prefix[anchor][row_idx]).astype(np.float64)
    excl = np.bincount(row_idx, weights=same, minlength=n)
    eff = population > 0
    hn = np.where(eff, (cnt[prefix[anchor]] - excl) / np.maximum(population, 1), 0.0)
    out = {}
    for name, w in weights.items():
        wm = np.where(eff & (w > 0), w, 0.0)
        sw = wm.sum()
        P_w = float((wm * hp).sum() / sw) if sw > 0 else None
        N_w = float((wm * hn).sum() / sw) if sw > 0 else None
        out[name] = {'P_w': P_w, 'N_w': N_w,
                     'ER_w': (P_w / N_w) if (P_w is not None and N_w not in (None, 0)) else None,
                     'D_w': (P_w - N_w) if (P_w is not None and N_w is not None) else None}
    return out, hp, hn


def main() -> None:
    P.mkdir(parents=True, exist_ok=True)
    tr = pq.read_table(TRAIN, columns=['history', 'seen_history', 'target']).to_pylist()
    hist = [r['history'] for r in tr]
    targ = np.array([r['target'] for r in tr], dtype=np.int64)

    # ---- directed edge counts from training behaviour only ----
    ea, eb = [], []
    for h in hist:
        if len(h) >= 2:
            arr = np.asarray(h, dtype=np.int64)
            ea.append(arr[:-1]); eb.append(arr[1:])
    last = np.array([h[-1] if h else -1 for h in hist], dtype=np.int64)
    ok = last >= 0
    ea.append(last[ok]); eb.append(targ[ok])
    ea = np.concatenate(ea); eb = np.concatenate(eb)
    keys = ea * N_ITEMS + eb
    uk, cnt = np.unique(keys, return_counts=True)
    A = (uk // N_ITEMS).astype(np.int64)
    B = (uk % N_ITEMS).astype(np.int64)
    C = cnt.astype(np.float64)
    out = np.bincount(A, weights=C, minlength=N_ITEMS)
    inn = np.bincount(B, weights=C, minlength=N_ITEMS)
    E = float(C.sum())
    q33, q66 = np.percentile(C, STRATA_QS).tolist()

    # fast lookup: sorted keys (uk is sorted by construction)
    def edge_count(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        k = a * N_ITEMS + b
        pos = np.searchsorted(uk, k)
        pos = np.clip(pos, 0, len(uk) - 1)
        hit = uk[pos] == k
        return np.where(hit, C[pos], 0.0)

    succ = defaultdict(list)
    pred = defaultdict(list)
    for a, b in zip(A.tolist(), B.tolist()):
        succ[a].append(b)
        pred[b].append(a)
    succ_set = {k: set(v) for k, v in succ.items()}
    pred_set = {k: set(v) for k, v in pred.items()}

    # ---- validation rows and weights ----
    va = pq.read_table(VALID, columns=['history', 'seen_history', 'target']).to_pylist()
    n = len(va)
    anchor = np.array([r['history'][-1] for r in va], dtype=np.int64)
    target = np.array([r['target'] for r in va], dtype=np.int64)
    c_ab = edge_count(anchor, target)
    p_cond = (c_ab + ALPHA) / (out[anchor] + ALPHA * N_ITEMS)
    p_node = (inn[target] + ALPHA) / (E + ALPHA * N_ITEMS)
    w_lift = np.clip(p_cond / p_node, 1e-3, 1e3)
    w_freq = np.log1p(c_ab)
    w_sqrt = np.sqrt(w_freq)

    # excluded items (unique) and population per row
    row_idx, item_idx = [], []
    n_excluded = np.zeros(n, dtype=np.int64)
    for i, r in enumerate(va):
        s = set(r['history']) | set(r.get('seen_history') or []) | {int(r['target'])}
        n_excluded[i] = len(s)
        row_idx.extend([i] * len(s))
        item_idx.extend(s)
    row_idx = np.asarray(row_idx, dtype=np.int64)
    item_idx = np.asarray(item_idx, dtype=np.int64)
    population = N_ITEMS - n_excluded
    effective = population > 0  # every valid row has non-empty history here

    # ---- per-SID weighted fidelity ----
    table = json.load(open('results/research/issue276/SID_TABLE.json'))
    rows = [r for g in table['groups'] for r in g['rows']]
    seen_sha = set()
    out_rows = []
    no_sid = []
    for r in rows:
        sha = r['sid_sha256']
        if sha in seen_sha:
            continue
        seen_sha.add(sha)
        fpath = (r.get('features') or {}).get('_source')
        if not fpath or not Path(fpath).is_file():
            no_sid.append(sha)
            continue
        frec = json.load(open(fpath))
        sid_path = Path((frec.get('sid') or {}).get('path') or '')
        if not sid_path.is_file():
            no_sid.append(sha)
            continue
        raw = json.loads(sid_path.read_text())
        codes = np.zeros((N_ITEMS, 4), dtype=np.int64)
        for k, v in raw.items():
            codes[int(k), :len(v)] = v[:4]
        p1 = codes[:, 0]
        p2 = codes[:, 0] * 256 + codes[:, 1]
        rec = {'sid_sha256': sha, 'sid_path': str(sid_path), 'arms': [a.get('arm') for a in r.get('arms', [])][:3],
               'label_recall@10': r.get('label_recall@10'), 'label_strength': r.get('label_strength'),
               'identity': sorted({a.get('identity') for a in r.get('arms', []) if a.get('identity')})}
        for level, prefix in (('L1', p1), ('L2', p2)):
            level_out, hp, hn = weighted_level(prefix, anchor, target, row_idx, item_idx, population,
                                               {'uniform': np.ones(n), 'w_freq': w_freq, 'w_lift': w_lift,
                                                'w_sqrt': w_sqrt,
                                                'w_freq_clip99': np.minimum(w_freq, np.percentile(w_freq, 99))})
            strata = {}
            for sname, mask in (('unseen', c_ab == 0), ('low', (c_ab > 0) & (c_ab <= q33)),
                                ('mid', (c_ab > q33) & (c_ab <= q66)), ('high', c_ab > q66)):
                m = mask & effective
                strata[sname] = {'n': int(m.sum()),
                                 'pos': float(hp[m].mean()) if m.any() else None,
                                 'neg': float(hn[m].mean()) if m.any() else None}
            rec[level] = {'all': level_out, 'strata': strata}
        rec['n_effective'] = int(effective.sum())
        out_rows.append(rec)

    imp = {
        'schema': 'genrec.issue278.importance/1',
        'training_rows': len(tr), 'edges_total': float(E), 'distinct_edges': int(len(uk)),
        'q33': q33, 'q66': q66,
        'weights': {
            'w_freq': {'min': float(w_freq.min()), 'max': float(w_freq.max()),
                       'mean': float(w_freq.mean()), 'p50': float(np.percentile(w_freq, 50)),
                       'p99': float(np.percentile(w_freq, 99)),
                       'nonzero_rows': int((c_ab > 0).sum()), 'zero_rows': int((c_ab == 0).sum())},
            'w_lift': {'p01': float(np.percentile(w_lift, 1)), 'p50': float(np.percentile(w_lift, 50)),
                       'p99': float(np.percentile(w_lift, 99))},
        },
        'strata_counts_rows': {s: int(((c_ab == 0) if s == 'unseen' else
                                      ((c_ab > 0) & (c_ab <= q33)) if s == 'low' else
                                      ((c_ab > q33) & (c_ab <= q66)) if s == 'mid' else
                                      (c_ab > q66)).sum()) for s in ('unseen', 'low', 'mid', 'high')},
        'coverage': {'table_rows': len(rows), 'with_sid_file': len(out_rows),
                     'without_sid_file': len(no_sid),
                     'without_sid_sha256_examples': no_sid[:10]},
        'inputs': {'train': {'path': str(TRAIN), 'sha256': sha_file(TRAIN)},
                   'valid': {'path': str(VALID), 'sha256': sha_file(VALID)}},
    }
    (P / 'IMPORTANCE.json').write_text(json.dumps(imp, indent=1) + '\n')
    (P / 'WEIGHTED_TABLE.json').write_text(json.dumps({
        'schema': 'genrec.issue278.weighted_table/1', 'definition': 'CONTRACT.md section 2',
        'strata_thresholds': {'q33': q33, 'q66': q66}, 'rows': out_rows}, indent=1) + '\n')
    print(json.dumps({k: imp[k] for k in ('edges_total', 'distinct_edges', 'q33', 'q66',
                                          'strata_counts_rows', 'coverage')}, indent=1))
    l1 = [r['L1']['all']['w_freq']['ER_w'] for r in out_rows if r['L1']['all']['w_freq']['ER_w'] is not None]
    u1 = [r['L1']['all']['uniform']['ER_w'] for r in out_rows if r['L1']['all']['uniform']['ER_w'] is not None]
    corr = np.corrcoef(l1, u1)[0, 1] if len(l1) > 2 else None
    diff = np.array(l1) - np.array(u1)
    print(json.dumps({'n_l1': len(l1), 'corr_uniform_vs_wfreq_ER': float(corr),
                      'mean_ER_wfreq_minus_uniform': float(diff.mean()),
                      'n_changed_beyond_1e-6': int((np.abs(diff) > 1e-6).sum())}, indent=1))


if __name__ == '__main__':
    main()
