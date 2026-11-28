"""Issue #279 — relation-level higher-order discrimination (positives vs fixed legal negatives).

Frozen definitions: CONTRACT.md §2/§3. Training graph only; no label is used for scoring.
Writes RELATION_LEVEL.json.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy import stats
from sklearn.metrics import average_precision_score

P = Path('results/research/issue279')
N = 24587
SEED = 42


def load_graph():
    tr = pq.read_table('results/stage0_build_parquet/train.parquet',
                       columns=['history', 'target']).to_pylist()
    ea, eb = [], []
    for r in tr:
        h = r['history']
        if len(h) >= 2:
            a = np.asarray(h, dtype=np.int64)
            ea.append(a[:-1]); eb.append(a[1:])
    last = np.array([r['history'][-1] if r['history'] else -1 for r in tr], dtype=np.int64)
    ok = last >= 0
    ea.append(last[ok]); eb.append(np.array([r['target'] for r in tr], dtype=np.int64)[ok])
    ea = np.concatenate(ea); eb = np.concatenate(eb)
    uk, cnt = np.unique(ea * N + eb, return_counts=True)
    A = (uk // N).astype(np.int64); B = (uk % N).astype(np.int64); C = cnt.astype(np.int64)
    c_from = defaultdict(dict); c_to = defaultdict(dict)
    for a, b, c in zip(A.tolist(), B.tolist(), C.tolist()):
        c_from[a][b] = c
        c_to[b][a] = c
    succ = {k: set(v) for k, v in c_from.items()}
    pred = {k: set(v) for k, v in c_to.items()}
    return A, B, C, succ, pred, c_from, c_to


def pair_scores(a, b, succ, pred, c_from, c_to):
    sa, sb, pa, pb = succ.get(a, set()), succ.get(b, set()), pred.get(a, set()), pred.get(b, set())
    inter = sa & sb
    common_succ = len(inter)
    union = sa | sb
    cs_j = common_succ / len(union) if union else 0.0
    interp = pa & pb
    common_pred = len(interp)
    unionp = pa | pb
    cp_j = common_pred / len(unionp) if unionp else 0.0
    mids = sa & pb
    two_hop = 0
    if mids:
        ca = c_from.get(a, {})
        ct = c_to.get(b, {})
        two_hop = sum(ca[m] * ct[m] for m in mids)
    direct = c_from.get(a, {}).get(b, 0)
    out_a = len(sa); in_b = len(pb)
    norm = two_hop / (np.sqrt(max(1, out_a) * max(1, in_b)) + 1.0)
    return {'two_hop': float(two_hop), 'two_hop_norm': float(norm),
            'common_succ': float(common_succ), 'cs_jaccard': float(cs_j),
            'common_pred': float(common_pred), 'cp_jaccard': float(cp_j),
            'direct': float(direct), 'out_deg': out_a, 'in_deg': in_b}


def auc_rank(pos, neg):
    """AUC via the rank formula (ties average)."""
    x = np.concatenate([pos, neg])
    r = stats.rankdata(x)
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> None:
    P.mkdir(parents=True, exist_ok=True)
    A, B, C, succ, pred, c_from, c_to = load_graph()
    va = pq.read_table('results/stage0_build_parquet/valid.parquet',
                       columns=['history', 'seen_history', 'target']).to_pylist()
    n = len(va)
    anchor = np.array([r['history'][-1] for r in va], dtype=np.int64)
    target = np.array([r['target'] for r in va], dtype=np.int64)

    # fixed legal negatives: one per row, uniform over the legal population, seed 42
    rng = np.random.default_rng(SEED)
    neg = np.empty(n, dtype=np.int64)
    for i, r in enumerate(va):
        excl = set(r['history']) | set(r.get('seen_history') or []) | {int(r['target'])}
        while True:
            cand = int(rng.integers(0, N))
            if cand not in excl:
                neg[i] = cand
                break

    keys = ['two_hop', 'two_hop_norm', 'common_succ', 'cs_jaccard', 'common_pred', 'cp_jaccard', 'direct']
    pos_s = {k: np.zeros(n) for k in keys}
    neg_s = {k: np.zeros(n) for k in keys}
    seen_pos = np.zeros(n, dtype=bool)
    for i in range(n):
        ps = pair_scores(int(anchor[i]), int(target[i]), succ, pred, c_from, c_to)
        ns = pair_scores(int(anchor[i]), int(neg[i]), succ, pred, c_from, c_to)
        for k in keys:
            pos_s[k][i] = ps[k]; neg_s[k][i] = ns[k]
        seen_pos[i] = ps['direct'] > 0

    out = {'schema': 'genrec.issue279.relation_level/1', 'n_rows': n,
           'unseen_share': float(1 - seen_pos.mean()),
           'graph': {'edges_total': int(C.sum()), 'distinct_edges': int(len(C))},
           'scores': {}}
    # degree deciles for matching
    out_deg_pos = np.array([len(succ.get(int(a), set())) for a in anchor])
    in_deg_neg = np.array([len(pred.get(int(b), set())) for b in target])
    od = np.clip(np.searchsorted(np.percentile(out_deg_pos, np.arange(10, 100, 10)), out_deg_pos), 0, 9)
    idg = np.clip(np.searchsorted(np.percentile(in_deg_neg, np.arange(10, 100, 10)), in_deg_neg), 0, 9)
    for k in keys:
        p, q = pos_s[k], neg_s[k]
        if np.all(p == 0) and np.all(q == 0):
            out['scores'][k] = {'auc': None, 'note': 'all zero'}
            continue
        auc = auc_rank(p, q)
        # anchor-clustered bootstrap
        uniq = np.unique(anchor)
        idx_of = defaultdict(list)
        for i, a in enumerate(anchor.tolist()):
            idx_of[a].append(i)
        boots = []
        for _ in range(500):
            pick = rng.choice(uniq, len(uniq), replace=True)
            sel = np.concatenate([idx_of[a] for a in pick])
            boots.append(auc_rank(p[sel], q[sel]))
        # degree-matched AUC within (out-decile x in-decile) cells
        cells = defaultdict(lambda: [[], []])
        for i in range(n):
            cells[(int(od[i]), int(idg[i]))][0].append(p[i])
            cells[(int(od[i]), int(idg[i]))][1].append(q[i])
        num = den = 0.0
        for (_, _), (pp, qq) in cells.items():
            if len(pp) > 5 and len(qq) > 5 and (np.ptp(pp) + np.ptp(qq)) > 0:
                a_cell = auc_rank(np.asarray(pp), np.asarray(qq))
                w = len(pp) * len(qq)
                num += a_cell * w; den += w
        out['scores'][k] = {
            'auc': auc, 'auc_ci95': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            'pr_auc': float(average_precision_score(np.r_[np.ones(n), np.zeros(n)], np.r_[p, q])),
            'median_pos': float(np.median(p)), 'median_neg': float(np.median(q)),
            'degree_matched_auc': (num / den) if den else None,
            'auc_seen': (auc_rank(p[seen_pos], q[seen_pos]) if seen_pos.sum() > 10 and (~seen_pos).sum() >= 0 else None),
            'auc_unseen': (auc_rank(p[~seen_pos], q[~seen_pos]) if (~seen_pos).sum() > 10 else None),
        }
    (P / 'RELATION_LEVEL.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                          for kk, vv in out['scores'][k].items()} for k in keys}, indent=1))
    print('unseen share', out['unseen_share'])


if __name__ == '__main__':
    main()
