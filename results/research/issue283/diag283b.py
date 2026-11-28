"""Issue #283: non-tautological validation-composition diagnostic.

The same-level hit is mechanically determined by the partition (ALL/AB => hit), so the hit-difference
in VALIDATION_CONTEXT.json is a tautology check, not an effect. This script reports the nontrivial
parts: (i) coverage selection (hit rate on covered vs uncovered pairs), (ii) hit decomposition into
ALL vs AB (where the intermediate sits), (iii) SID-level correlations of the validation ALL rate with
recall and with the #277 l1_pos_hit (redundancy), with BH FDR.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pyarrow.parquet as pq
from scipy import stats

sys.path.insert(0, '.')
sys.path.insert(0, 'results/research/issue283')
from results.research.issue283 import joint283 as J
from tools.graph_infra.assets import load_sparse

RELATIONS = ('chain', 'revchain', 'merge', 'fork')
FEATS8 = ['l1_pos_hit', 'l1_neg_hit', 'l1_hard_er', 'l2_pos_hit', 'l2_neg_hit',
          'l2_distinct', 'l3_collision', 'l3_refine_bits']


def main() -> None:
    g = load_sparse('assets/train/visible_forward.npz').tocsr(); gt = g.T.tocsr()
    va = pq.read_table('results/stage0_build_parquet/valid.parquet',
                       columns=['user', 'history', 'seen_history', 'target']).to_pylist()
    n = len(va)
    anchor = np.array([r['history'][-1] for r in va], dtype=np.int64)
    target = np.array([r['target'] for r in va], dtype=np.int64)
    users = np.array([r['user'] for r in va])
    seen = np.array([int(t) in set(g.indices[g.indptr[int(a)]:g.indptr[int(a) + 1]].tolist())
                     for a, t in zip(anchor, target)])

    def row(csr, i):
        return csr.indices[csr.indptr[int(i)]:csr.indptr[int(i) + 1]]

    ROW = {}; MID = {}
    for rel in RELATIONS:
        X, Y = {'chain': (g, gt), 'revchain': (gt, g), 'merge': (g, g), 'fork': (gt, gt)}[rel]
        rr, mm = [], []
        for i, (a, t) in enumerate(zip(anchor.tolist(), target.tolist())):
            ms = np.intersect1d(row(X, a), row(Y, t))
            if len(ms):
                rr.append(np.full(len(ms), i, dtype=np.int64)); mm.append(ms.astype(np.int64))
        ROW[rel] = np.concatenate(rr) if rr else np.zeros(0, dtype=np.int64)
        MID[rel] = np.concatenate(mm) if mm else np.zeros(0, dtype=np.int64)

    table = json.load(open('results/research/issue276/SID_TABLE.json'))
    paths = []; seen_sha = set(); labels = {}; l1feat = {}
    for grp in table['groups']:
        for r in grp['rows']:
            sha = r['sid_sha256']
            if sha in seen_sha: continue
            seen_sha.add(sha)
            rec = json.loads(open(r['features']['_source']).read())
            paths.append((sha, rec['sid']['path']))
            labels[sha] = r.get('label_recall@10') if isinstance(r.get('label_recall@10'), (int, float)) else None
            if all(isinstance((r.get('features') or {}).get(k), (int, float)) for k in FEATS8):
                l1feat[sha] = float(r['features']['l1_pos_hit'])
    out = {'schema': 'genrec.issue283.validation_composition/1', 'n_rows': n,
           'tautology_note': 'same-level AL L/AB implies hit; AM/MB/NONE implies no hit',
           'coverage_selection': {}, 'decomposition': {}, 'sid_correlations': {}, 'sids': {}}
    for rel in RELATIONS:
        rr, mm = ROW[rel], MID[rel]
        cov = np.zeros(n, dtype=bool); cov[rr] = True
        out['coverage_selection'][rel] = {}
        for subset_name, subset in (('unseen', ~seen), ('seen', seen)):
            # hit rate covered vs uncovered needs a per-SID hit; store coverage only here
            out['coverage_selection'][rel][subset_name] = {
                'covered': int((cov & subset).sum()), 'uncovered': int((~cov & subset).sum())}
    for sha, path in paths:
        codes = J.codes_matrix(path)
        hit = (codes[anchor, 0] == codes[target, 0])
        rec = {}
        for rel in RELATIONS:
            rr, mm = ROW[rel], MID[rel]
            w = 1.0 / np.bincount(rr, minlength=n)[rr]
            for k in range(1, 4):
                ca = codes[anchor[rr], k - 1]; cm = codes[mm, k - 1]; cb = codes[target[rr], k - 1]
                pat = J.pattern_of(ca == cm, cm == cb, ca == cb)
                key = (rr * 5 + pat)
                oned = np.bincount(key, weights=w, minlength=n * 5).reshape(n, 5)
                # normalize by the row's support weight (sum of w over its instances)
                tot = oned.sum(axis=1)
                nz = tot > 0
                prob = oned[nz] / tot[nz, None]
                hh = hit[nz]; ss = (~seen)[nz]
                for subset_name, mask in (('unseen', ss), ('seen', ~ss)):
                    if mask.sum() < 10: continue
                    p = prob[mask]; h = hh[mask]
                    comp = {'rows': int(mask.sum()),
                            'hit_rate': round(float(h.mean()), 6),
                            'all_share_of_hits': round(float(p[h, 0].sum() / max(h.sum(), 1)), 6),
                            'ab_share_of_hits': round(float(p[h, 3].sum() / max(h.sum(), 1)), 6),
                            'mean_pattern_hit': [round(float(v), 5) for v in p[h].mean(axis=0)] if h.any() else None,
                            'mean_pattern_miss': [round(float(v), 5) for v in p[~h].mean(axis=0)] if (~h).any() else None,
                            'mean_all': round(float(p[:, 0].mean()), 6)}
                    rec.setdefault(subset_name, {}).setdefault(rel, {})[f'level{k}'] = comp
                if k == 1:
                    rec.setdefault('unseen', {}).setdefault(rel, {})['level1_all_rate'] = \
                        round(float(prob[ss, 0].mean()), 6)
        rec['coverage_hit'] = {}
        for rel in RELATIONS:
            rr = ROW[rel]
            cov = np.zeros(n, dtype=bool); cov[rr] = True
            rec['coverage_hit'][rel] = {
                'hit_covered': round(float(hit[cov & ~seen].mean()), 6) if (cov & ~seen).any() else None,
                'hit_uncovered': round(float(hit[~cov & ~seen].mean()), 6) if (~cov & ~seen).any() else None}
        out['sids'][sha] = rec
    # SID-level correlations: validation ALL rate (unseen, chain/merge level 1) vs recall / l1_pos_hit
    tests = []
    for rel in ('chain', 'merge'):
        vals = np.array([out['sids'][sha]['unseen'][rel]['level1_all_rate'] for sha, _ in paths])
        ys = np.array([labels[sha] if labels[sha] is not None else np.nan for sha, _ in paths], float)
        xs = np.array([l1feat.get(sha, np.nan) for sha, _ in paths], float)
        for name, other in (('recall', ys), ('l1_pos_hit', xs)):
            m = np.isfinite(vals) & np.isfinite(other)
            if m.sum() > 10 and np.std(vals[m]) > 0 and np.std(other[m]) > 0:
                rho = float(stats.spearmanr(vals[m], other[m]).correlation)
                pv = float(2 * stats.t.sf(abs(rho) * np.sqrt(m.sum() - 2) / np.sqrt(max(1 - rho * rho, 1e-12)),
                                          df=m.sum() - 2))
                tests.append((f'{rel}_all1_vs_{name}', rho, pv, int(m.sum())))
    order = sorted(range(len(tests)), key=lambda i: tests[i][2])
    qs = [0.0] * len(tests); running = 1.0
    for rank, i in enumerate(reversed(order), 1):
        pass
    # BH
    m_tests = len(tests); prev = 1.0
    qval = [1.0] * m_tests
    for rank, i in enumerate(order, 1):
        qval[i] = min(1.0, tests[i][2] * m_tests / rank)
    for i in reversed(range(m_tests)):
        pass
    # monotone enforce
    order_rev = list(reversed(order)); run = 1.0
    for i in order_rev:
        run = min(run, qval[i]); qval[i] = run
    out['sid_correlations'] = {t[0]: {'spearman': round(t[1], 4), 'p': round(t[2], 6),
                                      'bh_q': round(qval[i], 6), 'n': t[3]}
                               for i, t in enumerate(tests)}
    (J.P / 'VALIDATION_COMPOSITION.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({'coverage_hit': out['sids'][paths[0][0]]['coverage_hit'],
                      'sid_correlations': out['sid_correlations']}, indent=1))


if __name__ == '__main__':
    main()
