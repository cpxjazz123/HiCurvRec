"""Issue #283: independent validation-context diagnostic (population 2, CONTRACT §1/§2).

Validation pairs (last history item -> target); the support of each relation is computed once from
the frozen visible graph. Each pair contributes weight 1 per relation split equally across its
supporting intermediates m (pre-frozen pair-uniform weighting); inverse-degree weighting is a
reported sensitivity. Aggregates are across SIDs (253 independent units), with a SID bootstrap CI.
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


def main() -> None:
    g = load_sparse('assets/train/visible_forward.npz').tocsr()
    gt = g.T.tocsr()
    und = load_sparse('assets/train/visible_undirected.npz').tocsr()
    degree = np.diff(und.indptr).astype(np.float64)
    va = pq.read_table('results/stage0_build_parquet/valid.parquet',
                       columns=['user', 'history', 'seen_history', 'target']).to_pylist()
    n = len(va)
    anchor = np.array([r['history'][-1] for r in va], dtype=np.int64)
    target = np.array([r['target'] for r in va], dtype=np.int64)
    users = np.array([r['user'] for r in va])
    seen = np.array([int(t) in set(g.indices[g.indptr[int(a)]:g.indptr[int(a) + 1]].tolist())
                     for a, t in zip(anchor, target)])
    print('rows', n, 'unseen', int((~seen).sum()), flush=True)

    def row(csr, i):
        return csr.indices[csr.indptr[int(i)]:csr.indptr[int(i) + 1]]

    rel_rows, rel_m, rel_id = [], [], []
    coverage = {}
    for rid, rel in enumerate(RELATIONS):
        X, Y = {'chain': (g, gt), 'revchain': (gt, g), 'merge': (g, g), 'fork': (gt, gt)}[rel]
        rr, mm = [], []
        for i, (a, t) in enumerate(zip(anchor.tolist(), target.tolist())):
            ms = np.intersect1d(row(X, a), row(Y, t))
            if len(ms):
                rr.append(np.full(len(ms), i, dtype=np.int64)); mm.append(ms.astype(np.int64))
        rr = np.concatenate(rr) if rr else np.zeros(0, dtype=np.int64)
        mm = np.concatenate(mm) if mm else np.zeros(0, dtype=np.int64)
        rel_rows.append(rr); rel_m.append(mm); rel_id.append(np.full(len(rr), rid, dtype=np.int8))
        coverage[rel] = {'covered_rows': int(len(set(rr.tolist()))), 'instances': int(len(rr))}
        print(rel, coverage[rel], flush=True)
    ROW = np.concatenate(rel_rows); MID = np.concatenate(rel_m); RID = np.concatenate(rel_id)
    inv_supp = np.zeros(n)
    uniq, cnt = np.unique(ROW[RID == 0], return_counts=True)
    # per (relation,row) inverse support for the sensitivity weighting
    w_sens = np.zeros(len(ROW))
    for rid in range(4):
        m = RID == rid
        w_sens[m] = (1.0 / (degree[MID[m]] + 1.0))
        rows = ROW[m]
        sums = np.bincount(rows, weights=w_sens[m], minlength=n)
        w_sens[m] = w_sens[m] / np.maximum(sums[rows], 1e-12)

    table = json.load(open('results/research/issue276/SID_TABLE.json'))
    paths = []
    seen_sha = set()
    for grp in table['groups']:
        for r in grp['rows']:
            sha = r['sid_sha256']
            if sha in seen_sha:
                continue
            seen_sha.add(sha)
            rec = json.loads(open(r['features']['_source']).read())
            paths.append((sha, rec['sid']['path']))
    out = {'schema': 'genrec.issue283.validation_context/1', 'n_rows': n,
           'unseen_rows': int((~seen).sum()), 'seen_rows': int(seen.sum()),
           'coverage': coverage,
           'weighting': {'primary': 'per (row,relation) unit weight split equally across supporting m',
                         'sensitivity': 'inverse-degree weight normalized per row'},
           'per_sid': {}}
    KEYS = [(rel, k) for rel in RELATIONS for k in range(1, 4)]
    for sha, path in paths:
        codes = J.codes_matrix(path)
        rec = {}
        hit = (codes[anchor, 0] == codes[target, 0])
        for rel in RELATIONS:
            rid = RELATIONS.index(rel)
            m = RID == rid
            rr, mm = ROW[m], MID[m]
            for k in range(1, 4):
                ca = codes[anchor[rr], k - 1]; cm = codes[mm, k - 1]; cb = codes[target[rr], k - 1]
                pat = J.pattern_of(ca == cm, cm == cb, ca == cb)
                w = 1.0 / np.bincount(rr, minlength=n)[rr]
                key = (rid * n + rr) * 5 + pat
                prob = np.bincount(key, weights=w, minlength=4 * n * 5).reshape(4, n, 5)
                probs = np.bincount(key, weights=w_sens[m], minlength=4 * n * 5).reshape(4, n, 5)
                cov = np.bincount(rr, minlength=n) > 0
                for subset_name, subset in (('unseen', ~seen & cov), ('seen', seen & cov)):
                    if subset.sum() < 10:
                        continue
                    p_all = prob[rid, subset, 0]
                    any_all = p_all > 0
                    h = hit[subset]
                    d = {'subset_rows': int(subset.sum())}
                    d['mean_pattern'] = [round(float(v), 6) for v in prob[rid, subset].mean(axis=0)]
                    d['mean_pattern_sens'] = [round(float(v), 6) for v in probs[rid, subset].mean(axis=0)]
                    d['any_all'] = round(float(any_all.mean()), 6)
                    d['hit_all'] = round(float(h[any_all].mean()), 6) if any_all.any() else None
                    d['hit_not_all'] = round(float(h[~any_all].mean()), 6) if (~any_all).any() else None
                    d['hit_diff'] = (round(d['hit_all'] - d['hit_not_all'], 5)
                                     if d['hit_all'] is not None and d['hit_not_all'] is not None else None)
                    d['spearman'] = (round(float(stats.spearmanr(p_all, h).correlation), 4)
                                     if len(set(p_all.tolist())) > 1 else None)
                    rec.setdefault(subset_name, {}).setdefault(rel, {})[f'level{k}'] = d
        out['per_sid'][sha] = rec
    # across-SID aggregates with a SID bootstrap
    agg = {'unseen': {}, 'seen': {}}
    rng = np.random.default_rng(42)
    for subset_name in ('unseen', 'seen'):
        for rel in RELATIONS:
            for k in range(1, 4):
                vals = [out['per_sid'][sha][subset_name][rel][f'level{k}']['hit_diff']
                        for sha in out['per_sid']
                        if subset_name in out['per_sid'][sha] and rel in out['per_sid'][sha][subset_name]
                        and f'level{k}' in out['per_sid'][sha][subset_name][rel]
                        and out['per_sid'][sha][subset_name][rel][f'level{k}']['hit_diff'] is not None]
                if not vals:
                    continue
                v = np.asarray(vals, dtype=float)
                boots = [float(np.mean(v[rng.integers(0, len(v), len(v))])) for _ in range(2000)]
                agg[subset_name].setdefault(rel, {})[f'level{k}'] = {
                    'n_sids': int(len(v)), 'mean': round(float(v.mean()), 5),
                    'median': round(float(np.median(v)), 5),
                    'positive': int((v > 0).sum()), 'negative': int((v < 0).sum()),
                    'ci95': [round(float(np.percentile(boots, 2.5)), 5),
                             round(float(np.percentile(boots, 97.5)), 5)]}
    out['aggregate'] = agg
    (J.P / 'VALIDATION_CONTEXT.json').write_text(json.dumps(out, indent=1) + '\n')
    print('written', len(out['per_sid']))


if __name__ == '__main__':
    main()
