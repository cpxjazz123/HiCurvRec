"""Issue #278 — C9 diagnostic: is the control residual explained by popularity of the SID's L1 groups?

Per SID: for every validation row take the mean training in-degree of the items in the anchor's L1
group (SID-dependent through the SID's own prefix grouping), average over the effective rows, and
correlate with the control OOF residual. Descriptive only; no model selection.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy import stats

P = Path('results/research/issue278')
N = 24587


def main() -> None:
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
    inn = np.bincount(uk % N, weights=cnt, minlength=N)

    va = pq.read_table('results/stage0_build_parquet/valid.parquet',
                       columns=['history', 'target']).to_pylist()
    anchor = np.array([r['history'][-1] for r in va], dtype=np.int64)

    table = json.load(open('results/research/issue276/SID_TABLE.json'))
    seen, sid_paths = set(), {}
    for g in table['groups']:
        for r in g['rows']:
            f = (r.get('features') or {}).get('_source')
            if f and Path(f).is_file() and r['sid_sha256'] not in seen:
                rec = json.load(open(f))
                sp = (rec.get('sid') or {}).get('path')
                if sp and Path(sp).is_file():
                    sid_paths[r['sid_sha256']] = sp
                    seen.add(r['sid_sha256'])
    pop = {}
    for sha, sp in sid_paths.items():
        raw = json.loads(Path(sp).read_text())
        p1 = np.zeros(N, dtype=np.int64)
        for k, v in raw.items():
            p1[int(k)] = int(v[0])
        G = int(p1.max()) + 1
        gsum = np.bincount(p1, weights=inn, minlength=G)
        gcnt = np.bincount(p1, minlength=G)
        gmean = np.divide(gsum, gcnt, out=np.zeros(G), where=gcnt > 0)
        pop[sha] = float(gmean[p1[anchor]].mean())

    oof = json.load(open(P / 'OOF_ROWS.json'))['rows']
    resid = np.array([r['y'] - r['pred_A'] for r in oof])
    er_w = np.array([r['er_w'] for r in oof], float)
    er_u = np.array([r['er_uniform'] for r in oof], float)
    popv = np.array([pop.get(r['sid_sha256'], np.nan) for r in oof], float)
    m = np.isfinite(popv)
    out = {'schema': 'genrec.issue278.degree_diagnostic/1', 'n_rows': len(oof), 'n_with_pop': int(m.sum()),
           'spearman_resid_vs_group_popularity': float(stats.spearmanr(popv[m], resid[m]).correlation),
           'spearman_pop_vs_ERw': float(stats.spearmanr(popv[m], er_w[m]).correlation),
           'spearman_resid_vs_er_w': float(stats.spearmanr(er_w, resid).correlation),
           'spearman_resid_vs_er_uniform': float(stats.spearmanr(er_u, resid).correlation),
           'note': 'group popularity = mean training in-degree of the items in the anchor L1 group, '
                   'averaged over the validation rows; SID-dependent through the SID partitioning'}
    (P / 'DEGREE_DIAGNOSTIC.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
