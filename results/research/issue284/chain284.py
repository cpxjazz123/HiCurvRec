"""Issue #284: per-SID chain table (recompute + verify vs #283), conditional refinement,
collinearity, structure contrast, residual association and frozen conditional bands."""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, '.')
sys.path.insert(0, 'results/research/issue284')
import common284 as C
from results.research.issue283 import joint283 as J

P = Path('results/research/issue284')
G = {}


def setup():
    C.load_row_tables()
    G['table'] = json.load(open('results/research/issue276/SID_TABLE.json'))
    G['join'] = json.load(open('results/research/issue283/JOIN_TABLE.json'))
    paths = {}
    for g in G['table']['groups']:
        for r in g['rows']:
            src = (r.get('features') or {}).get('_source')
            if src and r['sid_sha256'] not in paths:
                rec = json.loads(Path(src).read_text())
                paths[r['sid_sha256']] = rec['sid']['path']
    G['paths'] = paths


def worker(sha):
    codes = J.codes_matrix(G['paths'][sha])
    f = C.chain_features(codes)
    # verification against #283 JOIN_TABLE (chain class, k=1..3 counts)
    jt = G['join']['entries'].get(sha)
    ok = True
    if jt:
        cc = C.chain_counts(codes, ('chain',))
        for k in (1, 2, 3):
            if cc['chain']['counts'][k] != jt['classes']['chain'][str(k)]['counts']:
                ok = False
    f['verify_283'] = bool(ok)
    return sha, f


def cluster_spearman(x, y, unit, reps=2000, seed=42):
    rng = np.random.default_rng(seed)
    uniq = sorted(set(unit))
    idx_of = {u: np.where(np.array(unit) == u)[0] for u in uniq}
    vals = []
    for _ in range(reps):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([idx_of[int(k)] for k in pick])
        if np.std(x[idx]) == 0 or np.std(y[idx]) == 0:
            continue
        vals.append(float(stats.spearmanr(x[idx], y[idx]).correlation))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else None


def main() -> None:
    setup()
    shas = sorted(G['paths'])
    if '--reuse' in sys.argv and (P / 'CHAIN_TABLE.json').exists():
        feats = json.load(open(P / 'CHAIN_TABLE.json'))['entries']
        print('reused CHAIN_TABLE.json', len(feats), flush=True)
    else:
        with mp.Pool(8) as pool:
            results = pool.map(worker, shas, chunksize=1)
        feats = dict(results)
        (P / 'CHAIN_TABLE.json').write_text(json.dumps(
            {'schema': 'genrec.issue284.chain_table/1', 'entries': feats}, indent=1, default=float) + '\n')
    (P / 'CHAIN_TABLE.json').write_text(json.dumps(
        {'schema': 'genrec.issue284.chain_table/1', 'entries': feats}, indent=1, default=float) + '\n')
    bad = [s for s, f in feats.items() if not f['verify_283']]
    print('sid count', len(feats), 'recompute mismatches', len(bad), bad[:3], flush=True)
    # labelled rows joined with chain features
    rows = C.table_rows()
    unit = C.clusters(rows)
    y = np.array([r['y'] for r in rows])
    x6 = np.array([[r['x6'][k] for k in C.FEATS6] for r in rows])
    fold = np.array([C.table_rows.folds['assignment'][r['group']] for r in rows])
    A = C.oof(x6, y, fold)
    resid = y - A
    ch = {r['sha']: feats[r['sha']] for r in rows}
    all1 = np.array([ch[r['sha']]['all1_chain'] for r in rows])
    out = {'schema': 'genrec.issue284.chain_analysis/1',
           'recompute': {'n_sids': len(feats), 'mismatches': len(bad)},
           'n_rows': len(rows), 'n_units': len(set(unit))}
    # residual association + collinearity
    def sp(x, v, m=None):
        m = np.isfinite(v) if m is None else (m & np.isfinite(v))
        if m.sum() < 10 or np.std(x[m]) == 0 or np.std(v[m]) == 0:
            return None
        return round(float(stats.spearmanr(x[m], v[m]).correlation), 4)
    out['residual'] = {'spearman_all1_resid': sp(all1, resid),
                       'spearman_all1_resid_ci': cluster_spearman(all1, resid, unit)}
    coll = {'all1_vs_y': sp(all1, y)}
    for k in C.FEATS6 + ['l2_pos_hit', 'l2_neg_hit']:
        v = np.array([r['x8'].get(k, np.nan) for r in rows])
        coll[f'all1_vs_{k}'] = sp(all1, v)
    coll['all1_vs_l1_entropy'] = sp(all1, np.array([ch[r['sha']]['l1_entropy'] for r in rows]))
    coll['all1_vs_l1_distinct'] = sp(all1, np.array([ch[r['sha']]['l1_distinct'] for r in rows], float))
    out['collinearity'] = coll
    # structure contrast
    struct = {}
    for name in ('all1_chain', 'all1_fork', 'all1_merge', 'all1_revchain', 'all1_chain_both'):
        v = np.array([ch[r['sha']].get(name, np.nan) for r in rows], float)
        struct[name] = {'spearman_y': sp(v, y),
                        'ci': cluster_spearman(v, y, unit) if np.isfinite(v).all() else None}
    out['structure'] = struct
    # all1 vs ALL_norm vs cond_refine
    for name in ('all1_chain', 'all_norm', 'cond_refine', 'p3_split_given_l1_all'):
        v = np.array([ch[r['sha']].get(name, np.nan) for r in rows], float)
        out.setdefault('feature_y', {})[name] = {'spearman_y': sp(v, y),
                                                 'ci': cluster_spearman(v, y, unit) if np.isfinite(v).all() else None}
    # bands (feature-only terciles) of L1 neg, L2 pos, L3 refine, collision
    bands = {}
    for feat in ('l1_neg_hit', 'l2_pos_hit', 'l3_refine_bits', 'l3_collision'):
        v = np.array([r['x8'][feat] for r in rows], float)
        q = np.quantile(v, [1 / 3, 2 / 3])
        band = np.clip(np.searchsorted(q, v, side='right'), 0, 2)
        entry = {'edges': [round(float(x), 6) for x in q], 'bands': {}}
        for b in range(3):
            m = band == b
            if m.sum() < 10:
                entry['bands'][str(b)] = {'n': int(m.sum()), 'spearman': None, 'ci': None,
                                          'mean_all1': round(float(all1[m].mean()), 5)}
                continue
            entry['bands'][str(b)] = {
                'n': int(m.sum()), 'mean_all1': round(float(all1[m].mean()), 5),
                'spearman': sp(all1, y, m),
                'ci': cluster_spearman(all1[m], y[m], [u for u, sel in zip(unit, m) if sel])}
        bands[feat] = entry
    out['bands'] = bands
    # high/low recall split
    med = float(np.median(y))
    out['recall_split'] = {'median': med}
    for name in ('all1_chain', 'all_norm', 'cond_refine', 'all1_fork', 'all1_merge'):
        v = np.array([ch[r['sha']].get(name, np.nan) for r in rows], float)
        out['recall_split'][name] = {'low_mean': round(float(v[y <= med].mean()), 5),
                                     'high_mean': round(float(v[y > med].mean()), 5)}
    strong = np.array([r['strong'] for r in rows])
    out['strong_split'] = {'strong_mean_all1': round(float(all1[strong].mean()), 5),
                           'weak_mean_all1': round(float(all1[~strong].mean()), 5)}
    (P / 'CHAIN_ANALYSIS.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: out[k] for k in ('recompute', 'residual', 'collinearity', 'structure')}, indent=1)[:3000])


if __name__ == '__main__':
    main()
