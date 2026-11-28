"""Issue #284: predictive candidates A/B/C/D, #283 reproduction, paired CIs, BH and power."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '.')
sys.path.insert(0, 'results/research/issue284')
import common284 as C

P = Path('results/research/issue284')


def main() -> None:
    rows = C.table_rows()
    folds = C.table_rows.folds
    ch = json.load(open(P / 'CHAIN_TABLE.json'))['entries']
    jt = json.load(open('results/research/issue283/JOIN_TABLE.json'))['entries']
    y = np.array([r['y'] for r in rows])
    fold = np.array([folds['assignment'][r['group']] for r in rows])
    X = np.array([[r['x6'][k] for k in C.FEATS6] for r in rows])
    strong = np.array([r['strong'] for r in rows])
    unit = C.clusters(rows)
    cf = [ch[r['sha']] for r in rows]
    feat = {'all1_chain': np.array([c['all1_chain'] for c in cf]),
            'all_norm': np.array([c['all_norm'] for c in cf]),
            'cond_refine': np.array([c.get('cond_refine') or 0.0 for c in cf]),
            'all1_chain_lift': np.array([jt[r['sha']]['classes']['chain']['1']['lift']['all_u'] for r in rows]),
            'all3_chain_lift': np.array([jt[r['sha']]['classes']['chain']['3']['lift']['all_u'] for r in rows]),
            'am_mb_chain': np.array([(np.asarray(jt[r['sha']]['classes']['chain']['1']['counts'], float)[1] -
                                      np.asarray(jt[r['sha']]['classes']['chain']['1']['counts'], float)[2]) /
                                     np.asarray(jt[r['sha']]['classes']['chain']['1']['counts'], float).sum()
                                     for r in rows]),
            'all1_merge_lift': np.array([jt[r['sha']]['classes']['merge']['1']['lift']['all_u'] for r in rows])}
    A = C.oof(X, y, fold)
    base = C.metrics(y, A)
    assert abs(base['mae'] - 0.002971939445864855) < 2e-6, base
    out = {'schema': 'genrec.issue284.model/1', 'n_rows': len(rows), 'n_strong': int(strong.sum()),
           'n_high': int((y > 0.06).sum()), 'n_high_strong': int(((y > 0.06) & strong).sum()),
           'n_above_0.065': int((y > 0.065).sum()),
           'baseline_A': {'oof': base, 'confusion': C.confusion(y, A),
                          'strict': C.metrics(y[strong], A[strong])}}
    preds = {
        'B_all1_chain': C.oof(np.column_stack([X, feat['all1_chain']]), y, fold),
        'C_all_norm': C.oof(np.column_stack([X, feat['all_norm']]), y, fold),
        'diag_cond_refine_only': C.oof(np.column_stack([X, feat['cond_refine']]), y, fold),
        'D_all1_plus_refine': C.oof(np.column_stack([X, feat['all1_chain'], feat['cond_refine']]), y, fold)}
    arms = {}
    for name, pred in preds.items():
        arms[name] = {'oof': C.metrics(y, pred), 'confusion': C.confusion(y, pred),
                      'strict': C.metrics(y[strong], pred[strong]),
                      'boot': C.cluster_bootstrap(A, pred, y, unit)}
    out['fixed_arms'] = arms
    assert abs(arms['B_all1_chain']['oof']['mae'] - 0.0026990060141431345) < 1e-9, arms['B_all1_chain']
    assert abs(arms['B_all1_chain']['oof']['spearman'] - 0.7665458672800122) < 1e-6
    # #283 selection reproduction (its five candidates, its procedure)
    cand283 = ['all1_chain', 'all1_chain_lift', 'all3_chain_lift', 'am_mb_chain', 'all1_merge_lift']
    picks283, pred283 = {}, np.zeros(len(rows))
    for g in sorted(set(fold.tolist())):
        tr = fold != g
        base_inner = float(np.mean(np.abs(y[tr] - C.oof(X[tr], y[tr], fold[tr]))))
        scores = {}
        for cand in cand283:
            Xc = np.column_stack([X, feat[cand]])
            scores[cand] = float(np.mean(np.abs(y[tr] - C.oof(Xc[tr], y[tr], fold[tr]))))
        best = min(scores, key=scores.get)
        choice = best if scores[best] < base_inner else None
        picks283[str(g)] = choice
        XX = np.column_stack([X, feat[choice]]) if choice else X
        a = C.inner_alpha(XX[tr], y[tr], fold[tr])
        sc = StandardScaler().fit(XX[tr])
        rg = Ridge(alpha=a).fit(sc.transform(XX[tr]), y[tr])
        pred283[~tr] = rg.predict(sc.transform(XX[~tr]))
    out['p283_reproduction'] = {'picks': picks283, 'oof': C.metrics(y, pred283),
                                'expected': {'mae': 0.0028218078145555523, 'spearman': 0.7156116398497099}}
    # new selection among B/C/D
    newp, picks = np.zeros(len(rows)), {}
    for g in sorted(set(fold.tolist())):
        tr = fold != g
        base_inner = float(np.mean(np.abs(y[tr] - C.oof(X[tr], y[tr], fold[tr]))))
        cands = {'B_all1_chain': np.column_stack([X, feat['all1_chain']]),
                 'C_all_norm': np.column_stack([X, feat['all_norm']]),
                 'D_all1_plus_refine': np.column_stack([X, feat['all1_chain'], feat['cond_refine']])}
        scores = {name: float(np.mean(np.abs(y[tr] - C.oof(XX[tr], y[tr], fold[tr]))))
                  for name, XX in cands.items()}
        best = min(scores, key=scores.get)
        choice = best if scores[best] < base_inner else None
        picks[str(g)] = {'choice': choice, 'scores': {k: round(v, 7) for k, v in scores.items()},
                         'inner_base': round(base_inner, 7)}
        XX = cands[choice] if choice else X
        a = C.inner_alpha(XX[tr], y[tr], fold[tr])
        sc = StandardScaler().fit(XX[tr])
        rg = Ridge(alpha=a).fit(sc.transform(XX[tr]), y[tr])
        newp[~tr] = rg.predict(sc.transform(XX[~tr]))
    sel = {'oof': C.metrics(y, newp), 'confusion': C.confusion(y, newp),
           'strict': C.metrics(y[strong], newp[strong]), 'picks': picks,
           'boot': C.cluster_bootstrap(A, newp, y, unit)}
    out['selected'] = sel
    # BH over the three family tests (A vs B/C/D)
    out['family_tests'] = {k: arms[k]['boot'] for k in ('B_all1_chain', 'C_all_norm', 'D_all1_plus_refine')}
    # power
    def power(pred):
        d = np.abs(y - A) - np.abs(y - pred)
        mean, sd = float(d.mean()), float(d.std(ddof=1))
        req = ((1.645 + 0.8416) * sd / mean) ** 2 if mean > 0 else None
        mde = (1.645 + 0.8416) * sd / np.sqrt(len(d))
        cl = {}
        arr = np.array(unit)
        for u in sorted(set(unit)):
            cl[u] = float(d[arr == u].mean())
        clv = np.array(list(cl.values()))
        cmean, csd = float(clv.mean()), float(clv.std(ddof=1))
        creq = ((1.645 + 0.8416) * csd / cmean) ** 2 if cmean > 0 else None
        return {'mean_diff': round(mean, 8), 'sd': round(sd, 8),
                'n_required_80pct': None if req is None else int(np.ceil(req)),
                'min_detectable_mean_at_129': round(mde, 8),
                'cluster_mean': round(cmean, 8), 'cluster_sd': round(csd, 8),
                'cluster_n_required': None if creq is None else int(np.ceil(creq)),
                'cluster_mde_at_127': round((1.645 + 0.8416) * csd / np.sqrt(len(clv)), 8)}
    out['power'] = {'A_vs_B': power(preds['B_all1_chain']), 'A_vs_C': power(preds['C_all_norm']),
                    'A_vs_D': power(preds['D_all1_plus_refine']), 'A_vs_selected': power(newp)}
    out['passes'] = {name: bool(arms[name]['boot']['ci95'][0] > 0) for name in
                     ('B_all1_chain', 'C_all_norm', 'D_all1_plus_refine')}
    out['passes']['selected'] = bool(sel['boot']['ci95'][0] > 0)
    (P / 'MODEL_284.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({'baseline': base, 'arms': {k: v['oof'] for k, v in arms.items()},
                      'p283': out['p283_reproduction'], 'selected': sel['oof'], 'picks': picks,
                      'passes': out['passes'], 'power': out['power']}, indent=1)[:3500])


if __name__ == '__main__':
    main()
