"""Issue #285: pre-registered conditional predictors A/B/C/D, selection, CIs, BH, power."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '.')
sys.path.insert(0, 'results/research/issue285')
import common285 as W
import common284 as C

P = Path('results/research/issue285')


def main() -> None:
    d = W.load_table()
    y = d['y']; fold = d['fold']; X6 = d['X6']; strong = d['strong']; unit = d['unit']
    g = W.gates(d)
    G_N, G_C = g['G_N'], g['G_C']; ramp_n, ramp_c = g['ramp_N'], g['ramp_C']
    Pv, Nv, Cv, ALL = d['P'], d['N'], d['C'], d['ALL']
    A = C.oof(X6, y, fold)
    base = C.metrics(y, A)
    assert abs(base['mae'] - 0.002971939445864855) < 2e-6, base
    arms = {
        'A': X6,
        'B_N_gate': np.column_stack([X6, G_N, Pv * G_N]),
        'C_ALL_gate': np.column_stack([X6, ALL, ALL * G_N]),
        'D_smooth_double': np.column_stack([X6, Pv * ramp_n * ramp_c]),
        'B_C_gate_alt': np.column_stack([X6, G_C, Pv * G_C]),
        'ref_284_all1': np.column_stack([X6, ALL]),
    }
    out = {'schema': 'genrec.issue285.model/1', 'n_rows': len(y), 'n_strong': int(strong.sum()),
           'n_high': int((y > 0.06).sum()), 'n_high_strong': int(((y > 0.06) & strong).sum()),
           'n_above_0.065': int((y > 0.065).sum()), 'baseline_A': {'oof': base, 'confusion': C.confusion(y, A),
                                                                  'strict': C.metrics(y[strong], A[strong])},
           'fixed_arms': {}}
    preds = {'A': A}
    for name, XX in arms.items():
        if name == 'A':
            continue
        pred = C.oof(XX, y, fold)
        preds[name] = pred
        out['fixed_arms'][name] = {'oof': C.metrics(y, pred), 'confusion': C.confusion(y, pred),
                                   'strict': C.metrics(y[strong], pred[strong]),
                                   'boot': C.cluster_bootstrap(A, pred, y, unit)}
    assert abs(out['fixed_arms']['ref_284_all1']['oof']['mae'] - 0.0026990060141431345) < 1e-9
    assert abs(out['fixed_arms']['ref_284_all1']['oof']['spearman'] - 0.7665458672800122) < 1e-6
    # inner selection among B/C/D
    cands = {k: arms[k] for k in ('B_N_gate', 'C_ALL_gate', 'D_smooth_double')}
    newp, picks = np.zeros(len(y)), {}
    for f in sorted(set(fold.tolist())):
        tr = fold != f
        base_inner = float(np.mean(np.abs(y[tr] - C.oof(X6[tr], y[tr], fold[tr]))))
        scores = {name: float(np.mean(np.abs(y[tr] - C.oof(XX[tr], y[tr], fold[tr]))))
                  for name, XX in cands.items()}
        best = min(scores, key=scores.get)
        choice = best if scores[best] < base_inner else None
        picks[str(f)] = {'choice': choice, 'inner_base': round(base_inner, 7),
                         'scores': {k: round(v, 7) for k, v in scores.items()}}
        XX = cands[choice] if choice else X6
        a = C.inner_alpha(XX[tr], y[tr], fold[tr])
        sc = StandardScaler().fit(XX[tr])
        rg = Ridge(alpha=a).fit(sc.transform(XX[tr]), y[tr])
        newp[~tr] = rg.predict(sc.transform(XX[~tr]))
    sel = {'oof': C.metrics(y, newp), 'confusion': C.confusion(y, newp),
           'strict': C.metrics(y[strong], newp[strong]), 'picks': picks,
           'boot': C.cluster_bootstrap(A, newp, y, unit)}
    out['selected'] = sel
    out['family_tests'] = {k: out['fixed_arms'][k]['boot'] for k in cands}
    out['passes'] = {k: bool(out['fixed_arms'][k]['boot']['ci95'][0] > 0) for k in out['fixed_arms']}
    out['passes']['selected'] = bool(sel['boot']['ci95'][0] > 0)

    def power(pred):
        dd = np.abs(y - A) - np.abs(y - pred)
        mean, sd = float(dd.mean()), float(dd.std(ddof=1))
        req = ((1.645 + 0.8416) * sd / mean) ** 2 if mean > 0 else None
        mde = (1.645 + 0.8416) * sd / np.sqrt(len(dd))
        return {'mean_diff': round(mean, 8), 'sd': round(sd, 8),
                'n_required_80pct': None if req is None else int(np.ceil(req)),
                'min_detectable_mean_at_129': round(mde, 8)}
    out['power'] = {k: power(preds[k]) for k in ('B_N_gate', 'C_ALL_gate', 'D_smooth_double')}
    out['power']['selected'] = power(newp)
    # conditional strata effective sample and within-LL detectability
    cells = W.cell_labels(g['level_N'], g['level_C'])
    ll = cells == 'LL'
    out['strata'] = {c: int((cells == c).sum()) for c in ('LL', 'LN', 'LC', 'HH', 'other')}
    dd_ll = np.abs(y[ll] - A[ll]) - np.abs(y[ll] - newp[ll])
    out['within_LL'] = {'n': int(ll.sum()), 'mean_diff': round(float(dd_ll.mean()), 8),
                        'sd': round(float(dd_ll.std(ddof=1)), 8),
                        'mde': round(float((1.645 + 0.8416) * dd_ll.std(ddof=1) / np.sqrt(ll.sum())), 8)}
    (P / 'MODEL_285.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({'baseline': base, 'fixed': {k: v['oof'] for k, v in out['fixed_arms'].items()},
                      'selected': sel['oof'], 'picks': picks, 'passes': out['passes'],
                      'boot': {k: [round(x, 7) for x in v['boot']['ci95']] for k, v in out['fixed_arms'].items()},
                      'sel_boot': [round(x, 7) for x in sel['boot']['ci95']],
                      'power': out['power'], 'within_LL': out['within_LL']}, indent=1)[:3500])


if __name__ == '__main__':
    main()
