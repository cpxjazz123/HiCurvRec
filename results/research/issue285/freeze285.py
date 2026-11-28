"""Issue #285: freeze the best pre-registered conditional arm (research candidate) and predictions.

The #277 predictor, the 124-row prospective file and the #284 candidate files are NOT modified.
"""
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
from results.research.issue283 import joint283 as J

P = Path('results/research/issue285')


def sha(p):
    return J.sha256(p)


def main() -> None:
    d = W.load_table()
    y = d['y']; fold = d['fold']; X6 = d['X6']
    q1n = float(np.quantile(d['N'], 1 / 3))  # frozen full-cohort gate edge for the final formula
    G_N = (d['N'] <= q1n).astype(float)
    ALL = d['ALL']
    X = np.column_stack([X6, ALL, ALL * G_N])
    alpha = C.inner_alpha(X, y, fold)
    sc = StandardScaler().fit(X)
    rg = Ridge(alpha=alpha).fit(sc.transform(X), y)
    coef = rg.coef_ / sc.scale_
    intercept = float(rg.intercept_ - (coef * sc.mean_).sum())
    model = {'candidate': 'C_ALL_gate', 'features': C.FEATS6 + ['all1_chain', 'all1_chain*G_N'],
             'gate': {'variable': 'l1_neg_hit', 'rule': 'G_N = 1[N <= q1/3]', 'q1_3': q1n},
             'alpha': float(alpha), 'coef_raw_units': [float(x) for x in coef],
             'intercept_raw_units': intercept,
             'scaler_mean': [float(x) for x in sc.mean_], 'scaler_scale': [float(x) for x in sc.scale_],
             'train_rows': len(y), 'training_domain': '129 labelled SIDs (120 strong / 9 weak), 5 frozen folds',
             'input_hashes': {'CHAIN_TABLE.json': sha('results/research/issue284/CHAIN_TABLE.json'),
                              'SID_TABLE.json': sha('results/research/issue276/SID_TABLE.json'),
                              'FOLDS.json': sha('results/research/issue276/FOLDS.json'),
                              'MODEL_284.json': sha('results/research/issue284/MODEL_284.json')},
             'status': 'research_candidate_exploratory_redundant_with_284',
             'note': 'the gate interaction adds ~4e-6 MAE over the raw ALL term; this arm does not replace #277'}

    def predict(f7, all1):
        g = 1.0 if f7 and f7[1] <= q1n else 0.0  # N is FEATS6 index 1
        return float(intercept + np.dot(coef, list(f7) + [all1, all1 * g]))

    table = json.load(open('results/research/issue276/SID_TABLE.json'))
    feas = {}
    for g in table['groups']:
        for r in g['rows']:
            if all(isinstance((r.get('features') or {}).get(k), (int, float)) for k in C.FEATS8):
                feas[r['sid_sha256']] = r['features']
    pros = json.load(open('results/research/issue277/PROSPECTIVE.json'))
    p284 = json.load(open('results/research/issue284/PROSPECTIVE_284.json'))
    pred284 = {r['sid_sha256']: r for r in p284['predictions_pending']}
    ch = d['chain']
    known = {'9f6bf673', '61f898498c19'}
    pending, sens = [], []
    for r in pros['rows']:
        s = r['sid_sha256']
        rec = {'sid_sha256': s, 'pred277': r.get('predicted_recall@10'), 'pred284': None, 'pred285': None,
               'all1_chain': None, 'N': None}
        if s in feas:
            rec['N'] = float(feas[s]['l1_neg_hit'])
        if s in pred284:
            rec['pred284'] = pred284[s].get('pred284')
        if s in feas and s in ch:
            rec['all1_chain'] = float(ch[s]['all1_chain'])
            rec['pred285'] = predict([float(feas[s][k]) for k in C.FEATS6], ch[s]['all1_chain'])
        if any(s.startswith(k) for k in known):
            sens.append(rec)
        else:
            pending.append(rec)
    actual = {'9f6bf6731e88': 0.0479465171747419, '61f898498c19': 0.04846881039015303}
    for rec in sens:
        rec['actual_recall@10'] = next((v for k, v in actual.items() if rec['sid_sha256'].startswith(k)), None)
        rec['note'] = 'time-postordered, not blind; not used to modify the formula'
    out = {'schema': 'genrec.issue285.prospective/1', 'created_utc': '2026-11-28T06:30:00+00:00',
           'candidate': model, 'predictions_pending': pending, 'new_label_sensitivity': sens,
           'not_replacement': '#277 and the #284 candidate are unchanged; this file is exploratory'}
    (P / 'PROSPECTIVE_285.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({'q1_3_N': q1n, 'alpha': alpha, 'coef': model['coef_raw_units'],
                      'n_pending': len(pending), 'n_sensitivity': len(sens),
                      'sensitivity': [(r['sid_sha256'][:12], r.get('pred277'), r.get('pred284'),
                                       r.get('pred285'), r.get('actual_recall@10')) for r in sens]}, indent=1))


if __name__ == '__main__':
    main()
