"""Issue #286: freeze the exploratory gated candidate E and its predictions for truly unlabeled SIDs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'results/research/issue285')
sys.path.insert(0, 'results/research/issue286')
import common285 as W
import common284 as C
import gate286 as G
from results.research.issue283 import joint283 as J

OUT = Path('results/research/issue286')


def sha(p):
    return J.sha256(p)


def main() -> None:
    d = W.load_table()
    G.X6, G.y, G.fold, G.N, G.Cv, G.Pv, G.ERv = d['X6'], d['y'], d['fold'], d['N'], d['C'], d['P'], d['ER']
    y, fold = d['y'], d['fold']
    # full-cohort lambda via the same inner scheme
    scores = {lam: [] for lam in G.LAMBDAS}
    for h in sorted(set(fold.tolist())):
        tr = np.where(fold != h)[0]; te = np.where(fold == h)[0]
        bh = G.fit_bundle(tr)
        for lam in G.LAMBDAS:
            scores[lam].append(float(np.mean(np.abs(y[te] - G.predict_bundle(bh, te, lam)))))
    lam = min(G.LAMBDAS, key=lambda l: float(np.mean(scores[l])))
    bundle = G.fit_bundle(np.arange(len(y)))
    model = {
        'candidate': 'E_main_soft_gated_residual_shrink',
        'formula': 'Rhat = R6(x) + lambda*[g*e_good(P,ER) + (1-g)*e_other(x6)]',
        'lambda': float(lam), 'lambda_inner_means': {str(k): round(float(np.mean(v)), 8) for k, v in scores.items()},
        'gate': {'form': 'g = sigmoid(b0 - b1*N_std - b2*C_std), b1,b2>=0',
                 'b': [round(float(v), 6) for v in bundle['b']],
                 'target': 'feature region lowN & lowC (training-fold tertiles), not a label',
                 'edges_q1_3': {'N': bundle['qn'], 'C': bundle['qc']},
                 'std': {'N_mean': bundle['n_mu'], 'N_sd': bundle['n_sd'],
                         'C_mean': bundle['c_mu'], 'C_sd': bundle['c_sd']}},
        'r6': {'alpha': float(bundle['r6']['alpha']),
               'scaler_mean': [float(x) for x in bundle['r6']['sc'].mean_],
               'scaler_scale': [float(x) for x in bundle['r6']['sc'].scale_],
               'coef': [float(x) for x in bundle['r6']['rg'].coef_],
               'intercept': float(bundle['r6']['rg'].intercept_)},
        'experts': {'good': {'features': ['l1_pos_hit', 'l1_hard_er'],
                             'scaler_mean': [float(x) for x in bundle['e_good']['sc'].mean_],
                             'scaler_scale': [float(x) for x in bundle['e_good']['sc'].scale_],
                             'coef': [float(x) for x in bundle['e_good']['rg'].coef_],
                             'intercept': float(bundle['e_good']['rg'].intercept_)},
                    'other': {'features': C.FEATS6,
                              'scaler_mean': [float(x) for x in bundle['e_other']['sc'].mean_],
                              'scaler_scale': [float(x) for x in bundle['e_other']['sc'].scale_],
                              'coef': [float(x) for x in bundle['e_other']['rg'].coef_],
                              'intercept': float(bundle['e_other']['rg'].intercept_)}},
        'train_rows': len(y), 'domain_feature_ranges': {
            k: {'min': float(np.min(d['X6'][:, i])), 'max': float(np.max(d['X6'][:, i]))}
            for i, k in enumerate(C.FEATS6)},
        'input_hashes': {'SID_TABLE.json': sha('results/research/issue276/SID_TABLE.json'),
                         'FOLDS.json': sha('results/research/issue276/FOLDS.json'),
                         'MODEL_286.json': sha(OUT / 'MODEL_286.json')},
        'status': 'research_candidate_exploratory_does_not_pass_gate',
        'note': 'the main candidate failed the paired CI; this file is exploratory and does not replace #277'}
    # predictions for the 122 pending SIDs (features from the #276 rows; all six required)
    table = json.load(open('results/research/issue276/SID_TABLE.json'))
    feas = {}
    for g in table['groups']:
        for r in g['rows']:
            if all(isinstance((r.get('features') or {}).get(k), (int, float)) for k in C.FEATS8):
                feas[r['sid_sha256']] = r['features']
    pros = json.load(open('results/research/issue277/PROSPECTIVE.json'))
    p284 = {r['sid_sha256']: r for r in json.load(open('results/research/issue284/PROSPECTIVE_284.json'))['predictions_pending']}
    p285 = {r['sid_sha256']: r for r in json.load(open('results/research/issue285/PROSPECTIVE_285.json'))['predictions_pending']}
    known = {'9f6bf673', '61f898498c19'}
    pending, sens = [], []
    for r in pros['rows']:
        s = r['sid_sha256']
        rec = {'sid_sha256': s, 'pred277': r.get('predicted_recall@10'),
               'pred284': p284.get(s, {}).get('pred284'), 'pred285': p285.get(s, {}).get('pred285'),
               'pred286': None}
        f = feas.get(s)
        if f is not None:
            x6 = [float(f[k]) for k in C.FEATS6]
            gval = 1.0 / (1.0 + np.exp(-(bundle['b'][0] - bundle['b'][1] * (x6[1] - bundle['n_mu']) / (bundle['n_sd'] or 1)
                                         - bundle['b'][2] * (x6[4] - bundle['c_mu']) / (bundle['c_sd'] or 1))))
            r6 = float(bundle['r6']['rg'].predict(bundle['r6']['sc'].transform(np.array([x6])))[0])
            eg = float(bundle['e_good']['rg'].predict(bundle['e_good']['sc'].transform(
                np.array([[x6[0], x6[2]]])))[0])
            eo = float(bundle['e_other']['rg'].predict(bundle['e_other']['sc'].transform(np.array([x6])))[0])
            rec['pred286'] = r6 + lam * (gval * eg + (1 - gval) * eo)
            rec['gate'] = round(float(gval), 5)
        if any(s.startswith(k) for k in known):
            sens.append(rec)
        else:
            pending.append(rec)
    actual = {'9f6bf6731e88': 0.0479465171747419, '61f898498c19': 0.04846881039015303}
    for rec in sens:
        rec['actual_recall@10'] = next((v for k, v in actual.items() if rec['sid_sha256'].startswith(k)), None)
        rec['note'] = 'time-postordered, not blind; diagnostic only'
    out = {'schema': 'genrec.issue286.prospective/1', 'created_utc': '2026-11-28T07:00:00+00:00',
           'candidate': model, 'predictions_pending': pending, 'new_label_sensitivity': sens,
           'not_replacement': '#277/#284/#285 remain unchanged; the main candidate failed its CI gate'}
    (OUT / 'PROSPECTIVE_286.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({'lambda': lam, 'lambda_inner': model['lambda_inner_means'],
                      'b': model['gate']['b'], 'r6_alpha': model['r6']['alpha'],
                      'n_pending': len(pending), 'n_sensitivity': len(sens),
                      'sensitivity': [(r['sid_sha256'][:12], r.get('pred277'), r.get('pred286'), r.get('actual_recall@10')) for r in sens]},
                     indent=1))


if __name__ == '__main__':
    main()
