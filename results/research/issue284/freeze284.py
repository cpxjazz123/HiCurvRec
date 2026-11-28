"""Issue #284: freeze the research candidate B and its exploratory prospective predictions.

The #277 predictor and the 124-row prospective file are NOT modified. The output is explicitly a
research candidate; replacement of #277 requires the historical CI gate AND a future independent
label validation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '.')
sys.path.insert(0, 'results/research/issue284')
import common284 as C
from results.research.issue283 import joint283 as J

P = Path('results/research/issue284')
P283 = Path('results/research/issue283')


def sha(p):
    return J.sha256(p)


def main() -> None:
    rows = C.table_rows()
    folds = C.table_rows.folds
    ch = json.load(open(P / 'CHAIN_TABLE.json'))['entries']
    y = np.array([r['y'] for r in rows])
    fold = np.array([folds['assignment'][r['group']] for r in rows])
    X7 = np.array([[r['x6'][k] for k in C.FEATS6] + [ch[r['sha']]['all1_chain']] for r in rows])
    alpha = C.inner_alpha(X7, y, fold)
    sc = StandardScaler().fit(X7)
    rg = Ridge(alpha=alpha).fit(sc.transform(X7), y)
    coef = rg.coef_ / sc.scale_
    intercept = float(rg.intercept_ - (coef * sc.mean_).sum())
    model = {'candidate': 'B_all1_chain', 'features': C.FEATS6 + ['all1_chain'],
             'alpha': float(alpha), 'coef_raw_units': [float(x) for x in coef],
             'intercept_raw_units': intercept,
             'scaler_mean': [float(x) for x in sc.mean_], 'scaler_scale': [float(x) for x in sc.scale_],
             'train_rows': len(rows), 'training_domain': '129 labelled SIDs (120 strong / 9 weak), '
             '5 frozen folds, chain class support 22k rows per SID',
             'input_hashes': {'CHAIN_TABLE.json': sha(P / 'CHAIN_TABLE.json'),
                              'SID_TABLE.json': sha('results/research/issue276/SID_TABLE.json'),
                              'FOLDS.json': sha('results/research/issue276/FOLDS.json'),
                              'JOIN_TABLE.json': sha(P283 / 'JOIN_TABLE.json')},
             'status': 'research_candidate_not_replacement'}

    def predict(f7):
        return float(intercept + np.dot(coef, f7))

    # pending = 124 prospective minus the two new labels
    pros = json.load(open('results/research/issue277/PROSPECTIVE.json'))
    new_labels = {'9f6bf673', '61f898498c19'}
    pending, known = [], []
    for r in pros['rows']:
        s = r['sid_sha256']
        rec = {'sid_sha256': s, 'pred277': r.get('predicted_recall@10')}
        c = ch.get(s)
        if c is None:
            rec['pred284'] = None; rec['reason'] = 'chain_features_missing'
        else:
            f7 = [r0['x6'][k] for r0 in rows if r0['sha'] == s][0] if any(r0['sha'] == s for r0 in rows) else None
            # pending SIDs are unlabelled -> no x6 in the labelled rows; use their #276 features file
            if f7 is None:
                rec['pred284'] = None; rec['reason'] = 'six_features_not_in_labelled_rows'
            else:
                rec['pred284'] = predict(f7 + [c['all1_chain']])
        if s[:12] in new_labels or s[:12] in ('9f6bf6731e88', '61f898498c19'):
            known.append(rec)
        else:
            pending.append(rec)
    # new-label sensitivity: need the six features + chain features; the table rows for these SIDs
    # exist (unlabelled) and their features are in SID_TABLE -> rebuild rows including unlabelled SIDs
    table = json.load(open('results/research/issue276/SID_TABLE.json'))
    feas = {}
    for g in table['groups']:
        for r in g['rows']:
            if all(isinstance((r.get('features') or {}).get(k), (int, float)) for k in C.FEATS8):
                feas[r['sid_sha256']] = r['features']
    for rec in pending + known:
        s = rec['sid_sha256']
        if rec.get('pred284') is None and s in feas and s in ch:
            f7 = [float(feas[s][k]) for k in C.FEATS6] + [ch[s]['all1_chain']]
            rec['pred284'] = predict(f7); rec['reason'] = 'ok'
    sens = []
    for rec in known:
        s = rec['sid_sha256']
        if s in feas and s in ch:
            rec['pred284'] = predict([float(feas[s][k]) for k in C.FEATS6] + [ch[s]['all1_chain']])
        sens.append(rec)
    # recovered unlabelled SIDs outside the table: #274 trainer exports + #272 run exports
    audit = json.load(open(P / 'AUDIT_284.json'))
    rec_extra, seen = [], set()
    for e in audit['p274_exports']:
        for name, f in e['files'].items():
            s = f['sha256']
            if not s or any(r['sid_sha256'] == s for r in table['groups'][0]['rows']):
                continue
            if s in seen:
                continue
            seen.add(s)
            rec_extra.append({'sid_sha256': s, 'source': f"274 {e['label']} {name}", 'path': f['path']})
    for e in audit['p272_runs']:
        for f in e['sids']:
            if f['in_table'] or f['sha256'] in seen:
                continue
            seen.add(f['sha256'])
            rec_extra.append({'sid_sha256': f['sha256'], 'source': f"272 {e['run_id']}", 'path': f['path']})
    for rec in rec_extra:
        p = rec['path']
        if not Path(p).is_file():
            rec['pred284'] = None; rec['reason'] = 'file_missing'; continue
        codes = J.codes_matrix(p)
        cf = C.chain_features(codes)
        rec['all1_chain'] = cf['all1_chain']
        rec['pred284'] = None
        rec['reason'] = 'six_features_not_available_for_this_export'
    # the two labelled-in-the-meantime SIDs' actual recalls for the time-postordered sensitivity
    actual = {'9f6bf6731e88': 0.0479465171747419, '61f898498c19': 0.04846881039015303}
    for rec in sens:
        rec['actual_recall@10'] = next((v for k, v in actual.items() if rec['sid_sha256'].startswith(k)), None)
        rec['note'] = 'time-postordered, NOT a blind test; not used to modify the formula'
    out = {'schema': 'genrec.issue284.prospective/1',
           'created_utc': '2026-11-28T06:00:00+00:00',
           'candidate': model, 'predictions_pending': pending,
           'new_label_sensitivity': sens, 'recovered_unlabelled': rec_extra,
           'not_replacement': 'the #277 predictor and its 124-row prospective file are unchanged; '
                              'this file is an exploratory research candidate for pending SIDs'}
    (P / 'PROSPECTIVE_284.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({'alpha': alpha, 'coef': model['coef_raw_units'],
                      'n_pending': len(pending), 'n_sensitivity': len(sens),
                      'n_recovered': len(rec_extra),
                      'sensitivity': [(s['sid_sha256'][:12], s.get('pred277'), s.get('pred284'),
                                       s.get('actual_recall@10')) for s in sens]}, indent=1))


if __name__ == '__main__':
    main()
