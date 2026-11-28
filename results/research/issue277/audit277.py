"""Issue #277 — independent audit of the #276 six-feature Ridge + frozen predictor + prospective rows.

All inputs are pinned to the #276 commit f62fd6ca. This script:
  A. recomputes every OOF metric and the >0.06 confusion table from the stored predictions
  B. tiers the identity evidence (strong = ledger/record sha; weak = collision-equality only) and
     evaluates the strictly bound subset
  C. reruns the inner alpha selection with the scaler fitted inside each inner training fold
     (the #276 inner_ridge_alpha standardised before splitting) and reports both OOF tables
  D. fits the final Ridge6 once on the frozen 129-SID development queue and exports PREDICTOR.json
  E. writes timestamped predictions for every recoverable Stage2 SID without a public Stage3 label
Writes AUDIT_277.json, PREDICTOR.json, PROSPECTIVE.json. No training, no Stage3.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

REPO = Path('.').resolve()
P276 = REPO / 'results/research/issue276'
P = REPO / 'results/research/issue277'
FEATS = ['l1_pos_hit', 'l1_neg_hit', 'l1_hard_er', 'l2_distinct', 'l3_collision', 'l3_refine_bits']
ALPHAS = (0.01, 0.1, 1.0, 10.0)
ISSUE_REF_COMMIT = 'f62fd6caa6f6d1e7690e591bfd06a9308f3ebaf9'


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = {'mae': float(np.mean(np.abs(y - p))), 'rmse': float(np.sqrt(np.mean((y - p) ** 2)))}
    m['spearman'] = float(stats.spearmanr(y, p).correlation) if len(y) > 2 and np.std(p) > 0 else None
    m['kendall'] = float(stats.kendalltau(y, p).correlation) if len(y) > 2 and np.std(p) > 0 else None
    rev = tot = 0
    for i in range(len(y)):
        for k in range(i + 1, len(y)):
            if (y[i] - y[k]) * (p[i] - p[k]) < 0:
                rev += 1
            if y[i] != y[k]:
                tot += 1
    m['reversals'] = rev / tot if tot else None
    return m


def confusion(y, p, cut=0.06):
    y, p = np.array(y, float), np.array(p, float)
    pos = int((y > cut).sum())
    tp = int(((p > cut) & (y > cut)).sum())
    fp = int(((p > cut) & (y <= cut)).sum())
    fn = int(((p <= cut) & (y > cut)).sum())
    tn = int(((p <= cut) & (y <= cut)).sum())
    return {'positives': pos, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None}


def inner_alpha_leaky(X, y, groups):
    """#276 procedure: scaler fitted on the whole outer-training block before the inner split."""
    sc = StandardScaler().fit(X)
    Z = sc.transform(X)
    best, best_err = 1.0, np.inf
    for alpha in ALPHAS:
        errs = []
        for g in sorted(set(groups)):
            mask = groups != g
            if mask.sum() < 5:
                continue
            rg = Ridge(alpha=alpha).fit(Z[mask], y[mask])
            errs.append(np.mean(np.abs(rg.predict(Z[~mask]) - y[~mask])))
        if errs and np.mean(errs) < best_err:
            best, best_err = alpha, float(np.mean(errs))
    return best


def inner_alpha_fixed(X, y, groups):
    """Corrected: standardiser fitted inside every inner training fold."""
    best, best_err = 1.0, np.inf
    for alpha in ALPHAS:
        errs = []
        for g in sorted(set(groups)):
            mask = groups != g
            if mask.sum() < 5:
                continue
            sc = StandardScaler().fit(X[mask])
            rg = Ridge(alpha=alpha).fit(sc.transform(X[mask]), y[mask])
            errs.append(np.mean(np.abs(rg.predict(sc.transform(X[~mask])) - y[~mask])))
        if errs and np.mean(errs) < best_err:
            best, best_err = alpha, float(np.mean(errs))
    return best


def main() -> None:
    P.mkdir(parents=True, exist_ok=True)
    table = json.load(open(P276 / 'SID_TABLE.json'))
    folds = json.load(open(P276 / 'FOLDS.json'))
    analysis = json.load(open(P276 / 'ANALYSIS_276.json'))
    rows_by_group = {g['group']: g['rows'] for g in table['groups']}

    # rows used by #276 (must match build_rows: complete 8-feature set incl. l2_pos/l2_neg)
    FEATS8 = ['l1_pos_hit', 'l1_neg_hit', 'l1_hard_er', 'l2_pos_hit', 'l2_neg_hit',
              'l2_distinct', 'l3_collision', 'l3_refine_bits']
    used = []
    for g in table['groups']:
        cand = [r for r in g['rows']
                if all(isinstance((r.get('features') or {}).get(k), (int, float)) for k in FEATS8)]
        labels = [r['label_recall@10'] for r in g['rows'] if isinstance(r['label_recall@10'], (int, float))]
        if not cand or not labels:
            continue
        r = cand[0]
        ident = sorted({a.get('identity') for a in r.get('arms', []) if a.get('identity')})
        srcs = [s.get('source') for s in r.get('label_sources', [])]
        strong = ('L1_ledger' in ident) or any(str(s).startswith('inventory_unit') for s in srcs)
        used.append({'group': g['group'], 'label': float(np.mean(labels)),
                     'features6': {k: float(r['features'][k]) for k in FEATS},
                     'features8': {k: float(r['features'][k]) for k in FEATS8},
                     'identities': ident, 'strong': strong,
                     'arms': [a.get('arm') for a in r.get('arms', [])],
                     'sid_sha256': r['sid_sha256']})
    assert len(used) == analysis['n_groups_with_features_and_label'], (
        len(used), analysis['n_groups_with_features_and_label'])

    # ---------- A. metric recomputation from stored predictions ----------
    audit = {'schema': 'genrec.issue277.audit/1', 'issue_ref_commit': ISSUE_REF_COMMIT,
             'pinned_inputs': {k: {'path': str(P276 / f), 'sha256': sha_file(P276 / f)}
                               for k, f in [('SID_TABLE', 'SID_TABLE.json'),
                                            ('FOLDS', 'FOLDS.json'),
                                            ('ANALYSIS_276', 'ANALYSIS_276.json')]},
             'n_rows_used': len(used)}
    recomputed = {}
    for q in ('Q_SID', 'Q_VEC'):
        y = np.array(analysis[q]['y'])
        recomputed[q] = {'n': len(y), 'metrics': {}, 'confusion_0.06': {}}
        for model, pred in analysis[q]['predictions'].items():
            m = metrics(y, pred)
            stored = analysis[q]['metrics'][model]
            ok = all((m[k] is None and stored[k] is None) or
                     (m[k] is not None and stored[k] is not None and abs(m[k] - stored[k]) < 1e-12)
                     for k in ('mae', 'rmse', 'spearman', 'kendall', 'reversals'))
            recomputed[q]['metrics'][model] = {'recomputed': m, 'matches_stored': ok}
            recomputed[q]['confusion_0.06'][model] = confusion(y, pred)
    audit['recomputed'] = recomputed
    audit['stored_threshold_pysr_only'] = {q: analysis[q]['threshold_0.06'] for q in ('Q_SID', 'Q_VEC')}

    # ---------- B. identity tiers + strict subset ----------
    weak = [u for u in used if not u['strong']]
    strict = [u for u in used if u['strong']]
    audit['identity'] = {
        'n_strong': len(strict), 'n_weak': len(weak),
        'weak_rows': [{'group': u['group'], 'arms': u['arms'][:2], 'label': u['label'],
                       'identities': u['identities'], 'sid_sha256': u['sid_sha256']} for u in weak],
        'strong_high_recall': sum(1 for u in strict if u['label'] > 0.06),
        'weak_high_recall': sum(1 for u in weak if u['label'] > 0.06),
    }
    # strict-subset evaluation of the stored OOF predictions (sensitivity, not a refit)
    order_sid = analysis['Q_SID']['order']
    pred_sid = np.array(analysis['Q_SID']['predictions']['ridge6_275'])
    y_sid = np.array(analysis['Q_SID']['y'])
    mask = np.array([u['strong'] for u in
                     [next(x for x in used if x['group'] == g) for g in order_sid]])
    audit['identity']['strict_subset_oof_ridge6'] = {
        'n': int(mask.sum()),
        'metrics': metrics(y_sid[mask], pred_sid[mask]),
        'confusion_0.06': confusion(y_sid[mask], pred_sid[mask]),
    }
    audit['identity']['weak_subset_oof_ridge6'] = {
        'n': int((~mask).sum()),
        'metrics': metrics(y_sid[~mask], pred_sid[~mask]),
        'confusion_0.06': confusion(y_sid[~mask], pred_sid[~mask]),
    }

    # ---------- C. preprocessing-leak comparison (same folds, features, alpha grid) ----------
    grp = [u['group'] for u in used]
    X6 = np.array([[u['features6'][k] for k in FEATS] for u in used])
    y6 = np.array([u['label'] for u in used])
    fold_of = np.array([folds['assignment'][g] for g in grp])
    comp = {'leaky': {'pred': np.zeros(len(used)), 'alphas': []},
            'fixed': {'pred': np.zeros(len(used)), 'alphas': []}}
    for mode, picker in (('leaky', inner_alpha_leaky), ('fixed', inner_alpha_fixed)):
        for k in range(5):
            tr = fold_of != k
            te = fold_of == k
            alpha = picker(X6[tr], y6[tr], fold_of[tr])
            sc = StandardScaler().fit(X6[tr])
            rg = Ridge(alpha=alpha).fit(sc.transform(X6[tr]), y6[tr])
            comp[mode]['pred'][te] = rg.predict(sc.transform(X6[te]))
            comp[mode]['alphas'].append(alpha)
    audit['preprocessing'] = {m: {'alphas': c['alphas'], 'metrics': metrics(y6, c['pred']),
                                  'confusion_0.06': confusion(y6, c['pred'])}
                              for m, c in comp.items()}
    audit['preprocessing']['max_abs_pred_diff'] = float(np.max(np.abs(comp['leaky']['pred'] - comp['fixed']['pred'])))
    audit['preprocessing']['note'] = ('scaler was fitted on the whole outer-training block before the inner '
                                      'split in #276; corrected variant fits it inside each inner training fold')
    # compare with #276's stored ridge6 (which used the leaky inner selection)
    audit['preprocessing']['stored_ridge6_metrics'] = analysis['Q_SID']['metrics']['ridge6_275']

    # ---------- D. final frozen Ridge6 fit ----------
    alpha_final = inner_alpha_fixed(X6, y6, fold_of)
    sc = StandardScaler().fit(X6)
    Z = sc.transform(X6)
    rg = Ridge(alpha=alpha_final).fit(Z, y6)
    coef_std = rg.coef_.tolist()
    coef_raw = (rg.coef_ / sc.scale_).tolist()
    intercept_raw = float(rg.intercept_ - sum(rg.coef_ * sc.mean_ / sc.scale_))
    pred = rg.predict(Z)
    predictor = {
        'schema': 'genrec.issue277.predictor/1',
        'commit_reference': ISSUE_REF_COMMIT,
        'target': 'Stage3 test recall@10 (protocol-majority per SID)',
        'features': FEATS,
        'feature_definitions': 'results/research/issue277/FEATURE_CONTRACT.md',
        'fit_procedure': {'standardise': 'StandardScaler on the frozen 129-SID development queue',
                          'alpha_grid': list(ALPHAS),
                          'alpha_selection': 'inner SID-group CV with the scaler fitted inside each inner training fold',
                          'alpha': alpha_final, 'ridge_intercept_standardised': float(rg.intercept_)},
        'scaler': {'mean': sc.mean_.tolist(), 'scale': sc.scale_.tolist()},
        'coef_standardised': coef_std,
        'coef_raw_units': coef_raw,
        'intercept_raw_units': intercept_raw,
        'formula': 'r_hat = intercept_raw + sum_i coef_raw_units[i] * x_i '
                   '(equivalently intercept_std + sum_i coef_standardised[i] * (x_i - mean_i) / scale_i)',
        'training': {'n_sids': len(used), 'label_sha256': hashlib.sha256(
                         json.dumps([[u['group'], u['label']] for u in used], sort_keys=True).encode()).hexdigest(),
                     'features_sha256': hashlib.sha256(
                         json.dumps([[u['group'], u['features6']] for u in used], sort_keys=True).encode()).hexdigest(),
                     'table_sha256': audit['pinned_inputs']['SID_TABLE']['sha256'],
                     'folds_sha256': audit['pinned_inputs']['FOLDS']['sha256']},
        'fit_quality_full_data': {'n': len(used), 'mae': float(np.mean(np.abs(y6 - pred))),
                                  'spearman': float(stats.spearmanr(y6, pred).correlation)},
        'oof_reference': {'note': 'OOF metrics are the honest generalisation numbers; the full-data fit '
                                  'above is in-sample and NOT a performance estimate',
                          'oof_ridge6_corrected': audit['preprocessing']['fixed']['metrics']},
        'uncertainty': None,
        'domain': {k: {'min': float(X6[:, i].min()), 'max': float(X6[:, i].max())}
                   for i, k in enumerate(FEATS)},
        'created_utc': datetime.now(timezone.utc).isoformat(),
    }
    residual = y6 - np.array(comp['fixed']['pred'])
    predictor['uncertainty'] = {
        'oof_residual_quantiles': {'q05': float(np.quantile(residual, 0.05)),
                                   'q50': float(np.quantile(residual, 0.50)),
                                   'q95': float(np.quantile(residual, 0.95)),
                                   'abs_q90': float(np.quantile(np.abs(residual), 0.90))},
        'note': 'empirical OOF residual interval; finite-sample, no coverage guarantee',
    }
    (P / 'PREDICTOR.json').write_text(json.dumps(predictor, indent=1) + '\n')

    # ---------- E. prospective predictions for unlabeled rows ----------
    unlabeled = []
    for g in table['groups']:
        for r in g['rows']:
            f = r.get('features') or {}
            if r['label_recall@10'] is not None:
                continue
            if not all(isinstance(f.get(k), (int, float)) for k in FEATS):
                continue
            x = np.array([[float(f[k]) for k in FEATS]])
            z = (x - sc.mean_) / sc.scale_
            pred_value = float(rg.predict(z)[0])
            outside = [k for i, k in enumerate(FEATS)
                       if not (X6[:, i].min() <= x[0, i] <= X6[:, i].max())]
            unlabeled.append({'sid_sha256': r['sid_sha256'], 'mapping_sha256': r.get('mapping_sha256'),
                              'features_source': f.get('_source'), 'predicted_recall@10': pred_value,
                              'outside_training_range': outside,
                              'arms': [a.get('arm') for a in r.get('arms', [])][:3]})
    prospective = {'schema': 'genrec.issue277.prospective/1',
                   'created_utc': datetime.now(timezone.utc).isoformat(),
                   'predictor_sha256': sha_file(P / 'PREDICTOR.json'),
                   'note': 'predictions computed before any Stage3 label of these SIDs is read',
                   'rows': unlabeled}
    (P / 'PROSPECTIVE.json').write_text(json.dumps(prospective, indent=1) + '\n')

    (P / 'AUDIT_277.json').write_text(json.dumps(audit, indent=1) + '\n')
    print(json.dumps({
        'n_used': len(used), 'n_strong': len(strict), 'n_weak': len(weak),
        'strong_high': audit['identity']['strong_high_recall'],
        'weak_high': audit['identity']['weak_high_recall'],
        'strict_subset': audit['identity']['strict_subset_oof_ridge6'],
        'weak_subset': audit['identity']['weak_subset_oof_ridge6'],
        'preproc': {'leaky': audit['preprocessing']['leaky']['metrics'],
                    'fixed': audit['preprocessing']['fixed']['metrics'],
                    'alphas_leaky': audit['preprocessing']['leaky']['alphas'],
                    'alphas_fixed': audit['preprocessing']['fixed']['alphas'],
                    'max_diff': audit['preprocessing']['max_abs_pred_diff']},
        'ridge_recomputed_confusion': {q: recomputed[q]['confusion_0.06']['ridge6_275'] for q in ('Q_SID', 'Q_VEC')},
        'pysr_confusion': {q: recomputed[q]['confusion_0.06']['pysr'] for q in ('Q_SID', 'Q_VEC')},
        'alpha_final': alpha_final,
        'coef_std': [round(c, 5) for c in coef_std],
        'coef_raw': [round(c, 6) for c in coef_raw],
        'intercept_raw': intercept_raw,
        'n_unlabeled_predictions': len(unlabeled),
    }, indent=1))


if __name__ == '__main__':
    main()
