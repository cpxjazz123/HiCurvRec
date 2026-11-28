"""Issue #278 — pre-registered model comparison (see CONTRACT.md section 4/5).

Candidates A (control), B (weighted replacement), D (A + ER_w), E (A + P_w,N_w), F (A + lift-based ER_w);
controls: fold mean/median, single ER. Same folds, leak-free scaler, inner SID-group alpha CV.
Also: cross-fitted residual analysis (does importance explain the OOF residual of A?), collinearity,
stratified residuals, calibration. Writes MODEL_COMPARISON.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

P = Path('results/research/issue278')
P276 = Path('results/research/issue276')
FEATS = ['l1_pos_hit', 'l1_neg_hit', 'l1_hard_er', 'l2_distinct', 'l3_collision', 'l3_refine_bits']
ALPHAS = (0.01, 0.1, 1.0, 10.0)


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = {'mae': float(np.mean(np.abs(y - p))), 'rmse': float(np.sqrt(np.mean((y - p) ** 2)))}
    m['spearman'] = float(stats.spearmanr(y, p).correlation) if len(y) > 2 and np.std(p) > 0 else None
    m['kendall'] = float(stats.kendalltau(y, p).correlation) if len(y) > 2 and np.std(p) > 0 else None
    return m


def confusion(y, p, cut=0.06):
    y, p = np.array(y, float), np.array(p, float)
    return {'positives': int((y > cut).sum()), 'tp': int(((p > cut) & (y > cut)).sum()),
            'fp': int(((p > cut) & (y <= cut)).sum()), 'fn': int(((p <= cut) & (y > cut)).sum()),
            'tn': int(((p <= cut) & (y <= cut)).sum())}


def inner_alpha(X, y, groups):
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


def oof_predictions(X, y, fold):
    pred = np.zeros(len(y))
    alphas = []
    for k in range(5):
        tr, te = fold != k, fold == k
        alpha = inner_alpha(X[tr], y[tr], fold[tr])
        sc = StandardScaler().fit(X[tr])
        rg = Ridge(alpha=alpha).fit(sc.transform(X[tr]), y[tr])
        pred[te] = rg.predict(sc.transform(X[te]))
        alphas.append(alpha)
    return pred, alphas


def main() -> None:
    table = json.load(open(P276 / 'SID_TABLE.json'))
    folds = json.load(open(P276 / 'FOLDS.json'))
    weighted = {r['sid_sha256']: r for r in json.load(open(P / 'WEIGHTED_TABLE.json'))['rows']}
    used, missing = [], []
    for g in table['groups']:
        cand = [r for r in g['rows']
                if all(isinstance((r.get('features') or {}).get(k), (int, float)) for k in FEATS)]
        labels = [r['label_recall@10'] for r in g['rows'] if isinstance(r['label_recall@10'], (int, float))]
        if not cand or not labels:
            continue
        r = cand[0]
        sha = r['sid_sha256']
        w = weighted.get(sha)
        if w is None:
            missing.append(sha)
            continue
        ident = sorted({a.get('identity') for a in r.get('arms', []) if a.get('identity')})
        srcs = [s.get('source') for s in r.get('label_sources', [])]
        strong = ('L1_ledger' in ident) or any(str(s).startswith('inventory_unit') for s in srcs)
        used.append({
            'group': g['group'], 'label': float(np.mean(labels)),
            'sand': {k: float(r['features'][k]) for k in FEATS},
            'live': w['sid_sha256'], 'strong': strong,
            'w': {'l1_er_w': w['L1']['all']['w_freq']['ER_w'], 'l1_p_w': w['L1']['all']['w_freq']['P_w'],
                  'l1_n_w': w['L1']['all']['w_freq']['N_w'], 'l1_er_uniform': w['L1']['all']['uniform']['ER_w'],
                  'l1_er_sqrt': w['L1']['all']['w_sqrt']['ER_w'], 'l1_er_lift': w['L1']['all']['w_lift']['ER_w'],
                  'l2_er_w': w['L2']['all']['w_freq']['ER_w'], 'l2_er_uniform': w['L2']['all']['uniform']['ER_w']},
        })
    y = np.array([u['label'] for u in used])
    groups = np.array([u['group'] for u in used])
    fold = np.array([folds['assignment'][g] for g in groups])
    strict = np.array([u['strong'] for u in used])
    Xa = np.array([[u['sand'][k] for k in FEATS] for u in used])

    def col(name):
        return np.array([u['w'][name] if u['w'][name] is not None else np.nan for u in used])

    er_w, p_w, n_w = col('l1_er_w'), col('l1_p_w'), col('l1_n_w')
    er_u, er_s, er_l, l2_er_w = col('l1_er_uniform'), col('l1_er_sqrt'), col('l1_er_lift'), col('l2_er_w')
    impute = np.nanmedian  # NA handled by masking the affected candidates, reported explicitly
    na_counts = {k: int(np.isnan(v).sum()) for k, v in
                 [('er_w', er_w), ('p_w', p_w), ('n_w', n_w), ('er_uniform', er_u), ('er_lift', er_l),
                  ('l2_er_w', l2_er_w)]}

    def clean(*cols):
        ok = np.ones(len(y), bool)
        for c in cols:
            ok &= ~np.isnan(c)
        if not ok.all():
            # only keep rows with finite values; identical row set for every candidate is required
            pass
        return ok

    cand = {
        'A_control': (Xa, ['6 sand']),
        'B_replacement': (np.column_stack([p_w, n_w, er_w, Xa[:, 3], Xa[:, 4], Xa[:, 5]]),
                          ['P_w', 'N_w', 'ER_w', 'l2_distinct', 'l3_collision', 'l3_refine_bits']),
        'D_plus_ERw': (np.column_stack([Xa, er_w]), FEATS + ['ER_w']),
        'E_plus_PwNw': (np.column_stack([Xa, p_w, n_w]), FEATS + ['P_w', 'N_w']),
        'F_plus_lift': (np.column_stack([Xa, er_l]), FEATS + ['ER_w_lift']),
    }
    # every candidate uses the same row set: drop rows with any NaN in any candidate's columns
    ok = clean(er_w, p_w, n_w, er_l)
    n_dropped = int((~ok).sum())
    results = {'schema': 'genrec.issue278.model_comparison/1',
               'n_rows': int(ok.sum()), 'n_dropped_nan': n_dropped,
               'dropped_groups': [g for g, keep in zip(groups.tolist(), ok) if not keep][:10],
               'na_counts': na_counts, 'n_strict': int((strict & ok).sum()),
               'n_high_strict': int(((y > 0.06) & strict & ok).sum()),
               'n_high_full': int(((y > 0.06) & ok).sum()),
               'candidates': {}}
    yk = y[ok]
    fold_k = fold[ok]
    groups_k = groups[ok]
    strict_k = strict[ok]
    preds = {}
    for name, (X, feats) in cand.items():
        Xk = X[ok]
        pred, alphas = oof_predictions(Xk, yk, fold_k)
        preds[name] = pred
        results['candidates'][name] = {
            'features': feats, 'alphas': alphas,
            'oof': metrics(yk, pred), 'confusion_0.06': confusion(yk, pred),
            'strict': metrics(yk[strict_k], pred[strict_k]),
            'strict_confusion': confusion(yk[strict_k], pred[strict_k]),
        }
    # controls
    controls = {'fold_mean': np.full(len(yk), yk.mean()),
                'fold_median': np.full(len(yk), float(np.median(yk)))}
    er_idx = FEATS.index('l1_hard_er')
    er_pred, _ = oof_predictions(Xa[ok][:, [er_idx]], yk, fold_k)
    controls['single_er'] = er_pred
    for name, pred in controls.items():
        results['candidates'][name] = {'oof': metrics(yk, pred), 'confusion_0.06': confusion(yk, pred),
                                       'strict': metrics(yk[strict_k], pred[strict_k]),
                                       'strict_confusion': confusion(yk[strict_k], pred[strict_k])}

    # paired SID-group bootstrap of MAE(A) - MAE(candidate)
    rng = np.random.default_rng(42)
    uniq = sorted(set(groups_k.tolist()))
    idx_of = {g: np.where(groups_k == g)[0] for g in uniq}
    base = np.abs(yk - preds['A_control'])
    boot = {}
    for name in ('B_replacement', 'D_plus_ERw', 'E_plus_PwNw', 'F_plus_lift'):
        alt = np.abs(yk - preds[name])
        diffs = []
        for _ in range(2000):
            pick = rng.choice(uniq, len(uniq), replace=True)
            sel = np.concatenate([idx_of[g] for g in pick])
            diffs.append(float(base[sel].mean() - alt[sel].mean()))
        boot[name] = {'mae_diff_A_minus_candidate': float(base.mean() - alt.mean()),
                      'ci95': [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                      'passes_ci': bool(np.percentile(diffs, 2.5) > 0)}
    results['bootstrap_vs_A'] = boot

    # cross-fitted residual analysis (C9): does ER_w explain the A residual beyond the six features?
    resid = yk - preds['A_control']
    delta_er = er_w[ok] - er_u[ok]
    def gcorr(x, r):
        m = np.isfinite(x) & np.isfinite(r)
        return float(stats.spearmanr(x[m], r[m]).correlation) if m.sum() > 10 else None
    results['residual_analysis'] = {
        'spearman_resid_vs_delta_ER': gcorr(delta_er, resid),
        'spearman_resid_vs_ERw': gcorr(er_w[ok], resid),
        'spearman_resid_vs_ERuniform': gcorr(er_u[ok], resid),
        'spearman_resid_vs_lift_ER': gcorr(er_l[ok], resid),
        'corr_ERw_vs_ERuniform': gcorr(er_w[ok], er_u[ok]),
        'corr_ERw_vs_L1pos': gcorr(er_w[ok], Xa[ok][:, 0]),
        'spearman_ERw_vs_label': gcorr(er_w[ok], yk),
        'note': 'OOF residuals of the control A against importance-derived quantities; correlations are '
                'descriptive and cross-fitted (the predictions never see their own test rows)',
    }
    # calibration and high/low strata residuals
    lo = yk <= np.quantile(yk, 1 / 3)
    mid = (yk > np.quantile(yk, 1 / 3)) & (yk <= np.quantile(yk, 2 / 3))
    hi = yk > np.quantile(yk, 2 / 3)
    results['residual_by_label_stratum'] = {
        name: {'low': float(np.mean(resid[lo])), 'mid': float(np.mean(resid[mid])),
               'high': float(np.mean(resid[hi])),
               'low_mae': float(np.mean(np.abs(resid[lo]))),
               'high_mae': float(np.mean(np.abs(resid[hi])))}
        for name, resid in [('D_plus_ERw', yk - preds['D_plus_ERw']),
                            ('A_control', resid)]}
    # R2 diagnostic case
    r2_sha = '9f6bf6731e88f5ec3e83bdf54b99784a5eafbf9412efe5e63f3b46b12cecaf97'
    r2_rows = [i for i, u in enumerate(used) if u['live'] == r2_sha]
    results['r2_diagnostic'] = {
        'present_in_common_queue': bool(r2_rows),
        'note': '#274 R2 has no public label in the common queue at model time (its label is a #277 '
                'prospective outcome); it is not used to fit or select anything here',
    }
    (P / 'MODEL_COMPARISON.json').write_text(json.dumps(results, indent=1) + '\n')
    oof_rows = [{'group': g, 'sid_sha256': u['live'], 'y': float(yy),
                 'pred_A': float(pa), 'pred_E': float(pe), 'er_w': (None if np.isnan(ew) else float(ew)),
                 'er_uniform': (None if np.isnan(eu) else float(eu))}
                for g, u, yy, pa, pe, ew, eu in zip(groups_k, [u for u, k in zip(used, ok) if k], yk,
                                                    preds['A_control'], preds['E_plus_PwNw'],
                                                    er_w[ok], er_u[ok])]
    (P / 'OOF_ROWS.json').write_text(json.dumps({'schema': 'genrec.issue278.oof_rows/1', 'rows': oof_rows}, indent=1) + '\n')
    print(json.dumps({
        'n_rows': results['n_rows'], 'n_dropped': n_dropped,
        'oof': {k: {'mae': round(v['oof']['mae'], 6), 'rho': None if v['oof']['spearman'] is None else round(v['oof']['spearman'], 3)}
                for k, v in results['candidates'].items()},
        'bootstrap': boot,
        'residual': results['residual_analysis'],
    }, indent=1))


if __name__ == '__main__':
    main()
