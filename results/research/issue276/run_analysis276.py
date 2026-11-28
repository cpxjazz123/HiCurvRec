"""Issue #276 — SID-group 5-fold OOF analysis with controls, PySR, bootstrap CIs.

Reads SID_TABLE.json + FOLDS.json (frozen) and writes ANALYSIS_276.json. Pre-registered design:
folds by SID group (never family), band-stratified; features/preprocessing fitted on training folds
only; formula selection by inner SID-group CV; the outer labels never select a formula.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

P = Path('results/research/issue276')
SEED = 42
FEATS = ['l1_pos_hit', 'l1_neg_hit', 'l1_hard_er', 'l2_pos_hit', 'l2_neg_hit',
         'l2_distinct', 'l3_collision', 'l3_refine_bits']
FEATS_275 = ['l1_pos_hit', 'l1_neg_hit', 'l1_hard_er', 'l2_distinct', 'l3_collision', 'l3_refine_bits']
FEATS_VEC = ['pr_z', 'pr_q', 'quant_err', 'recon_mse', 'recon_cos']


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = {'mae': float(np.mean(np.abs(y - p))), 'rmse': float(np.sqrt(np.mean((y - p) ** 2)))}
    if len(y) > 2 and np.std(p) > 0:
        m['spearman'] = float(stats.spearmanr(y, p).correlation)
        m['kendall'] = float(stats.kendalltau(y, p).correlation)
    else:
        m['spearman'] = None
        m['kendall'] = None
    rev = tot = 0
    for i in range(len(y)):
        for k in range(i + 1, len(y)):
            if (y[i] - y[k]) * (p[i] - p[k]) < 0:
                rev += 1
            if y[i] != y[k]:
                tot += 1
    m['reversals'] = rev / tot if tot else None
    return m


def build_rows(table):
    rows = []
    for g in table['groups']:
        cand = []
        for r in g['rows']:
            f = r.get('features') or {}
            if all(isinstance(f.get(k), (int, float)) for k in FEATS):
                cand.append((f, r))
        if not cand:
            continue
        f, r = cand[0]
        labels = [x.get('label_recall@10') for x in g['rows'] if isinstance(x.get('label_recall@10'), (int, float))]
        if not labels:
            continue
        vec = next((x['vector'] for x in g['rows'] if x.get('vector')), None)
        vecf = next((x.get('vector_forward') for x in g['rows'] if x.get('vector_forward')), None)
        rows.append({
            'group': g['group'],
            'label': float(np.mean(labels)),
            'n_label_rows': len(labels),
            'features': {k: float(f[k]) for k in FEATS},
            'vector': vec,
            'vector_forward': vecf,
            'arms': [a.get('arm') for r2 in g['rows'] for a in r2.get('arms', [])],
        })
    return rows


def inner_ridge_alpha(X, y, groups):
    best, best_err = 1.0, np.inf
    sc = StandardScaler().fit(X)
    for alpha in (0.01, 0.1, 1.0, 10.0):
        errs = []
        for g in sorted(set(groups)):
            mask = groups != g
            if mask.sum() < 5:
                continue
            rg = Ridge(alpha=alpha).fit(sc.transform(X[mask]), y[mask])
            errs.append(np.mean(np.abs(rg.predict(sc.transform(X[~mask])) - y[~mask])))
        if errs and np.mean(errs) < best_err:
            best, best_err = alpha, float(np.mean(errs))
    return best


def fit_fold(train, test, feats):
    Xtr = np.array([[r['features'][k] for k in feats] for r in train])
    ytr = np.array([r['label'] for r in train])
    Xte = np.array([[r['features'][k] for k in feats] for r in test])
    yte = np.array([r['label'] for r in test])
    out = {'mean': np.full(len(yte), ytr.mean()), 'median': np.full(len(yte), np.median(ytr))}
    # best single feature by train Spearman
    best, best_rho = None, -2
    for i, f in enumerate(feats):
        if np.std(Xtr[:, i]) == 0:
            continue
        rho = stats.spearmanr(Xtr[:, i], ytr).correlation
        if rho is not None and rho > best_rho:
            best, best_rho = i, rho
    if best is not None:
        A = np.vstack([Xtr[:, best], np.ones(len(ytr))]).T
        coef = np.linalg.lstsq(A, ytr, rcond=None)[0]
        out['best_single'] = Xte[:, best] * coef[0] + coef[1]
        out['best_feature'] = feats[best]
    else:
        out['best_single'] = out['mean']
        out['best_feature'] = None
    # ridge with inner SID-group CV
    sc = StandardScaler().fit(Xtr)
    gtr = np.array([r['group'] for r in train])
    alpha = inner_ridge_alpha(Xtr, ytr, gtr)
    rg = Ridge(alpha=alpha).fit(sc.transform(Xtr), ytr)
    out['ridge'] = rg.predict(sc.transform(Xte))
    out['ridge_alpha'] = alpha
    # #275 six-feature ridge refit inside the fold (fixed formula)
    if set(FEATS_275) <= set(feats):
        i275 = [feats.index(k) for k in FEATS_275]
        a6 = inner_ridge_alpha(Xtr[:, i275], ytr, gtr)
        sc6 = StandardScaler().fit(Xtr[:, i275])
        rg6 = Ridge(alpha=a6).fit(sc6.transform(Xtr[:, i275]), ytr)
        out['ridge6_275'] = rg6.predict(sc6.transform(Xte[:, i275]))
    # prior: l1_pos_hit alone (1-D linear map)
    i = feats.index('l1_pos_hit')
    A = np.vstack([Xtr[:, i], np.ones(len(ytr))]).T
    coef = np.linalg.lstsq(A, ytr, rcond=None)[0]
    out['prior_l1pos'] = Xte[:, i] * coef[0] + coef[1]
    # PySR (frozen config in ANALYSIS_CONTRACT.md)
    from pysr import PySRRegressor
    import os
    (P / 'pysr').mkdir(exist_ok=True)
    model = PySRRegressor(niterations=100, populations=8,
                          binary_operators=['+', '-', '*', '/'], unary_operators=['square'],
                          maxsize=10, maxdepth=4, timeout_in_seconds=60, random_state=SEED,
                          deterministic=True, procs=0, multithreading=False,
                          model_selection='best', progress=False, verbosity=0)
    old_cwd = os.getcwd()
    os.chdir(P / 'pysr')
    try:
        model.fit(Xtr, ytr, variable_names=feats)
    finally:
        os.chdir(old_cwd)
    out['pysr'] = model.predict(Xte)
    expr = model.sympy()
    import sympy as sp
    fn = sp.lambdify([sp.Symbol(n) for n in feats], expr, modules=['numpy'])
    val = np.asarray(fn(*[Xte[:, i] for i in range(len(feats))]), dtype=float)
    out['pysr_expr'] = str(expr)
    out['pysr_expr_values'] = val
    out['pysr_expr_max_abs_diff'] = float(np.max(np.abs(val - out['pysr']))) if np.all(np.isfinite(val)) else None
    try:
        eq = model.equations_
        out['pareto'] = [{'complexity': float(e['complexity']), 'loss': float(e['loss']),
                          'equation': str(e['equation'])} for e in eq[['complexity', 'loss', 'equation']].to_dict('records')]
    except Exception:
        out['pareto'] = []
    return out, yte


def oof(rows, folds, feats, label):
    assigned = np.array([folds['assignment'].get(r['group'], -1) for r in rows])
    keep = assigned >= 0
    rows = [r for r, k in zip(rows, keep) if k]
    assigned = assigned[keep]
    preds, exprs, y = {}, {}, []
    order = []
    for k in range(5):
        train = [r for r, a in zip(rows, assigned) if a != k]
        test = [r for r, a in zip(rows, assigned) if a == k]
        if len(train) < 8 or len(test) < 1:
            continue
        out, yte = fit_fold(train, test, feats)
        for key in ('mean', 'median', 'best_single', 'ridge', 'ridge6_275', 'prior_l1pos', 'pysr'):
            preds.setdefault(key, []).extend(list(out[key]))
        exprs[k] = {'n_test': len(test), 'expr': out['pysr_expr'],
                    'expr_max_abs_diff': out['pysr_expr_max_abs_diff'],
                    'best_feature': out['best_feature'], 'ridge_alpha': out['ridge_alpha'],
                    'pareto': out['pareto']}
        order.extend([r['group'] for r in test])
        y.extend([r['label'] for r in test])
    res = {'label': label, 'n': len(y), 'feature_list': feats, 'folds': exprs,
           'metrics': {k: metrics(y, v) for k, v in preds.items()},
           'predictions': {k: [float(x) for x in v] for k, v in preds.items()},
           'y': [float(x) for x in y], 'order': order}
    # paired bootstrap over SID groups vs the strongest control (lowest OOF MAE)
    groups = list(order)
    err = {k: np.abs(np.array(preds[k]) - np.array(y)) for k in preds}
    ctrls = [k for k in preds if k != 'pysr']
    best_ctrl = min(ctrls, key=lambda k: float(err[k].mean()))
    res['strongest_control'] = best_ctrl
    res['mae_diff_pysr_minus_control'] = float(err['pysr'].mean() - err[best_ctrl].mean())
    rng = np.random.default_rng(SEED)
    uniq = sorted(set(groups))
    idx_of = {g: [i for i, x in enumerate(groups) if x == g] for g in uniq}
    diffs = []
    for _ in range(2000):
        pick = rng.choice(uniq, len(uniq), replace=True)
        sel = [i for g in pick for i in idx_of[g]]
        diffs.append(float(err['pysr'][sel].mean() - err[best_ctrl][sel].mean()))
    res['mae_diff_ci95'] = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
    # threshold >0.06
    yv = np.array(y)
    pv = np.array(preds['pysr'])
    pos = int((yv > 0.06).sum())
    res['threshold_0.06'] = {'positives': pos,
                             'evaluable': bool(pos > 0 and pos < len(yv)),
                             'tp': int(((pv > 0.06) & (yv > 0.06)).sum()),
                             'fp': int(((pv > 0.06) & (yv <= 0.06)).sum()),
                             'fn': int(((pv <= 0.06) & (yv > 0.06)).sum())}
    return res


def main() -> None:
    table = json.load(open(P / 'SID_TABLE.json'))
    folds = json.load(open(P / 'FOLDS.json'))
    assert folds['table_sha256'] == sha(P / 'SID_TABLE.json'), 'SID_TABLE changed after folds were frozen'
    rows = build_rows(table)
    out = {'schema': 'genrec.issue276.analysis/1', 'seed': SEED,
           'table_sha256': sha(P / 'SID_TABLE.json'), 'folds_sha256': sha(P / 'FOLDS.json'),
           'n_groups_with_features_and_label': len(rows),
           'label_range': [min(r['label'] for r in rows), max(r['label'] for r in rows)],
           'n_high_recall': sum(1 for r in rows if r['label'] > 0.06)}
    out['Q_SID'] = oof(rows, folds, FEATS, 'Q_SID')
    vec_rows = [r for r in rows if r.get('vector_forward') and
                all(isinstance(r['vector_forward'].get(k), (int, float)) for k in FEATS_VEC)]
    out['Q_VEC_audit'] = {'n': len(vec_rows),
                          'high_recall': sum(1 for r in vec_rows if r['label'] > 0.06)}
    if len(vec_rows) >= 20:
        for r in vec_rows:
            r['features'] = {**r['features'], **{k: float(r['vector_forward'][k]) for k in FEATS_VEC}}
        out['Q_VEC'] = oof(vec_rows, folds, FEATS + FEATS_VEC, 'Q_VEC')
    (P / 'ANALYSIS_276.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({'n': out['n_groups_with_features_and_label'], 'label_range': out['label_range'],
                      'n_high': out['n_high_recall'], 'q_sid': out['Q_SID']['metrics'],
                      'ctrl': out['Q_SID']['strongest_control'],
                      'diff': out['Q_SID']['mae_diff_pysr_minus_control'],
                      'ci': out['Q_SID']['mae_diff_ci95']}, indent=1))


if __name__ == '__main__':
    main()
