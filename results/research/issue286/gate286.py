"""Issue #286: gated two-regime predictor (region-gate, cross-fitted residual experts), comparisons."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import optimize, stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '.')
sys.path.insert(0, 'results/research/issue285')
sys.path.insert(0, 'results/research/issue286')
import common285 as W
import common284 as C

OUT = Path('results/research/issue286')
ALPHA_EXPERT = 1.0
LAMBDAS = (0.25, 0.5, 0.75, 1.0)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def fit_constrained_logit(design, target, l2=1.0):
    design = np.asarray(design, float); target = np.asarray(target, float)
    if target.min() == target.max():
        return np.zeros(design.shape[1])
    n = len(target)
    def nll(b):
        z = design @ b
        ll = np.sum(target * z - np.logaddexp(0, z))
        return -ll / n + 0.5 * (l2 / n) * float(np.sum(b[1:] ** 2))
    b0 = np.zeros(design.shape[1]); b0[0] = np.log((target.mean() + 1e-9) / (1 - target.mean() + 1e-9))
    bounds = [(None, None)] + [(0.0, None)] * (design.shape[1] - 1)
    res = optimize.minimize(nll, b0, method='L-BFGS-B', bounds=bounds)
    return res.x


def fit_ridge(X, y, alpha):
    sc = StandardScaler().fit(X)
    rg = Ridge(alpha=alpha).fit(sc.transform(X), y)
    return {'sc': sc, 'rg': rg}


def pred_ridge(m, X):
    return m['rg'].predict(m['sc'].transform(X))


def fit_r6(idx, groups):
    a = C.inner_alpha(X6[idx], y[idx], groups)
    sc = StandardScaler().fit(X6[idx])
    rg = Ridge(alpha=a).fit(sc.transform(X6[idx]), y[idx])
    return {'alpha': a, 'sc': sc, 'rg': rg}


def pred_r6(m, idx):
    return m['rg'].predict(m['sc'].transform(X6[idx]))


def crossfit_r6(idx, groups):
    pred = np.zeros(len(idx))
    for h in sorted(set(groups.tolist())):
        tr = groups != h; te = ~tr
        m = fit_r6(idx[tr], groups[tr])
        pred[te] = pred_r6(m, idx[te])
    return pred


def fit_bundle(train_idx):
    groups = fold[train_idx]
    qn = float(np.quantile(N[train_idx], 1 / 3)); qc = float(np.quantile(Cv[train_idx], 1 / 3))
    n_mu, n_sd = float(N[train_idx].mean()), float(N[train_idx].std(ddof=0))
    c_mu, c_sd = float(Cv[train_idx].mean()), float(Cv[train_idx].std(ddof=0))
    n_std = (N - n_mu) / (n_sd or 1.0); c_std = (Cv - c_mu) / (c_sd or 1.0)
    target = ((N[train_idx] <= qn) & (Cv[train_idx] <= qc)).astype(float)
    design = np.column_stack([np.ones(len(train_idx)), -n_std[train_idx], -c_std[train_idx]])
    b = fit_constrained_logit(design, target)
    g = sigmoid(b[0] - b[1] * n_std - b[2] * c_std)
    r6 = fit_r6(train_idx, groups)
    resid_cf = y[train_idx] - crossfit_r6(train_idx, groups)
    e_good = fit_ridge(np.column_stack([Pv, ERv])[train_idx], resid_cf, ALPHA_EXPERT)
    e_other = fit_ridge(X6[train_idx], resid_cf, ALPHA_EXPERT)
    return {'b': b, 'g': g, 'r6': r6, 'e_good': e_good, 'e_other': e_other,
            'qn': qn, 'qc': qc, 'n_mu': n_mu, 'n_sd': n_sd, 'c_mu': c_mu, 'c_sd': c_sd}


def predict_bundle(bundle, idx, lam):
    r6p = pred_r6(bundle['r6'], idx)
    g = bundle['g'][idx]
    eg = pred_ridge(bundle['e_good'], np.column_stack([Pv, ERv])[idx])
    eo = pred_ridge(bundle['e_other'], X6[idx])
    return r6p + lam * (g * eg + (1 - g) * eo)


def auc_brier(score, target):
    target = np.asarray(target, bool); score = np.asarray(score, float)
    if target.sum() in (0, len(target)):
        return None, None
    u = stats.mannwhitneyu(score[target], score[~target], alternative='two-sided').statistic
    auc = float(u / (target.sum() * (~target).sum()))
    brier = float(np.mean((score - target.astype(float)) ** 2))
    return round(auc, 4), round(brier, 5)


def main() -> None:
    global X6, y, fold, N, Cv, Pv, ERv
    d = W.load_table()
    y = d['y']; fold = d['fold']; X6 = d['X6']; N = d['N']; Cv = d['C']; Pv = d['P']; ERv = d['ER']
    strong = d['strong']; unit = d['unit']
    arms = {k: np.zeros(len(y)) for k in ('A', 'B', 'C_hard', 'D_soft', 'E_main',
                                          'D_Ponly', 'D_ERonly', 'D_gateN', 'D_gateC')}
    label_gate_oof = np.zeros(len(y)); label_gate_coef = {}
    folds_info = []
    for f in sorted(set(fold.tolist())):
        tr = np.where(fold != f)[0]; te = np.where(fold == f)[0]
        bundle = fit_bundle(tr)
        # lambda inner selection
        scores = {lam: [] for lam in LAMBDAS}
        for h in sorted(set(fold[tr].tolist())):
            tr2 = tr[fold[tr] != h]; te2 = tr[fold[tr] == h]
            bh = fit_bundle(tr2)
            for lam in LAMBDAS:
                pr = predict_bundle(bh, te2, lam)
                scores[lam].append(float(np.mean(np.abs(y[te2] - pr))))
        lam = min(LAMBDAS, key=lambda l: np.mean(scores[l]))
        arms['E_main'][te] = predict_bundle(bundle, te, lam)
        arms['A'][te] = pred_r6(bundle['r6'], te)
        # B fixed single branch [P,ER]
        mB = fit_ridge(np.column_stack([Pv, ERv])[tr], y[tr], ALPHA_EXPERT)
        arms['B'][te] = pred_ridge(mB, np.column_stack([Pv, ERv])[te])
        g = bundle['g']
        # C hard gate
        hi = g[tr] >= 0.5; lo = ~hi
        mh = fit_ridge(np.column_stack([Pv, ERv])[tr][hi], y[tr][hi], ALPHA_EXPERT)
        ml = fit_ridge(X6[tr][lo], y[tr][lo], ALPHA_EXPERT)
        gte = g[te] >= 0.5
        predC = np.where(gte, pred_ridge(mh, np.column_stack([Pv, ERv])[te]),
                         pred_ridge(ml, X6[te]))
        arms['C_hard'][te] = predC
        # D soft
        mg = fit_ridge(np.column_stack([Pv, ERv])[tr], y[tr], ALPHA_EXPERT)
        mo = fit_ridge(X6[tr], y[tr], ALPHA_EXPERT)
        arms['D_soft'][te] = g[te] * pred_ridge(mg, np.column_stack([Pv, ERv])[te]) + \
                             (1 - g[te]) * pred_ridge(mo, X6[te])
        # ablations
        for name, xg in (('D_Ponly', Pv[:, None]), ('D_ERonly', ERv[:, None])):
            mgx = fit_ridge(xg[tr], y[tr], ALPHA_EXPERT)
            arms[name][te] = g[te] * pred_ridge(mgx, xg[te]) + (1 - g[te]) * pred_ridge(mo, X6[te])
        # single-variable gates in the D formulation
        for name, cov in (('D_gateN', N), ('D_gateC', Cv)):
            mu, sd = cov[tr].mean(), cov[tr].std(ddof=0)
            cs = (cov - mu) / (sd or 1.0)
            b1 = fit_constrained_logit(np.column_stack([np.ones(len(tr)), -cs[tr]]),
                                       ((N[tr] <= bundle['qn']) & (Cv[tr] <= bundle['qc'])).astype(float))
            g1 = sigmoid(b1[0] - b1[1] * cs)
            arms[name][te] = g1[te] * pred_ridge(mg, np.column_stack([Pv, ERv])[te]) + \
                             (1 - g1[te]) * pred_ridge(mo, X6[te])
        # diagnostic label gate (trained on the training labels only)
        bl = fit_constrained_logit(np.column_stack([np.ones(len(tr)),
                                                    -(N[tr] - bundle['n_mu']) / (bundle['n_sd'] or 1.0),
                                                    -(Cv[tr] - bundle['c_mu']) / (bundle['c_sd'] or 1.0)]),
                                   (y[tr] > 0.06).astype(float))
        label_gate_oof[te] = sigmoid(bl[0] - bl[1] * (N[te] - bundle['n_mu']) / (bundle['n_sd'] or 1.0)
                                     - bl[2] * (Cv[te] - bundle['c_mu']) / (bundle['c_sd'] or 1.0))
        label_gate_coef[str(f)] = [round(float(v), 5) for v in bl]
        folds_info.append({'fold': int(f), 'lambda': lam,
                           'lam_scores': {str(k): round(float(np.mean(v)), 7) for k, v in scores.items()},
                           'gate_b': [round(float(v), 5) for v in bundle['b']],
                           'gate_edge_N': bundle['qn'], 'gate_edge_C': bundle['qc'],
                           'g_train_mean': round(float(g[tr].mean()), 4),
                           'g_train_hi_count': int((g[tr] >= 0.5).sum()),
                           'g_test_mean': round(float(g[te].mean()), 4),
                           'g_test_hi_count': int((g[te] >= 0.5).sum()),
                           'r6_alpha': float(bundle['r6']['alpha'])})
    out = {'schema': 'genrec.issue286.model/1', 'n_rows': len(y),
           'n_strong': int(strong.sum()), 'n_high': int((y > 0.06).sum()),
           'n_high_strong': int(((y > 0.06) & strong).sum()),
           'n_above_0.065': int((y > 0.065).sum()),
           'folds': folds_info, 'arms': {}}
    A = arms['A']
    base = C.metrics(y, A)
    out['baseline_A'] = {'oof': base, 'confusion': C.confusion(y, A),
                         'strict': C.metrics(y[strong], A[strong])}
    assert abs(base['mae'] - 0.002971939445864855) < 2e-6, base
    for name, pred in arms.items():
        if name == 'A':
            continue
        out['arms'][name] = {'oof': C.metrics(y, pred), 'confusion': C.confusion(y, pred),
                             'strict': C.metrics(y[strong], pred[strong]),
                             'boot': C.cluster_bootstrap(A, pred, y, unit)}
    out['passes'] = {k: bool(v['boot']['ci95'][0] > 0) for k, v in out['arms'].items()}
    # calibration: region gate vs the feature region; diagnostic label gate vs >0.06
    g = W.gates(d)
    region = ((g['level_N'] == -1) & (g['level_C'] == -1)).astype(float)
    # the region gate score is not stored per row; recompute the per-fold gate at the outer level
    gate_region = np.zeros(len(y)); gate_label = label_gate_oof
    for f in sorted(set(fold.tolist())):
        tr = fold != f; te = fold == f
        bundle = fit_bundle(np.where(tr)[0])
        gate_region[te] = bundle['g'][te]
    out['calibration'] = {'region_gate_vs_region': dict(zip(('auc', 'brier'), auc_brier(gate_region, region))),
                          'label_gate_vs_high': dict(zip(('auc', 'brier'), auc_brier(gate_label, y > 0.06))),
                          'label_gate_coef_per_fold': label_gate_coef,
                          'note': 'a gate AUC/Brier is classification; it is not regression accuracy'}
    # cell errors and counterexamples
    cells = W.cell_labels(g['level_N'], g['level_C'])
    errA = np.abs(y - A); errE = np.abs(y - arms['E_main'])
    out['cells'] = {}
    for c in ('LL', 'HH', 'other'):
        m = cells == c
        if m.sum():
            out['cells'][c] = {'n': int(m.sum()), 'mae_A': round(float(errA[m].mean()), 6),
                               'mae_E': round(float(errE[m].mean()), 6),
                               'bias_A': round(float((y - A)[m].mean()), 6),
                               'bias_E': round(float((y - arms['E_main'])[m].mean()), 6)}
    # counterexamples: largest |err_E| - |err_A| changes
    delta = errE - errA
    order = np.argsort(-np.abs(delta))
    out['counterexamples'] = [{'i': int(i), 'y': round(float(y[i]), 6), 'A': round(float(A[i]), 6),
                               'E': round(float(arms['E_main'][i]), 6),
                               'd_abs_error': round(float(delta[i]), 6)} for i in order[:6]]
    # power
    def power(pred):
        dd = np.abs(y - A) - np.abs(y - pred)
        mean, sd = float(dd.mean()), float(dd.std(ddof=1))
        req = ((1.645 + 0.8416) * sd / mean) ** 2 if mean > 0 else None
        return {'mean_diff': round(mean, 8), 'sd': round(sd, 8),
                'n_required_80pct': None if req is None else int(np.ceil(req)),
                'min_detectable_mean_at_129': round((1.645 + 0.8416) * sd / np.sqrt(len(dd)), 8)}
    out['power'] = {k: power(v) for k, v in arms.items() if k != 'A'}
    # BH context over the five main arms
    names = ('B', 'C_hard', 'D_soft', 'E_main')
    ci = {k: out['arms'][k]['boot']['ci95'] for k in names}
    out['family_cis'] = ci
    (OUT / 'MODEL_286.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({'baseline': base, 'arms': {k: v['oof'] for k, v in out['arms'].items()},
                      'passes': out['passes'],
                      'ci': {k: [round(x, 7) for x in v['boot']['ci95']] for k, v in out['arms'].items()},
                      'calibration': out['calibration'], 'cells': out['cells'],
                      'power': out['power'], 'folds': folds_info}, indent=1)[:5000])


if __name__ == '__main__':
    main()
