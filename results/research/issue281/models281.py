"""Issue #281 — pre-registered nonlinear/piecewise comparisons of the six SID metrics."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

P = Path('results/research/issue281')
FEATS = ['l1_pos_hit', 'l1_neg_hit', 'l1_hard_er', 'l2_distinct', 'l3_collision', 'l3_refine_bits']
ALPHAS = (0.01, 0.1, 1.0, 10.0)
KNOT_QS = (25, 50, 75)
MIN_SIDE = 0.20


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {'mae': float(np.mean(np.abs(y - p))), 'rmse': float(np.sqrt(np.mean((y - p) ** 2))),
            'spearman': float(stats.spearmanr(y, p).correlation) if len(y) > 2 and np.std(p) > 0 else None,
            'kendall': float(stats.kendalltau(y, p).correlation) if len(y) > 2 and np.std(p) > 0 else None}


def confusion(y, p, cut=0.06):
    y, p = np.array(y, float), np.array(p, float)
    return {'positives': int((y > cut).sum()), 'tp': int(((p > cut) & (y > cut)).sum()),
            'fp': int(((p > cut) & (y <= cut)).sum()), 'fn': int(((p <= cut) & (y > cut)).sum()),
            'tn': int(((p <= cut) & (y <= cut)).sum())}


def design(X, model, knots=None):
    if model == 'A':
        return X
    if model == 'B':
        return np.column_stack([X, X ** 2])
    if model == 'C':
        return np.column_stack([X, X[:, 0] * X[:, 1], X[:, 2] * X[:, 4]])
    if model == 'D':
        Z = [X]
        for j, q in (knots or []):
            Z.append(np.maximum(0.0, X[:, j] - q)[:, None])
        return np.column_stack(Z)
    raise ValueError(model)


def ridge_fit_predict(Xtr, ytr, Xte, alpha):
    sc = StandardScaler().fit(Xtr)
    rg = Ridge(alpha=alpha).fit(sc.transform(Xtr), ytr)
    return rg.predict(sc.transform(Xte))


def inner_cv(X, y, groups, model, knots=None):
    """Leave-one-group-out inner OOF; returns (best_alpha, oof_pred, mae)."""
    best = (None, None, np.inf)
    Xd = design(X, model, knots)
    for alpha in ALPHAS:
        oof = np.zeros(len(y))
        evaluated = np.zeros(len(y), dtype=bool)
        group_maes = []
        for g in sorted(set(groups.tolist())):
            m = groups != g
            if m.sum() < 5:
                continue
            oof[~m] = ridge_fit_predict(Xd[m], y[m], Xd[~m], alpha)
            evaluated[~m] = True
            group_maes.append(float(np.mean(np.abs(y[~m] - oof[~m]))))
        if not group_maes:
            continue
        # #277's pre-registered inner rule: macro average over inner groups
        mae = float(np.mean(group_maes))
        if mae < best[2]:
            best = (alpha, oof, mae, evaluated)
    return best[0], best[1], best[2], best[3]


def main() -> None:
    table = json.load(open('results/research/issue276/SID_TABLE.json'))
    folds = json.load(open('results/research/issue276/FOLDS.json'))
    used = []
    for g in table['groups']:
        cand = [r for r in g['rows'] if all(isinstance((r.get('features') or {}).get(k), (int, float)) for k in FEATS)]
        labels = [r['label_recall@10'] for r in g['rows'] if isinstance(r['label_recall@10'], (int, float))]
        if not cand or not labels:
            continue
        r = cand[0]
        ident = sorted({a.get('identity') for a in r.get('arms', []) if a.get('identity')})
        srcs = [x.get('source') for x in r.get('label_sources', [])]
        strong = ('L1_ledger' in ident) or any(str(x).startswith('inventory_unit') for x in srcs)
        used.append({'group': g['group'], 'y': float(np.mean(labels)), 'strong': strong,
                     'x': [float(r['features'][k]) for k in FEATS]})
    X = np.array([u['x'] for u in used]); y = np.array([u['y'] for u in used])
    groups = np.array([u['group'] for u in used]); fold = np.array([folds['assignment'][g] for g in groups])
    strong = np.array([u['strong'] for u in used])
    out = {'schema': 'genrec.issue281.models/1', 'n_rows': len(y), 'n_strict': int(strong.sum()),
           'n_high_full': int((y > 0.06).sum()), 'n_high_strict': int(((y > 0.06) & strong).sum()),
           'label_range': [float(y.min()), float(y.max())], 'models': {}, 'selected': [], 'auto_pred': None}
    per_model = {m: np.zeros(len(y)) for m in ('A', 'B', 'C', 'D', 'F')}
    auto = np.zeros(len(y))
    selections = []
    for k in range(5):
        tr, te = fold != k, fold == k
        Xtr, ytr, gtr = X[tr], y[tr], groups[tr]
        inner = {}
        # A-D
        for m in ('A', 'B', 'C', 'D'):
            if m == 'D':
                best = None
                for q3, q5 in itertools.product(KNOT_QS, KNOT_QS):
                    knots = [(2, float(np.percentile(Xtr[:, 2], q3))), (4, float(np.percentile(Xtr[:, 4], q5)))]
                    # frozen minimum side support on the training fold
                    ok = all(np.mean(Xtr[:, j] >= q) >= MIN_SIDE and np.mean(Xtr[:, j] < q) >= MIN_SIDE
                             for j, q in knots)
                    if not ok:
                        continue
                    alpha, oof, mae, _ = inner_cv(Xtr, ytr, gtr, 'D', knots)
                    if best is None or mae < best[0]:
                        best = (mae, alpha, oof, knots)
                if best is None:
                    inner['D'] = (np.inf, None, None, None)
                else:
                    mae, alpha, oof, knots = best
                    inner['D'] = (mae, alpha, oof, knots)
            else:
                alpha, oof, mae, _ = inner_cv(Xtr, ytr, gtr, m)
                inner[m] = (mae, alpha, oof, None)
        # F: calibrate A's inner OOF
        maeA, alphaA, oofA, _ = inner['A']
        # calibration uses only the rows the inner CV actually evaluated
        alphaA2, oofA2, maeA2, evaluatedA = inner_cv(Xtr, ytr, gtr, 'A')
        A1 = np.vstack([oofA2[evaluatedA], np.ones(evaluatedA.sum())]).T
        coef = np.linalg.lstsq(A1, ytr[evaluatedA], rcond=None)[0]
        cal = oofA2 * coef[0] + coef[1]
        inner['F'] = (float(np.mean(np.abs(ytr - cal))), None, cal, tuple(coef))
        best_m = min(inner, key=lambda m: (inner[m][0], 'ABCD F'.index(m) if False else
                                           {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'F': 4}[m]))
        selections.append({'fold': k, 'inner_mae': {m: inner[m][0] for m in inner}, 'best': best_m})
        # refit each model on the outer training fold and predict the test fold
        for m in ('A', 'B', 'C', 'D'):
            mae, alpha, _, knots = inner[m]
            if alpha is None:
                continue
            per_model[m][te] = ridge_fit_predict(design(Xtr, m, knots), ytr, design(X[te], m, knots), alpha)
        # F outer prediction: A's outer prediction calibrated with the inner map
        pA_te = ridge_fit_predict(design(Xtr, 'A'), ytr, design(X[te], 'A'), alphaA)
        per_model['F'][te] = pA_te * coef[0] + coef[1]
        # auto-selected prediction
        if best_m == 'F':
            auto[te] = per_model['F'][te]
        else:
            auto[te] = per_model[best_m][te]
    out['auto_pred'] = auto.tolist()
    out['preds'] = {m: pr.tolist() for m, pr in per_model.items()}
    out['selected'] = selections
    for m, pr in per_model.items():
        out['models'][m] = {'oof': metrics(y, pr), 'confusion_0.06': confusion(y, pr),
                            'strict': metrics(y[strong], pr[strong]), 'strict_confusion': confusion(y[strong], pr[strong])}
    out['models']['auto'] = {'oof': metrics(y, auto), 'confusion_0.06': confusion(y, auto),
                             'strict': metrics(y[strong], auto[strong]),
                             'strict_confusion': confusion(y[strong], auto[strong])}
    # paired bootstrap auto vs A
    rng = np.random.default_rng(42)
    uniq = sorted(set(groups.tolist())); idx_of = {g: np.where(groups == g)[0] for g in uniq}
    base = np.abs(y - per_model['A']); alt = np.abs(y - auto)
    diffs = []
    for _ in range(2000):
        pick = rng.choice(uniq, len(uniq), replace=True)
        sel = np.concatenate([idx_of[g] for g in pick])
        diffs.append(float(base[sel].mean() - alt[sel].mean()))
    out['bootstrap_auto_vs_A'] = {'mae_diff': float(base.mean() - alt.mean()),
                                  'ci95': [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                                  'passes_ci': bool(np.percentile(diffs, 2.5) > 0)}
    # per-model single-variable curves (descriptive, full fit)
    curves = {'features': FEATS}
    sc = StandardScaler().fit(X)
    rgB = Ridge(alpha=1.0).fit(StandardScaler().fit(design(X, 'B')).transform(design(X, 'B')), y)
    grid = {f: np.linspace(np.percentile(X[:, i], 5), np.percentile(X[:, i], 95), 25) for i, f in enumerate(FEATS)}
    med = np.median(X, axis=0)
    curve_data = {}
    for i, f in enumerate(FEATS):
        pts = []
        for v in grid[f]:
            x = med.copy(); x[i] = v
            pts.append({'x': float(v), 'pred_B': float(rgB.predict(
                StandardScaler().fit(design(X, 'B')).transform(design(x[None, :], 'B')))[0])})
        curve_data[f] = pts
    curves['B_partial_rows'] = curve_data
    # per-fold interaction coefficients from C
    inter = []
    for k in range(5):
        tr = fold != k
        Xd = design(X[tr], 'C')
        sc2 = StandardScaler().fit(Xd)
        rgc = Ridge(alpha=1.0).fit(sc2.transform(Xd), y[tr])
        inter.append({'fold': k, 'coef_pos_x_neg': float(rgc.coef_[6]), 'coef_er_x_collision': float(rgc.coef_[7])})
    curves['C_interaction_coefs'] = inter
    # D knots per fold
    curves['D_knots'] = [{'fold': s['fold'], 'best': s['best']} for s in selections]
    (P / 'MODELS281.json').write_text(json.dumps(out, indent=1) + '\n')
    (P / 'CURVES281.json').write_text(json.dumps({'schema': 'genrec.issue281.curves/1', **curves}, indent=1) + '\n')
    print(json.dumps({'n': len(y), 'selected': [(s['fold'], s['best'], {m: round(v, 6) for m, v in s['inner_mae'].items()}) for s in selections],
                      'oof': {m: {'mae': round(v['oof']['mae'], 6),
                                  'rho': None if v['oof']['spearman'] is None else round(v['oof']['spearman'], 3)}
                              for m, v in out['models'].items()},
                      'boot_auto_vs_A': out['bootstrap_auto_vs_A']}, indent=1))


if __name__ == '__main__':
    main()
