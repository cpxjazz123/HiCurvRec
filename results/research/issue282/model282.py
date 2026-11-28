"""Issue #282 — baseline #277-corrected Ridge6 vs type-contrast increments (CONTRACT.md §4/§5)."""
from __future__ import annotations

import json
import itertools
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

P = Path('results/research/issue282')
FEATS = ['l1_pos_hit', 'l1_neg_hit', 'l1_hard_er', 'l2_distinct', 'l3_collision', 'l3_refine_bits']
FEATS8 = ['l1_pos_hit', 'l1_neg_hit', 'l1_hard_er', 'l2_pos_hit', 'l2_neg_hit',
          'l2_distinct', 'l3_collision', 'l3_refine_bits']
ALPHAS = (0.01, 0.1, 1.0, 10.0)


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    rev = tot = 0
    for i in range(len(y)):
        for k in range(i + 1, len(y)):
            if (y[i] - y[k]) * (p[i] - p[k]) < 0:
                rev += 1
            if y[i] != y[k]:
                tot += 1
    return {'mae': float(np.mean(np.abs(y - p))), 'rmse': float(np.sqrt(np.mean((y - p) ** 2))),
            'spearman': float(stats.spearmanr(y, p).correlation) if len(y) > 2 and np.std(p) > 0 else None,
            'kendall': float(stats.kendalltau(y, p).correlation) if len(y) > 2 and np.std(p) > 0 else None,
            'reversals': rev / tot if tot else None}


def confusion(y, p, cut=0.06):
    y, p = np.array(y, float), np.array(p, float)
    return {'positives': int((y > cut).sum()), 'tp': int(((p > cut) & (y > cut)).sum()),
            'fp': int(((p > cut) & (y <= cut)).sum()), 'fn': int(((p <= cut) & (y > cut)).sum()),
            'tn': int(((p <= cut) & (y <= cut)).sum())}


def inner_alpha_fixed(X, y, groups):
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


def oof(X, y, fold):
    pred = np.zeros(len(y))
    for k in range(5):
        tr, te = fold != k, fold == k
        a = inner_alpha_fixed(X[tr], y[tr], fold[tr])
        sc = StandardScaler().fit(X[tr])
        rg = Ridge(alpha=a).fit(sc.transform(X[tr]), y[tr])
        pred[te] = rg.predict(sc.transform(X[te]))
    return pred


def main() -> None:
    table = json.load(open('results/research/issue276/SID_TABLE.json'))
    folds = json.load(open('results/research/issue276/FOLDS.json'))
    tt = {r['sid_sha256']: r for r in json.load(open(P / 'TYPE_TABLE.json'))['rows']}
    used = []
    for g in table['groups']:
        cand = [r for r in g['rows'] if all(isinstance((r.get('features') or {}).get(k), (int, float)) for k in FEATS8)]
        labels = [r['label_recall@10'] for r in g['rows'] if isinstance(r['label_recall@10'], (int, float))]
        if not cand or not labels:
            continue
        r = cand[0]
        ident = sorted({a.get('identity') for a in r.get('arms', []) if a.get('identity')})
        srcs = [x.get('source') for x in r.get('label_sources', [])]
        strong = ('L1_ledger' in ident) or any(str(x).startswith('inventory_unit') for x in srcs)
        t = tt.get(r['sid_sha256'], {})
        used.append({'group': g['group'], 'y': float(np.mean(labels)), 'strong': strong,
                     'x': [float(r['features'][k]) for k in FEATS],
                     'f1': t.get('F1_merge_minus_chain'), 'f2': t.get('F2_fork_minus_chain'),
                     'support_ok': t.get('support_ok', False)})
    X = np.array([u['x'] for u in used]); y = np.array([u['y'] for u in used])
    groups = np.array([u['group'] for u in used]); fold = np.array([folds['assignment'][g] for g in groups])
    strong = np.array([u['strong'] for u in used])
    f1 = np.array([u['f1'] if u['f1'] is not None else np.nan for u in used], float)
    f2 = np.array([u['f2'] if u['f2'] is not None else np.nan for u in used], float)
    out = {'schema': 'genrec.issue282.model_comparison/1', 'n_rows': len(y), 'n_strict': int(strong.sum()),
           'n_high_full': int((y > 0.06).sum()), 'n_high_strict': int(((y > 0.06) & strong).sum()),
           'nan_counts': {'f1': int(np.isnan(f1).sum()), 'f2': int(np.isnan(f2).sum())},
           'models': {}}
    ok = np.isfinite(f1) & np.isfinite(f2)
    ys = y[ok]; Xs = X[ok]; fold_s = fold[ok]; strong_s = strong[ok]
    out['n_rows_with_both'] = int(ok.sum())
    A = oof(Xs, ys, fold_s)
    out['models']['A_277_corrected'] = {'oof': metrics(ys, A), 'confusion_0.06': confusion(ys, A),
                                        'strict': metrics(ys[strong_s], A[strong_s]),
                                        'strict_confusion': confusion(ys[strong_s], A[strong_s])}
    preds = {'A': A}
    for name, cols in (('D1_plus_F1', [f1[ok]]), ('D2_plus_F2', [f2[ok]]), ('D3_both', [f1[ok], f2[ok]])):
        Xd = np.column_stack([Xs] + cols)
        pr = oof(Xd, ys, fold_s)
        preds[name] = pr
        out['models'][name] = {'oof': metrics(ys, pr), 'confusion_0.06': confusion(ys, pr),
                               'strict': metrics(ys[strong_s], pr[strong_s]),
                               'strict_confusion': confusion(ys[strong_s], pr[strong_s])}
    rng = np.random.default_rng(42)
    uniq = sorted(set(groups[ok].tolist())); idx_of = {g: np.where(groups[ok] == g)[0] for g in uniq}
    base = np.abs(ys - A)
    boot = {}
    for name in ('D1_plus_F1', 'D2_plus_F2', 'D3_both'):
        alt = np.abs(ys - preds[name]); diffs = []
        for _ in range(2000):
            pick = rng.choice(uniq, len(uniq), replace=True)
            sel = np.concatenate([idx_of[g] for g in pick])
            diffs.append(float(base[sel].mean() - alt[sel].mean()))
        boot[name] = {'mae_diff_A_minus': float(base.mean() - alt.mean()),
                      'ci95': [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                      'passes_ci': bool(np.percentile(diffs, 2.5) > 0)}
    out['bootstrap_vs_A'] = boot
    resid = ys - A
    out['residual_association'] = {}
    tests = []
    for name, v in (('F1', f1[ok]), ('F2', f2[ok])):
        m = np.isfinite(v)
        rho = float(stats.spearmanr(v[m], resid[m]).correlation) if m.sum() > 10 else None
        out['residual_association'][f'resid_vs_{name}'] = rho
        tests.append(rho)
    # BH FDR over the two contrast tests (plus the type-level P/N/ER contrasts would inflate; keep 2)
    order = np.argsort([abs(t) if t is not None else 0 for t in tests])
    out['residual_association']['bh_fdr_2_tests'] = {'rhos': tests, 'note': 'descriptive; only two contrasts'}
    # similar-overall contrast: |d l1_pos| <= 0.01 and relative ER within 20%
    pos = Xs[:, 0]; er = Xs[:, 2]
    pairs = []
    for i, j in itertools.combinations(range(len(ys)), 2):
        if abs(pos[i] - pos[j]) <= 0.01 and er[i] > 0 and er[j] > 0 and abs(er[i] - er[j]) / max(er[i], er[j]) <= 0.20:
            df1 = (f1[ok][i] - f1[ok][j]) if np.isfinite(f1[ok][i]) and np.isfinite(f1[ok][j]) else None
            df2 = (f2[ok][i] - f2[ok][j]) if np.isfinite(f2[ok][i]) and np.isfinite(f2[ok][j]) else None
            pairs.append({'i': int(i), 'j': int(j), 'd_recall': float(ys[i] - ys[j]),
                          'd_F1': df1, 'd_F2': df2,
                          'l1_pos': [float(pos[i]), float(pos[j])], 'ER': [float(er[i]), float(er[j])]})
    out['similar_overall_pairs'] = {'n': len(pairs), 'pairs': pairs[:50]}
    d1 = [p['d_F1'] for p in pairs if p['d_F1'] is not None]
    dr = [p['d_recall'] for p in pairs if p['d_F1'] is not None]
    out['similar_overall_pairs']['spearman_dF1_vs_drecall'] = (
        float(stats.spearmanr(d1, dr).correlation) if len(d1) > 10 else None)
    d2 = [p['d_F2'] for p in pairs if p['d_F2'] is not None]
    dr2 = [p['d_recall'] for p in pairs if p['d_F2'] is not None]
    out['similar_overall_pairs']['spearman_dF2_vs_drecall'] = (
        float(stats.spearmanr(d2, dr2).correlation) if len(d2) > 10 else None)
    (P / 'MODEL_COMPARISON.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({'n': out['n_rows_with_both'], 'oof': {k: {'mae': round(v['oof']['mae'], 6),
                                                               'rho': None if v['oof']['spearman'] is None else round(v['oof']['spearman'], 3)}
                                                           for k, v in out['models'].items()},
                      'boot': boot, 'resid': out['residual_association'],
                      'pairs': {k: out['similar_overall_pairs'][k] for k in
                                ('n', 'spearman_dF1_vs_drecall', 'spearman_dF2_vs_drecall')}}, indent=1))


if __name__ == '__main__':
    main()
