"""Issue #280 — pre-registered SID-level comparison (CONTRACT.md §6)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

P = Path('results/research/issue280')
FEATS = ['l1_pos_hit', 'l1_neg_hit', 'l1_hard_er', 'l2_distinct', 'l3_collision', 'l3_refine_bits']
ALPHAS = (0.01, 0.1, 1.0, 10.0)


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


def inner_alpha(X, y, groups):
    best, best_err = 1.0, np.inf
    for alpha in ALPHAS:
        errs = []
        for g in sorted(set(groups)):
            m = groups != g
            if m.sum() < 5:
                continue
            sc = StandardScaler().fit(X[m])
            rg = Ridge(alpha=alpha).fit(sc.transform(X[m]), y[m])
            errs.append(np.mean(np.abs(rg.predict(sc.transform(X[~m])) - y[~m])))
        if errs and np.mean(errs) < best_err:
            best, best_err = alpha, float(np.mean(errs))
    return best


def oof(X, y, fold):
    pred = np.zeros(len(y)); alphas = []
    for k in range(5):
        tr, te = fold != k, fold == k
        a = inner_alpha(X[tr], y[tr], fold[tr])
        sc = StandardScaler().fit(X[tr])
        rg = Ridge(alpha=a).fit(sc.transform(X[tr]), y[tr])
        pred[te] = rg.predict(sc.transform(X[te]))
        alphas.append(a)
    return pred, alphas


def main() -> None:
    t = json.load(open('results/research/issue276/SID_TABLE.json'))
    folds = json.load(open('results/research/issue276/FOLDS.json'))
    dec = {r['sid_sha256']: r for r in json.load(open(P / 'SID_DECODER_FEATURES.json'))['rows']}
    s279 = {r['sid_sha256']: r for r in json.load(open('results/research/issue279/SID_FEATURES.json'))['rows']}
    used = []
    for g in t['groups']:
        cand = [r for r in g['rows'] if all(isinstance((r.get('features') or {}).get(k), (int, float)) for k in FEATS)]
        labels = [r['label_recall@10'] for r in g['rows'] if isinstance(r['label_recall@10'], (int, float))]
        if not cand or not labels:
            continue
        r = cand[0]; sha = r['sid_sha256']
        d = dec.get(sha); s9 = s279.get(sha)
        if not d or not s9:
            continue
        ident = sorted({a.get('identity') for a in r.get('arms', []) if a.get('identity')})
        srcs = [x.get('source') for x in r.get('label_sources', [])]
        strong = ('L1_ledger' in ident) or any(str(x).startswith('inventory_unit') for x in srcs)
        used.append({'group': g['group'], 'y': float(np.mean(labels)), 'strong': strong,
                     'sand': [float(r['features'][k]) for k in FEATS],
                     'dec_auc': d['dec_auc_unseen'], 'dec_margin': d['dec_margin_degmatched'],
                     'er279': s9['L1']['er_unseen_2hop'], 'dp279': s9['L1']['dp_unseen_degmatch']})
    y = np.array([u['y'] for u in used]); groups = np.array([u['group'] for u in used])
    fold = np.array([folds['assignment'][g] for g in groups]); strong = np.array([u['strong'] for u in used])
    Xa = np.array([u['sand'] for u in used])
    def col(k): return np.array([u[k] if u[k] is not None else np.nan for u in used], float)
    da, dm, er, dp = col('dec_auc'), col('dec_margin'), col('er279'), col('dp279')
    ok = np.isfinite(da) & np.isfinite(dm) & np.isfinite(er) & np.isfinite(dp)
    cands = {'A_control': (Xa, ['6 sand']),
             'D1_dec_auc': (np.column_stack([Xa, da]), FEATS + ['dec_auc_unseen']),
             'D2_dec_margin': (np.column_stack([Xa, dm]), FEATS + ['dec_margin_degmatched']),
             'D3_both': (np.column_stack([Xa, da, dm]), FEATS + ['dec_auc_unseen', 'dec_margin_degmatched']),
             'G279_sharing': (np.column_stack([Xa, er, dp]), FEATS + ['er_unseen_2hop', 'dp_unseen_degmatch'])}
    out = {'schema': 'genrec.issue280.model_comparison/1', 'n_rows': int(ok.sum()), 'n_strict': int((strong & ok).sum()),
           'n_high_strict': int(((y > 0.06) & strong & ok).sum()), 'n_high_full': int(((y > 0.06) & ok).sum()),
           'candidates': {}}
    yk = y[ok]; fold_k = fold[ok]; groups_k = groups[ok]; sk = strong[ok]
    preds = {}
    for name, (X, feats) in cands.items():
        pr, alphas = oof(X[ok], yk, fold_k)
        preds[name] = pr
        out['candidates'][name] = {'features': feats, 'alphas': alphas, 'oof': metrics(yk, pr),
                                   'confusion_0.06': confusion(yk, pr),
                                   'strict': metrics(yk[sk], pr[sk]), 'strict_confusion': confusion(yk[sk], pr[sk])}
    for name in ('fold_mean', 'fold_median'):
        pr = np.full(len(yk), yk.mean() if name == 'fold_mean' else np.median(yk))
        out['candidates'][name] = {'oof': metrics(yk, pr), 'confusion_0.06': confusion(yk, pr),
                                   'strict': metrics(yk[sk], pr[sk]), 'strict_confusion': confusion(yk[sk], pr[sk])}
    rng = np.random.default_rng(42)
    uniq = sorted(set(groups_k.tolist())); idx_of = {g: np.where(groups_k == g)[0] for g in uniq}
    base = np.abs(yk - preds['A_control'])
    boot = {}
    for name in ('D1_dec_auc', 'D2_dec_margin', 'D3_both', 'G279_sharing'):
        alt = np.abs(yk - preds[name]); diffs = []
        for _ in range(2000):
            pick = rng.choice(uniq, len(uniq), replace=True)
            sel = np.concatenate([idx_of[g] for g in pick])
            diffs.append(float(base[sel].mean() - alt[sel].mean()))
        boot[name] = {'mae_diff_A_minus_candidate': float(base.mean() - alt.mean()),
                      'ci95': [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                      'passes_ci': bool(np.percentile(diffs, 2.5) > 0)}
    out['bootstrap_vs_A'] = boot
    resid = yk - preds['A_control']
    out['residual_analysis'] = {}
    for k, v in (('dec_auc_unseen', da[ok]), ('dec_margin_degmatched', dm[ok]),
                 ('er279', er[ok]), ('dp279', dp[ok])):
        m = np.isfinite(v)
        out['residual_analysis'][f'spearman_resid_vs_{k}'] = float(stats.spearmanr(v[m], resid[m]).correlation) if m.sum() > 10 else None
    lo = yk <= np.quantile(yk, 1 / 3); hi = yk > np.quantile(yk, 2 / 3)
    out['residual_by_label_stratum'] = {name: {'low': float(np.mean(r[lo])), 'high': float(np.mean(r[hi])),
                                               'low_mae': float(np.mean(np.abs(r[lo]))),
                                               'high_mae': float(np.mean(np.abs(r[hi])))}
                                        for name, r in [('A_control', resid), ('D3_both', yk - preds['D3_both'])]}
    (P / 'MODEL_COMPARISON.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({'n_rows': out['n_rows'], 'oof': {k: {'mae': round(v['oof']['mae'], 6),
                                                          'rho': None if v['oof']['spearman'] is None else round(v['oof']['spearman'], 3)}
                                                      for k, v in out['candidates'].items()},
                      'boot': boot, 'residual': out['residual_analysis']}, indent=1))


if __name__ == '__main__':
    main()
