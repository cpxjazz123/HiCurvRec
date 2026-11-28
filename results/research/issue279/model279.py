"""Issue #279 — pre-registered model comparison (CONTRACT.md §5/§6)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

P = Path('results/research/issue279')
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
    sf = {r['sid_sha256']: r for r in json.load(open(P / 'SID_FEATURES.json'))['rows']}
    w278 = {r['sid_sha256']: r for r in json.load(open('results/research/issue278/WEIGHTED_TABLE.json'))['rows']}
    used = []
    for g in t['groups']:
        cand = [r for r in g['rows'] if all(isinstance((r.get('features') or {}).get(k), (int, float)) for k in FEATS)]
        labels = [r['label_recall@10'] for r in g['rows'] if isinstance(r['label_recall@10'], (int, float))]
        if not cand or not labels:
            continue
        r = cand[0]; sha = r['sid_sha256']
        s = sf.get(sha); w = w278.get(sha)
        if not s or not w:
            continue
        ident = sorted({a.get('identity') for a in r.get('arms', []) if a.get('identity')})
        srcs = [x.get('source') for x in r.get('label_sources', [])]
        strong = ('L1_ledger' in ident) or any(str(x).startswith('inventory_unit') for x in srcs)
        used.append({'group': g['group'], 'y': float(np.mean(labels)), 'strong': strong,
                     'sand': [float(r['features'][k]) for k in FEATS],
                     'er_unseen_2hop': s['L1']['er_unseen_2hop'],
                     'dp_unseen_degmatch': s['L1']['dp_unseen_degmatch'],
                     'P_w': w['L1']['all']['w_freq']['P_w'], 'N_w': w['L1']['all']['w_freq']['N_w']})
    y = np.array([u['y'] for u in used]); groups = np.array([u['group'] for u in used])
    fold = np.array([folds['assignment'][g] for g in groups]); strong = np.array([u['strong'] for u in used])
    Xa = np.array([u['sand'] for u in used])
    nan = {k: int(sum(1 for u in used if u[k] is None)) for k in ('er_unseen_2hop', 'dp_unseen_degmatch')}
    def col(k):
        v = np.array([u[k] if u[k] is not None else np.nan for u in used], float)
        return v
    c1, c2 = col('er_unseen_2hop'), col('dp_unseen_degmatch')
    pw, nw = col('P_w'), col('N_w')
    ok = np.isfinite(c1) & np.isfinite(c2) & np.isfinite(pw) & np.isfinite(nw)
    cands = {
        'A_control': (Xa, ['6 sand']),
        'C1_unseen2hop': (np.column_stack([Xa, c1]), FEATS + ['er_unseen_2hop']),
        'C2_degmatch_diff': (np.column_stack([Xa, c2]), FEATS + ['dp_unseen_degmatch']),
        'C3_both': (np.column_stack([Xa, c1, c2]), FEATS + ['er_unseen_2hop', 'dp_unseen_degmatch']),
        'E278_freq': (np.column_stack([Xa, pw, nw]), FEATS + ['P_w', 'N_w']),
    }
    out = {'schema': 'genrec.issue279.model_comparison/1', 'n_rows': int(ok.sum()),
           'nan_counts': nan, 'n_strict': int((strong & ok).sum()),
           'n_high_strict': int(((y > 0.06) & strong & ok).sum()),
           'n_high_full': int(((y > 0.06) & ok).sum()), 'candidates': {}}
    yk = y[ok]; fold_k = fold[ok]; groups_k = groups[ok]; strict_k = strong[ok]
    preds = {}
    for name, (X, feats) in cands.items():
        pr, alphas = oof(X[ok], yk, fold_k)
        preds[name] = pr
        out['candidates'][name] = {'features': feats, 'alphas': alphas, 'oof': metrics(yk, pr),
                                   'confusion_0.06': confusion(yk, pr),
                                   'strict': metrics(yk[strict_k], pr[strict_k]),
                                   'strict_confusion': confusion(yk[strict_k], pr[strict_k])}
    for name in ('fold_mean', 'fold_median'):
        pr = np.full(len(yk), yk.mean() if name == 'fold_mean' else np.median(yk))
        out['candidates'][name] = {'oof': metrics(yk, pr), 'confusion_0.06': confusion(yk, pr),
                                   'strict': metrics(yk[strict_k], pr[strict_k]),
                                   'strict_confusion': confusion(yk[strict_k], pr[strict_k])}
    rng = np.random.default_rng(42)
    uniq = sorted(set(groups_k.tolist())); idx_of = {g: np.where(groups_k == g)[0] for g in uniq}
    base = np.abs(yk - preds['A_control'])
    boot = {}
    for name in ('C1_unseen2hop', 'C2_degmatch_diff', 'C3_both', 'E278_freq'):
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
    res_an = {}
    for k, v in (('er_unseen_2hop', c1[ok]), ('dp_unseen_degmatch', c2[ok]),
                 ('P_w_freq', pw[ok]), ('N_w_freq', nw[ok])):
        m = np.isfinite(v)
        res_an[f'spearman_resid_vs_{k}'] = float(stats.spearmanr(v[m], resid[m]).correlation) if m.sum() > 10 else None
    out['residual_analysis'] = res_an
    lo = yk <= np.quantile(yk, 1 / 3); hi = yk > np.quantile(yk, 2 / 3)
    out['residual_by_label_stratum'] = {name: {'low': float(np.mean(res[lo])), 'high': float(np.mean(res[hi])),
                                               'low_mae': float(np.mean(np.abs(res[lo]))),
                                               'high_mae': float(np.mean(np.abs(res[hi])))}
                                        for name, res in [('A_control', resid),
                                                          ('C1_unseen2hop', yk - preds['C1_unseen2hop']),
                                                          ('C3_both', yk - preds['C3_both'])] }
    (P / 'MODEL_COMPARISON.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({'n_rows': out['n_rows'], 'n_strict': out['n_strict'],
                      'oof': {k: {'mae': round(v['oof']['mae'], 6), 'rho': None if v['oof']['spearman'] is None else round(v['oof']['spearman'], 3)}
                              for k, v in out['candidates'].items()},
                      'boot': boot, 'residual': res_an}, indent=1))


if __name__ == '__main__':
    main()
