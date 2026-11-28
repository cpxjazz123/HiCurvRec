"""Issue #283: predictive acceptance (CONTRACT §7) — #277 baseline vs inner-selected joint features."""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '.')
from results.research.issue283 import joint283 as J

FEATS6 = ['l1_pos_hit', 'l1_neg_hit', 'l1_hard_er', 'l2_distinct', 'l3_collision', 'l3_refine_bits']
FEATS8 = ['l1_pos_hit', 'l1_neg_hit', 'l1_hard_er', 'l2_pos_hit', 'l2_neg_hit',
          'l2_distinct', 'l3_collision', 'l3_refine_bits']
CANDIDATES = ['all1_chain', 'all1_chain_lift', 'all3_chain_lift', 'am_mb_chain', 'all1_merge_lift']
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
            'spearman': float(stats.spearmanr(y, p).correlation) if np.std(p) > 0 else None,
            'kendall': float(stats.kendalltau(y, p).correlation) if np.std(p) > 0 else None,
            'reversals': rev / tot if tot else None,
            'bias_low': float(np.mean((y - p)[y <= np.median(y)])),
            'bias_high': float(np.mean((y - p)[y > np.median(y)]))}


def confusion(y, p, cut=0.06):
    y, p = np.array(y, float), np.array(p, float)
    return {'positives': int((y > cut).sum()), 'tp': int(((p > cut) & (y > cut)).sum()),
            'fp': int(((p > cut) & (y <= cut)).sum()), 'fn': int(((p <= cut) & (y > cut)).sum()),
            'tn': int(((p <= cut) & (y <= cut)).sum())}


def inner_alpha(X, y, groups):
    best, best_err = 1.0, np.inf
    for alpha in ALPHAS:
        errs = []
        for g in sorted(set(groups.tolist())):
            mask = groups != g
            if mask.sum() < 5:
                continue
            sc = StandardScaler().fit(X[mask])
            rg = Ridge(alpha=alpha).fit(sc.transform(X[mask]), y[mask])
            errs.append(np.mean(np.abs(rg.predict(sc.transform(X[~mask])) - y[~mask])))
        if errs and np.mean(errs) < best_err:
            best, best_err = alpha, float(np.mean(errs))
    return best


def oof(X, y, groups):
    pred = np.zeros(len(y))
    for g in sorted(set(groups.tolist())):
        tr, te = groups != g, groups == g
        a = inner_alpha(X[tr], y[tr], groups[tr])
        sc = StandardScaler().fit(X[tr])
        rg = Ridge(alpha=a).fit(sc.transform(X[tr]), y[tr])
        pred[te] = rg.predict(sc.transform(X[te]))
    return pred


def fit_predict(Xtr, ytr, Xte, groups_tr):
    a = inner_alpha(Xtr, ytr, groups_tr)
    sc = StandardScaler().fit(Xtr)
    rg = Ridge(alpha=a).fit(sc.transform(Xtr), ytr)
    return rg.predict(sc.transform(Xte)), a


def load():
    table = json.load(open('results/research/issue276/SID_TABLE.json'))
    folds = json.load(open('results/research/issue276/FOLDS.json'))
    join = json.load(open('results/research/issue283/JOIN_TABLE.json'))
    rows = []
    seen = set()
    for g in table['groups']:
        cand = [r for r in g['rows'] if all(isinstance((r.get('features') or {}).get(k), (int, float)) for k in FEATS8)]
        labels = [r['label_recall@10'] for r in g['rows'] if isinstance(r['label_recall@10'], (int, float))]
        if not cand or not labels:
            continue
        r = cand[0]
        sha = r['sid_sha256']
        if sha in seen:
            continue
        seen.add(sha)
        ident = sorted({a.get('identity') for a in r.get('arms', []) if a.get('identity')})
        srcs = [s.get('source') for s in r.get('label_sources', [])]
        feats = join['entries'][sha]
        f = {}
        for name, d in feats['classes'].items():
            for k in (1, 2, 3):
                lvl = d[str(k)]
                cnt = np.asarray(lvl['counts'], dtype=float); S = cnt.sum()
                f[f'all{k}_{name}'] = float(cnt[0] / S)
                f[f'all{k}_{name}_lift'] = float(lvl['lift']['all_u'])
        chain_counts = np.asarray(feats['classes']['chain']['1']['counts'], dtype=float)
        f['am_mb_chain'] = float((chain_counts[1] - chain_counts[2]) / chain_counts.sum())
        rows.append({'sha': sha, 'y': float(np.mean(labels)),
                     'x6': [float(r['features'][k]) for k in FEATS6],
                     'joint': {k: float(f[k]) for k in CANDIDATES},
                     'strong': ('L1_ledger' in ident) or any(str(s).startswith('inventory_unit') for s in srcs),
                     'group': g['group'],
                     'labels': [float(x) for x in labels]})
    return rows, folds


def main() -> None:
    rows, folds = load()
    y = np.array([r['y'] for r in rows])
    fold = np.array([folds['assignment'][r['group']] for r in rows])
    X6 = np.array([r['x6'] for r in rows])
    strong = np.array([r['strong'] for r in rows])
    A = oof(X6, y, fold)
    base_m = metrics(y, A)
    out = {'schema': 'genrec.issue283.model_comparison/1', 'n_rows': len(rows),
           'n_strong': int(strong.sum()), 'n_high': int((y > 0.06).sum()),
           'n_high_strong': int(((y > 0.06) & strong).sum()),
           'n_above_0_065': int((y > 0.065).sum()),
           'baseline_A': {'oof': base_m, 'confusion_0.06': confusion(y, A),
                          'strict': metrics(y[strong], A[strong]),
                          'strict_confusion': confusion(y[strong], A[strong])},
           'candidates': {}, 'selection': {}, 'fixed_arms': {}}
    assert abs(base_m['mae'] - 0.002972) < 2e-6, base_m
    assert abs(base_m['spearman'] - 0.667) < 5e-3, base_m
    # inner-fold selection per outer fold
    new_pred = np.zeros(len(rows))
    picks = []
    for g in sorted(set(fold.tolist())):
        tr = fold != g; te = fold == g
        inner_groups = fold[tr]
        inner_base = oof(X6[tr], y[tr], inner_groups)
        inner_base_mae = float(np.mean(np.abs(y[tr] - inner_base)))
        scores = {}
        for cand in CANDIDATES:
            Xc = np.column_stack([X6[tr], np.array([r['joint'][cand] for r in rows])[tr]])
            p = oof(Xc, y[tr], inner_groups)
            scores[cand] = float(np.mean(np.abs(y[tr] - p)))
        best = min(scores, key=scores.get)
        choice = best if scores[best] < inner_base_mae else None
        picks.append({'fold': int(g), 'inner_base_mae': round(inner_base_mae, 7),
                      'scores': {k: round(v, 7) for k, v in scores.items()},
                      'chosen': choice})
        if choice is None:
            pred, alpha = fit_predict(X6[tr], y[tr], X6[te], fold[tr])
        else:
            Xtr = np.column_stack([X6, np.array([r['joint'][choice] for r in rows])])
            pred, alpha = fit_predict(Xtr[tr], y[tr], Xtr[te], fold[tr])
        new_pred[te] = pred
    new_m = metrics(y, new_pred)
    rng = np.random.default_rng(42)
    # cluster bootstrap over near-duplicate label clusters
    table = json.load(open('results/research/issue276/SID_TABLE.json'))
    rowmap = {r0['sid_sha256']: r0 for g in table['groups'] for r0 in g['rows']}
    clusters = {}
    for i, r in enumerate(rows):
        sigk = tuple(sorted((str(s.get('source')), round(float(s.get('recall', 0)), 6))
                            for s in rowmap[r['sha']].get('label_sources', [])))
        clusters.setdefault(sigk, []).append(i)
    uniq = list(clusters.values())
    diffs = []
    for _ in range(2000):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([uniq[int(k)] for k in pick])
        diffs.append(float(np.mean(np.abs(y[idx] - A[idx])) - np.mean(np.abs(y[idx] - new_pred[idx]))))
    out['selection'] = {'picks': picks,
                        'chosen_folds': {str(p['fold']): p['chosen'] for p in picks}}
    out['new_model'] = {'oof': new_m, 'confusion_0.06': confusion(y, new_pred),
                        'strict': metrics(y[strong], new_pred[strong]),
                        'strict_confusion': confusion(y[strong], new_pred[strong]),
                        'mae_diff_A_minus': round(float(np.mean(np.abs(y - A)) - np.mean(np.abs(y - new_pred))), 8),
                        'ci95': [round(float(np.percentile(diffs, 2.5)), 8),
                                 round(float(np.percentile(diffs, 97.5)), 8)],
                        'passes_ci': bool(np.percentile(diffs, 2.5) > 0)}
    out['passes'] = bool(out['new_model']['passes_ci']
                         and new_m['spearman'] is not None and base_m['spearman'] is not None
                         and new_m['spearman'] >= base_m['spearman']
                         and new_m['reversals'] <= base_m['reversals']
                         and confusion(y, new_pred)['fp'] <= confusion(y, A)['fp']
                         and confusion(y, new_pred)['fn'] <= confusion(y, A)['fn'])
    # fixed pre-registered arms (no selection) for ablation
    for cand in CANDIDATES:
        Xc = np.column_stack([X6, np.array([r['joint'][cand] for r in rows])])
        p = oof(Xc, y, fold)
        out['fixed_arms'][cand] = {'oof': metrics(y, p), 'confusion_0.06': confusion(y, p)}
        diffs = []
        for _ in range(2000):
            pick = rng.integers(0, len(uniq), len(uniq))
            idx = np.concatenate([uniq[int(k)] for k in pick])
            diffs.append(float(np.mean(np.abs(y[idx] - A[idx])) - np.mean(np.abs(y[idx] - p[idx]))))
        out['fixed_arms'][cand]['mae_diff'] = round(float(np.mean(np.abs(y - A)) - np.mean(np.abs(y - p))), 8)
        out['fixed_arms'][cand]['ci95'] = [round(float(np.percentile(diffs, 2.5)), 8),
                                           round(float(np.percentile(diffs, 97.5)), 8)]
    # #282 type features as analysis comparison (reproduce their OOF)
    tt = {r['sid_sha256']: r for r in json.load(open('results/research/issue282/TYPE_TABLE.json')).get('rows', [])}
    f1 = np.array([tt[r['sha']].get('F1_merge_minus_chain') if r['sha'] in tt else np.nan for r in rows], float)
    f2 = np.array([tt[r['sha']].get('F2_fork_minus_chain') if r['sha'] in tt else np.nan for r in rows], float)
    ok = np.isfinite(f1) & np.isfinite(f2)
    if ok.sum() == len(rows):
        for name, cols in (('282_F1', [f1]), ('282_F2', [f2]), ('282_both', [f1, f2])):
            p = oof(np.column_stack([X6] + cols), y, fold)
            out.setdefault('comparison_282', {})[name] = {'oof': metrics(y, p), 'confusion_0.06': confusion(y, p)}
    (J.P / 'MODEL_COMPARISON.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({'baseline': base_m, 'new': new_m, 'picks': picks,
                      'ci': out['new_model']['ci95'], 'passes': out['passes']}, indent=1)[:3000])


if __name__ == '__main__':
    main()
