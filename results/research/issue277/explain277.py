"""Issue #277 — explain the frozen six-feature Ridge: per-fold coefficients, independent NumPy check,
correlation/collinearity, ablations, contributions, bootstrap CIs, TIGER reference audit, fold/duplicate
leak checks. Writes COEFFICIENTS.json and LEAK_AND_REFERENCE.json. Analysis-only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

P276 = Path('results/research/issue276')
P = Path('results/research/issue277')
FEATS = ['l1_pos_hit', 'l1_neg_hit', 'l1_hard_er', 'l2_distinct', 'l3_collision', 'l3_refine_bits']
FEATS8 = ['l1_pos_hit', 'l1_neg_hit', 'l1_hard_er', 'l2_pos_hit', 'l2_neg_hit',
          'l2_distinct', 'l3_collision', 'l3_refine_bits']
ALPHAS = (0.01, 0.1, 1.0, 10.0)


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = {'mae': float(np.mean(np.abs(y - p))), 'rmse': float(np.sqrt(np.mean((y - p) ** 2)))}
    m['spearman'] = float(stats.spearmanr(y, p).correlation) if len(y) > 2 and np.std(p) > 0 else None
    m['kendall'] = float(stats.kendalltau(y, p).correlation) if len(y) > 2 and np.std(p) > 0 else None
    return m


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


def main() -> None:
    table = json.load(open(P276 / 'SID_TABLE.json'))
    folds = json.load(open(P276 / 'FOLDS.json'))
    used = []
    for g in table['groups']:
        cand = [r for r in g['rows']
                if all(isinstance((r.get('features') or {}).get(k), (int, float)) for k in FEATS8)]
        labels = [r['label_recall@10'] for r in g['rows'] if isinstance(r['label_recall@10'], (int, float))]
        if not cand or not labels:
            continue
        r = cand[0]
        used.append({'group': g['group'], 'label': float(np.mean(labels)),
                     'x': {k: float(r['features'][k]) for k in FEATS},
                     'sid_sha256': r['sid_sha256'],
                     'arms': [a.get('arm') for a in r.get('arms', [])],
                     'identities': sorted({a.get('identity') for a in r.get('arms', []) if a.get('identity')})})
    X = np.array([[u['x'][k] for k in FEATS] for u in used])
    y = np.array([u['label'] for u in used])
    groups = np.array([u['group'] for u in used])
    fold = np.array([folds['assignment'][g] for g in groups])

    # ---------- per-fold coefficients ----------
    folds_out = []
    oof = np.zeros(len(y))
    for k in range(5):
        tr, te = fold != k, fold == k
        alpha = inner_alpha_fixed(X[tr], y[tr], fold[tr])
        sc = StandardScaler().fit(X[tr])
        rg = Ridge(alpha=alpha).fit(sc.transform(X[tr]), y[tr])
        coef_std = rg.coef_.tolist()
        coef_raw = (rg.coef_ / sc.scale_).tolist()
        intercept_raw = float(rg.intercept_ - sum(rg.coef_ * sc.mean_ / sc.scale_))
        pred_sk = rg.predict(sc.transform(X[te]))
        # independent NumPy recomputation of the same formula
        pred_np = rg.intercept_ + ((X[te] - sc.mean_) / sc.scale_) @ rg.coef_
        folds_out.append({'fold': k, 'n_train': int(tr.sum()), 'n_test': int(te.sum()),
                          'alpha': alpha, 'scaler_mean': sc.mean_.tolist(), 'scaler_scale': sc.scale_.tolist(),
                          'intercept_standardised': float(rg.intercept_),
                          'coef_standardised': coef_std, 'coef_raw_units': coef_raw,
                          'intercept_raw_units': intercept_raw,
                          'numpy_vs_sklearn_max_abs_diff': float(np.max(np.abs(pred_np - pred_sk)))})
        oof[te] = pred_sk
    coef_matrix = np.array([f['coef_standardised'] for f in folds_out])

    # ---------- correlation / collinearity ----------
    corr = np.corrcoef(X.T)
    # variance inflation factors from the inverse correlation matrix
    vif = np.diag(np.linalg.inv(corr)).tolist()

    # ---------- ablations (corrected scaler procedure, same folds) ----------
    def oof_for(feats):
        idx = [FEATS.index(f) for f in feats]
        p = np.zeros(len(y))
        for k in range(5):
            tr, te = fold != k, fold == k
            alpha = inner_alpha_fixed(X[tr][:, idx], y[tr], fold[tr])
            sc = StandardScaler().fit(X[tr][:, idx])
            rg = Ridge(alpha=alpha).fit(sc.transform(X[tr][:, idx]), y[tr])
            p[te] = rg.predict(sc.transform(X[te][:, idx]))
        return metrics(y, p), p

    ablations = {}
    full_m, full_p = oof_for(FEATS)
    ablations['full_6'] = full_m
    ablations['minus_er'] = oof_for([f for f in FEATS if f != 'l1_hard_er'])[0]
    ablations['minus_hit_rates'] = oof_for([f for f in FEATS if f not in ('l1_pos_hit', 'l1_neg_hit')])[0]
    ablations['er_only'] = oof_for(['l1_hard_er'])[0]
    for f in FEATS:
        ablations[f'single:{f}'] = oof_for([f])[0]
    ablations['mean'] = metrics(y, np.full(len(y), y.mean()))
    ablations['median'] = metrics(y, np.full(len(y), np.median(y)))

    # ---------- contributions with the frozen final predictor ----------
    pred_doc = json.load(open(P / 'PREDICTOR.json'))
    mean = np.array(pred_doc['scaler']['mean'])
    scale = np.array(pred_doc['scaler']['scale'])
    coef = np.array(pred_doc['coef_standardised'])
    intercept = pred_doc['fit_procedure']['ridge_intercept_standardised']
    Z = (X - mean) / scale
    contrib = Z * coef
    pred_final = intercept + contrib.sum(1)
    high = np.argsort(-y)[:13]
    low = np.argsort(y)[:5]
    def row(i):
        return {'sid_sha256': used[i]['sid_sha256'], 'arms': used[i]['arms'][:2],
                'y': float(y[i]), 'pred': float(pred_final[i]),
                'contributions': {f: float(contrib[i][j]) for j, f in enumerate(FEATS)},
                'intercept': float(intercept)}
    contrib_table = {'high_recall': [row(i) for i in high], 'low_recall': [row(i) for i in low]}

    # ---------- bootstrap CI of the OOF MAE and of the high-recall confusion ----------
    rng = np.random.default_rng(42)
    uniq = sorted(set(groups))
    idx_of = {g: np.where(groups == g)[0] for g in uniq}
    mae_boot, tp_boot, fp_boot = [], [], []
    for _ in range(2000):
        pick = rng.choice(uniq, len(uniq), replace=True)
        sel = np.concatenate([idx_of[g] for g in pick])
        mae_boot.append(float(np.mean(np.abs(y[sel] - full_p[sel]))))
        tp_boot.append(int(((full_p[sel] > 0.06) & (y[sel] > 0.06)).sum()))
        fp_boot.append(int(((full_p[sel] > 0.06) & (y[sel] <= 0.06)).sum()))
    ci = {'mae_q025': float(np.percentile(mae_boot, 2.5)), 'mae_q975': float(np.percentile(mae_boot, 97.5)),
          'tp_q025': int(np.percentile(tp_boot, 2.5)), 'tp_q975': int(np.percentile(tp_boot, 97.5)),
          'fp_q025': int(np.percentile(fp_boot, 2.5)), 'fp_q975': int(np.percentile(fp_boot, 97.5))}

    coef_out = {
        'schema': 'genrec.issue277.coefficients/1',
        'features': FEATS,
        'folds': folds_out,
        'coef_standardised_mean': coef_matrix.mean(0).tolist(),
        'coef_standardised_std': coef_matrix.std(0).tolist(),
        'correlation_matrix': corr.tolist(),
        'vif': vif,
        'oof_full_corrected': full_m,
        'ablations': ablations,
        'bootstrap_ci': ci,
        'contributions': contrib_table,
        'note': 'l1_pos_hit / l1_neg_hit / l1_hard_er are functionally dependent (ER = pos/neg); '
                'their separate coefficients are not independent causal effects',
    }
    (P / 'COEFFICIENTS.json').write_text(json.dumps(coef_out, indent=1) + '\n')

    # ---------- leak / reference audit ----------
    counts = {k: int((fold == k).sum()) for k in range(5)}
    dup_groups = len(used) - len(set(groups))
    # near-duplicates via the canonical mapping hash (only #276-computed features carry it)
    mapping = {}
    for u in used:
        for g in table['groups']:
            if g['group'] == u['group']:
                m = next((r.get('mapping_sha256') for r in g['rows'] if r.get('mapping_sha256')), None)
                if m:
                    mapping.setdefault(m, []).append(u['group'])
                break
    cross_fold = [m for m, gs in mapping.items() if len({folds['assignment'][g] for g in gs}) > 1]
    # TIGER reference identity
    tiger_rows = []
    for g in table['groups']:
        for r in g['rows']:
            arms = [a.get('arm') for a in r.get('arms', [])]
            if any('TIGER_RQ-VAE' == a for a in arms):
                tiger_rows.append({'sid_sha256': r['sid_sha256'], 'label': r['label_recall@10'],
                                   'strength': r.get('label_strength'),
                                   'sources': [s.get('source') for s in r.get('label_sources', [])]})
    leak = {
        'schema': 'genrec.issue277.leak_reference/1',
        'fold_counts': counts,
        'duplicate_groups': dup_groups,
        'n_groups_with_mapping': len(mapping),
        'mappings_spanning_folds': len(cross_fold),
        'cross_fold_mapping_examples': cross_fold[:5],
        'tiger_rows': tiger_rows,
        'note': 'global code permutation leaves every one of the six features unchanged; '
                'content-identical SIDs are one group by construction; a mapping-hash split can only '
                'occur when two exports share the first three codes but differ in content hash, '
                'which the table treats as separate groups',
    }
    (P / 'LEAK_AND_REFERENCE.json').write_text(json.dumps(leak, indent=1) + '\n')
    print(json.dumps({
        'folds': [{'fold': f['fold'], 'alpha': f['alpha'], 'coef_std': [round(c, 5) for c in f['coef_standardised']],
                   'numpy_diff': f['numpy_vs_sklearn_max_abs_diff']} for f in folds_out],
        'vif': [round(v, 2) for v in vif],
        'ablations': {k: {'mae': round(v['mae'], 6), 'rho': round(v['spearman'] or 0, 3)}
                      for k, v in ablations.items()},
        'ci': ci,
        'leak': {k: leak[k] for k in ('fold_counts', 'duplicate_groups', 'n_groups_with_mapping', 'mappings_spanning_folds')},
        'tiger_rows': tiger_rows,
    }, indent=1))


if __name__ == '__main__':
    main()
