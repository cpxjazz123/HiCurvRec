"""Issue #276 — post-analysis descriptive extras (explicitly NOT pre-registered decisions):
threshold tables for every model, the fidelity-only ridge ablation, effective-rank bins, the
corrected recon correlation, and the largest OOF residuals. Reads the frozen predictions.
"""
from __future__ import annotations

import json

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

P = 'results/research/issue276'
FEATS_275 = ['l1_pos_hit', 'l1_neg_hit', 'l1_hard_er', 'l2_distinct', 'l3_collision', 'l3_refine_bits']
FEATS_L1 = ['l1_pos_hit', 'l1_neg_hit', 'l1_hard_er']


def threshold(y, p, cut=0.06):
    y, p = np.array(y), np.array(p)
    return {'positives': int((y > cut).sum()),
            'tp': int(((p > cut) & (y > cut)).sum()),
            'fp': int(((p > cut) & (y <= cut)).sum()),
            'fn': int(((p <= cut) & (y > cut)).sum()),
            'precision': (float(((p > cut) & (y > cut)).sum() / max(1, (p > cut).sum()))),
            'recall': float(((p > cut) & (y > cut)).sum() / max(1, (y > cut).sum()))}


def main() -> None:
    d = json.load(open(f'{P}/ANALYSIS_276.json'))
    t = json.load(open(f'{P}/SID_TABLE.json'))
    out = {}

    for q in ('Q_SID', 'Q_VEC'):
        if q not in d:
            continue
        y = d[q]['y']
        out[q] = {'threshold_0.06': {k: threshold(y, p) for k, p in d[q]['predictions'].items()},
                  'largest_residuals_ridge6': None}
        # largest OOF residuals of the ridge6 control (arm names via the group order)
        order = d[q]['order']
        pred = np.array(d[q]['predictions']['ridge6_275'])
        res = np.abs(pred - np.array(y))
        idx = np.argsort(-res)[:5]
        by_group = {}
        for g in t['groups']:
            arms = [a.get('arm') for r in g['rows'] for a in r.get('arms', [])]
            by_group[g['group']] = arms[:2]
        out[q]['largest_residuals_ridge6'] = [
            {'group': order[i][:12], 'arms': by_group.get(order[i]), 'y': y[i],
             'pred': float(pred[i]), 'abs_err': float(res[i])} for i in idx]

    # effective-rank bins (descriptive): recall by pr_z quartile band
    rows = [r for g in t['groups'] for r in g['rows']
            if r.get('vector_forward') and isinstance(r['label_recall@10'], (int, float))]
    prs = sorted((r['vector_forward']['pr_z'], r['label_recall@10']) for r in rows)
    out['pr_z_bins'] = [{'band': f'{lo:.1f}-{hi:.1f}', 'n': len(chunk),
                         'recall_mean': float(np.mean([c[1] for c in chunk]))}
                        for (lo, hi), chunk in [
                            ((0, 4), [x for x in prs if x[0] < 4]),
                            ((4, 6), [x for x in prs if 4 <= x[0] < 6]),
                            ((6, 8), [x for x in prs if 6 <= x[0] < 8]),
                            ((8, 12), [x for x in prs if 8 <= x[0] < 12]),
                            ((12, 1e9), [x for x in prs if x[0] >= 12])]]
    out['pr_z_spearman'] = float(stats.spearmanr([x[0] for x in prs], [x[1] for x in prs]).correlation)
    rec = [(r['vector_forward']['recon_mse'], r['label_recall@10']) for r in rows
           if isinstance(r['vector_forward'].get('recon_mse'), (int, float))]
    out['recon_mse_spearman'] = float(stats.spearmanr([x[0] for x in rec], [x[1] for x in rec]).correlation)
    hi = [x for x in rec if x[1] > 0.06]
    out['recon_mse_spearman_high'] = (float(stats.spearmanr([x[0] for x in hi], [x[1] for x in hi]).correlation)
                                      if len(hi) > 2 else None)
    out['n_recon'] = len(rec)
    out['n_recon_high'] = len(hi)

    # exploratory fidelity-only ridge ablation (3 L1 features) inside the frozen folds
    folds = json.load(open(f'{P}/FOLDS.json'))
    table_rows = [r for g in t['groups'] for r in g['rows']]
    data = []
    for g in t['groups']:
        cand = [r for r in g['rows']
                if all(isinstance((r.get('features') or {}).get(k), (int, float)) for k in FEATS_275)]
        labels = [r['label_recall@10'] for r in g['rows'] if isinstance(r['label_recall@10'], (int, float))]
        if cand and labels:
            data.append({'group': g['group'], 'y': float(np.mean(labels)),
                         'x': [(cand[0]['features'] or {}) for _ in [0]][0]})
    for feats in (FEATS_L1, FEATS_275):
        preds, ys = [], []
        for k in range(5):
            tr = [r for r in data if folds['assignment'].get(r['group']) != k]
            te = [r for r in data if folds['assignment'].get(r['group']) == k]
            if len(tr) < 8 or not te:
                continue
            Xtr = np.array([[r['x'][f] for f in feats] for r in tr])
            Xte = np.array([[r['x'][f] for f in feats] for r in te])
            ytr = np.array([r['y'] for r in tr])
            sc = StandardScaler().fit(Xtr)
            rg = Ridge(alpha=10.0).fit(sc.transform(Xtr), ytr)
            preds.extend(list(rg.predict(sc.transform(Xte))))
            ys.extend([r['y'] for r in te])
        preds, ys = np.array(preds), np.array(ys)
        out[f'ridge_{len(feats)}feat'] = {'mae': float(np.mean(np.abs(ys - preds))),
                                          'spearman': float(stats.spearmanr(ys, preds).correlation),
                                          'n': len(ys)}
    print(json.dumps(out, indent=1))
    json.dump({'schema': 'genrec.issue276.extras/1', 'note': 'post-analysis descriptive; not pre-registered',
               'results': out}, open(f'{P}/EXTRAS.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
