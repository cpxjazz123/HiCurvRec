"""Issue #281 — descriptive calibration/stratified-error extras + bootstrap CIs for the fixed groups."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

P = Path('results/research/issue281')
FEATS = ['l1_pos_hit', 'l1_neg_hit', 'l1_hard_er', 'l2_distinct', 'l3_collision', 'l3_refine_bits']


def main() -> None:
    d = json.load(open(P / 'MODELS281.json'))
    t = json.load(open('results/research/issue276/SID_TABLE.json'))
    folds = json.load(open('results/research/issue276/FOLDS.json'))
    used = []
    for g in t['groups']:
        cand = [r for r in g['rows'] if all(isinstance((r.get('features') or {}).get(k), (int, float)) for k in FEATS)]
        labels = [r['label_recall@10'] for r in g['rows'] if isinstance(r['label_recall@10'], (int, float))]
        if not cand or not labels:
            continue
        used.append({'group': g['group'], 'y': float(np.mean(labels))})
    y = np.array([u['y'] for u in used]); groups = np.array([u['group'] for u in used])
    preds = {m: np.array(v) for m, v in d['preds'].items()}
    preds['auto'] = np.array(d['auto_pred'])
    out = {'schema': 'genrec.issue281.extras/1', 'n': len(y), 'models': {}}
    lo = y <= np.quantile(y, 1 / 3); hi = y > np.quantile(y, 2 / 3)
    for m, p in preds.items():
        resid = y - p
        bins = np.quantile(p, np.linspace(0, 1, 6))
        cal = []
        for i in range(5):
            mask = (p >= bins[i]) & (p <= bins[i + 1])
            if mask.sum():
                cal.append({'pred_mean': float(p[mask].mean()), 'y_mean': float(y[mask].mean()),
                            'n': int(mask.sum())})
        out['models'][m] = {
            'signed_error_low': float(resid[lo].mean()), 'signed_error_high': float(resid[hi].mean()),
            'mae_low': float(np.abs(resid[lo]).mean()), 'mae_high': float(np.abs(resid[hi]).mean()),
            'residual_q05_q50_q95': [float(np.quantile(resid, 0.05)), float(np.quantile(resid, 0.5)),
                                     float(np.quantile(resid, 0.95))],
            'abs_residual_q90': float(np.quantile(np.abs(resid), 0.9)),
            'calibration_bins': cal,
        }
    rng = np.random.default_rng(42)
    uniq = sorted(set(groups.tolist())); idx_of = {g: np.where(groups == g)[0] for g in uniq}
    base = np.abs(y - preds['A'])
    boot = {}
    for m in ('B', 'C', 'D', 'F', 'auto'):
        alt = np.abs(y - preds[m]); diffs = []
        for _ in range(2000):
            pick = rng.choice(uniq, len(uniq), replace=True)
            sel = np.concatenate([idx_of[g] for g in pick])
            diffs.append(float(base[sel].mean() - alt[sel].mean()))
        boot[m] = {'mae_diff_A_minus': float(base.mean() - alt.mean()),
                   'ci95': [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}
    out['bootstrap_vs_A'] = boot
    (P / 'EXTRAS281.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({'models': {m: {'low': round(v['signed_error_low'], 5), 'high': round(v['signed_error_high'], 5),
                                    'q90': round(v['abs_residual_q90'], 5)} for m, v in out['models'].items()},
                      'boot': boot}, indent=1))


if __name__ == '__main__':
    main()
