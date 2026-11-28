"""Issue #286: audit of #285/#277 artifacts, labels, folds, the #285 headline numbers and #277 domain."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, '.')
sys.path.insert(0, 'results/research/issue286')
sys.path.insert(0, 'results/research/issue285')
import common285 as W

P = Path('results/research/issue286')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    out = {'schema': 'genrec.issue286.audit/1'}
    files285 = ['REPORT.md', 'VERDICT.md', 'GRID_285.json', 'MODEL_285.json', 'AUDIT_285.json']
    files277 = ['PREDICTOR.json', 'AUDIT_277.json', 'REPORT.md', 'VERDICT.md', 'PROSPECTIVE_OUTCOME.json',
                'COEFFICIENTS.json', 'FEATURE_CONTRACT.md']
    out['p285_files'] = {f: sha(f'results/research/issue285/{f}') for f in files285}
    out['p277_files'] = {f: sha(f'results/research/issue277/{f}') for f in files277}
    out['commits'] = {'p285': 'a433172f2774c5d633f3798b2151f01683ad7ec2',
                      'p277': 'd1cb5cbc8bf1cee998ce8e7ca7fb0e8afdf16b0d'}
    d = W.load_table()
    y = d['y']; fold = d['fold']
    out['labels'] = {'n': len(y), 'n_strong': int(d['strong'].sum()), 'n_weak': int((~d['strong']).sum()),
                     'n_high': int((y > 0.06).sum()), 'n_high_strong': int(((y > 0.06) & d['strong']).sum()),
                     'n_units': len(set(d['unit'])),
                     'fold_sizes': {str(f): int((fold == f).sum()) for f in range(5)},
                     'label_min': round(float(y.min()), 6), 'label_max': round(float(y.max()), 6),
                     'n_above_0_065': int((y > 0.065).sum())}
    # reproduce the #277 baseline and 13/0/0
    A = W.C.oof(d['X6'], y, fold)
    base = W.C.metrics(y, A)
    conf = W.C.confusion(y, A)
    out['baseline_reproduction'] = {'mae': base['mae'], 'spearman': base['spearman'], 'confusion': conf,
                                    'expected': {'mae': 0.002971939445864855, 'spearman': 0.6666088991472278,
                                                 'tp': 13, 'fp': 0, 'fn': 0}}
    # #285 headline cells / reversals
    g = W.gates(d)
    cells = W.cell_labels(g['level_N'], g['level_C'])
    ll = np.where(cells == 'LL')[0]
    grid = json.load(open('results/research/issue285/GRID_285.json'))
    out['p285_cells'] = {'LL_n': int(len(ll)), 'LL_high': int((y[ll] > 0.06).sum()),
                         'LL_high_strong': int(((y[ll] > 0.06) & d['strong'][ll]).sum()),
                         'LL_P_spearman': round(float(stats.spearmanr(d['P'][ll], y[ll]).correlation), 4),
                         'grid_LL_P_spearman': grid['cells']['LL']['P']['spearman'],
                         'grid_LL_P_reversals': grid['cells']['LL']['P']['reversals'],
                         'grid_note': 'the 7.7% reversal rate and the 0.951 LL correlation must match'}
    # #277 predictor domain / residual interval / label aggregation
    pred = json.load(open('results/research/issue277/PREDICTOR.json'))
    out['p277_predictor'] = {
        'target': pred.get('target'), 'formula': pred.get('formula'),
        'alpha': pred.get('fit_procedure', {}).get('alpha'),
        'domain_feature_ranges': pred.get('domain'),
        'residual_interval_from_verdict': 'MAE 95% CI [0.002376, 0.003620]; FP=0 in 2000 bootstrap resamples',
        'label_range': '0.0260-0.0648 (0.065 NOT_EVALUABLE)',
        'note': 'the domain is the observed feature range of the 129-row training cohort'}
    # time-postordered labels (diagnostic only)
    po = json.load(open('results/research/issue277/PROSPECTIVE_OUTCOME.json'))
    out['time_postordered'] = [{'sid_sha256': r['sid_sha256'][:12], 'recall': r['recall@10'],
                                'pred277': r['predicted_recall@10'], 'status': r['status']}
                               for r in po['rows']]
    (P / 'AUDIT_286.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: out[k] for k in ('labels', 'baseline_reproduction', 'p285_cells',
                                          'time_postordered')}, indent=1)[:2500])


if __name__ == '__main__':
    main()
