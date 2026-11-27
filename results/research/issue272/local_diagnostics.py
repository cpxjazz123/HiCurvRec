"""Issue #272: variance/scale diagnostics for the frozen local-curvature predictions."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / 'results/research/issue272'
SHARDS = sorted((REPO / 'assets/shards').glob('shard_*.npz'))
ARMS = {
    'TIGER_RQ-VAE': REPO / 'results/research/issue271/assets/geometry/TIGER_RQ-VAE.npz',
    'TIGER_euclid_local_curve_supervision_rq': REPO / 'results/research/issue270/assets/geometry/TIGER_euclid_local_curve_supervision_rq.npz',
    'TIGER_calibrated_kappa_rq': REPO / 'results/research/issue270/assets/geometry/TIGER_calibrated_kappa_rq.npz',
    'TIGER_global_j_rq': REPO / 'results/research/issue271/assets/geometry/TIGER_global_j_rq.npz',
    'TIGER_active_j_rq': REPO / 'results/research/issue271/assets/geometry/TIGER_active_j_rq.npz',
}


def main():
    out = {'schema': 'genrec.issue272.local_diagnostics/1', 'arms': {}}
    for arm, path in ARMS.items():
        z = np.load(path)
        error = z['error'].astype(np.float64)
        centres = np.zeros(len(error), dtype=np.int64)
        targets = np.zeros(len(error), dtype=np.float64)
        cursor = 0
        for shard in SHARDS:
            with np.load(shard) as blob:
                valid = blob['valid']
                n = int(valid.sum())
                centre = blob['centre'][valid].astype(np.int64)
                target = blob['curvature'][valid].astype(np.float64)
            centres[cursor:cursor + n] = centre
            targets[cursor:cursor + n] = target
            cursor += n
        if cursor != len(error):
            raise ValueError('%s rows %d != %d' % (arm, cursor, len(error)))
        predicted = targets + error
        rng = np.random.default_rng(272)
        sample = np.sort(rng.choice(len(error), 400_000, replace=False))
        p, t = predicted[sample], targets[sample]
        var_p, var_t = float(p.var()), float(t.var())
        corr = float(np.corrcoef(p, t)[0, 1])
        slope = float(np.cov(p, t, bias=True)[0, 1] / var_t) if var_t > 0 else None
        within = []
        unique = np.unique(centres)
        rng2 = np.random.default_rng(273)
        for c in rng2.choice(unique, min(400, len(unique)), replace=False):
            mask = centres[sample] == c
            if mask.sum() >= 8:
                rho = spearmanr(predicted[sample][mask], targets[sample][mask]).statistic
                if np.isfinite(rho):
                    within.append(float(rho))
        out['arms'][arm] = {
            'geometry': str(path), 'rows': int(len(error)),
            'mean_error': float(error.mean()), 'mean_abs_error': float(np.abs(error).mean()),
            'var_pred': var_p, 'var_target': var_t, 'var_ratio': var_p / var_t if var_t else None,
            'pearson': corr, 'regression_slope': slope,
            'within_centre_spearman_mean': float(np.mean(within)) if within else None,
            'within_centre_spearman_n': len(within),
        }
        print(arm, 'var_ratio', round(out['arms'][arm]['var_ratio'], 4),
              'slope', None if slope is None else round(slope, 4),
              'pearson', round(corr, 4), 'within', round(out['arms'][arm]['within_centre_spearman_mean'], 4), flush=True)
    (OUT / 'local_diagnostics.json').write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")


if __name__ == '__main__':
    main()
