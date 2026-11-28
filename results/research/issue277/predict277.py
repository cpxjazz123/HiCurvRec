"""Research-only read-only predictor for the frozen six-feature SID Ridge (Issue #277).

    python3 results/research/issue277/predict277.py <feature_json> [<feature_json> ...]

The feature JSON is one file of the shape written by the frozen evaluator
(`schema: genrec.issue276.layer_features/1`, keys `catalogue`, `validation`), exactly the files in
`results/research/issue276/features/layer/`. Missing or non-finite inputs are rejected; values outside
the frozen training domain are reported as warnings, not clipped. No infra is modified; no training.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

P = Path(__file__).resolve().parent
PREDICTOR = json.loads((P / 'PREDICTOR.json').read_text())
FEATS = PREDICTOR['features']


def features_from(path: Path) -> dict:
    rec = json.loads(Path(path).read_text())
    cat, val = rec.get('catalogue') or {}, rec.get('validation') or {}
    values = {
        'l1_pos_hit': val.get('l1_positive_hit_rate'),
        'l1_neg_hit': val.get('l1_negative_hit_probability'),
        'l1_hard_er': val.get('l1_hard_er'),
        'l2_distinct': cat.get('distinct_l2_prefixes'),
        'l3_collision': cat.get('l3_collision'),
        'l3_refine_bits': cat.get('l3_refinement_gain_bits'),
    }
    missing = [k for k in FEATS if not isinstance(values[k], (int, float)) or not math.isfinite(values[k])]
    if missing:
        raise ValueError('missing or non-finite features: ' + ', '.join(missing))
    return values


def predict(values: dict) -> dict:
    missing = [k for k in FEATS if k not in values]
    if missing:
        raise ValueError('missing features: ' + ', '.join(missing))
    x = [float(values[k]) for k in FEATS]
    if not all(math.isfinite(v) for v in x):
        raise ValueError('non-finite feature value')
    mean, scale = PREDICTOR['scaler']['mean'], PREDICTOR['scaler']['scale']
    coef_std = PREDICTOR['coef_standardised']
    z = [(xi - m) / s for xi, m, s in zip(x, mean, scale)]
    by_std = PREDICTOR['fit_procedure']['ridge_intercept_standardised'] + sum(c * zi for c, zi in zip(coef_std, z))
    by_raw = PREDICTOR['intercept_raw_units'] + sum(c * xi for c, xi in zip(PREDICTOR['coef_raw_units'], x))
    if abs(by_std - by_raw) > 1e-9:
        raise AssertionError('raw-unit and standardised formulas disagree')
    dom = PREDICTOR['domain']
    outside = [k for k in FEATS if not (dom[k]['min'] <= values[k] <= dom[k]['max'])]
    q = PREDICTOR['uncertainty']['oof_residual_quantiles']
    return {'recall@10': by_std,
            'interval_q05_q95': [by_std + q['q05'], by_std + q['q95']],
            'outside_training_domain': outside,
            'warnings': ['out-of-domain: ' + ','.join(outside)] if outside else []}


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for arg in argv:
        values = features_from(Path(arg))
        out = predict(values)
        print(json.dumps({'file': arg, **out}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
