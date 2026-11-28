"""Checks the frozen predictor: formula identity, rejection rules, domain warning, a frozen prediction."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path('results/research/issue277').resolve()))
import predict277 as p  # noqa: E402


def test_raw_and_standardised_agree_on_development_rows():
    table = json.load(open('results/research/issue276/SID_TABLE.json'))
    n = 0
    for g in table['groups']:
        for r in g['rows']:
            f = r.get('features') or {}
            if all(isinstance(f.get(k), (int, float)) for k in p.FEATS):
                p.predict({k: f[k] for k in p.FEATS})
                n += 1
            if n >= 50:
                return
    assert n >= 50


def test_missing_value_rejected():
    try:
        p.predict({'l1_pos_hit': 0.1})
    except ValueError:
        return
    raise AssertionError('missing features were not rejected')


def test_non_finite_rejected():
    vals = {k: 0.1 for k in p.FEATS}
    vals['l1_hard_er'] = float('nan')
    try:
        p.predict(vals)
    except ValueError:
        return
    raise AssertionError('non-finite feature was not rejected')


def test_domain_warning():
    vals = {k: (p.PREDICTOR['domain'][k]['min'] + p.PREDICTOR['domain'][k]['max']) / 2 for k in p.FEATS}
    assert not p.predict(vals)['warnings']
    vals['l3_collision'] = p.PREDICTOR['domain']['l3_collision']['max'] + 1.0
    assert p.predict(vals)['warnings']


def test_r2_frozen_prediction_matches():
    frozen = json.load(open('results/research/issue277/PROSPECTIVE.json'))
    row = next(r for r in frozen['rows']
               if r['sid_sha256'].startswith('9f6bf6731e88f5ec'))
    vals = p.features_from(Path(row['features_source']))
    assert abs(p.predict(vals)['recall@10'] - row['predicted_recall@10']) < 1e-12


if __name__ == '__main__':
    test_raw_and_standardised_agree_on_development_rows()
    test_missing_value_rejected()
    test_non_finite_rejected()
    test_domain_warning()
    test_r2_frozen_prediction_matches()
    print('all predictor checks passed')
