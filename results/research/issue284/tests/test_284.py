"""Deterministic hand-computed tests for Issue #284 (ALL bounds, conditional refinement,
raw-unit coefficient conversion, matching predicate, chain-feature identities)."""
import sys

import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'results/research/issue284')
import common284 as C


def test_all_bounds_hand():
    assert C.all_bounds(0.5, 0.5, 0.5) == (0.25, 0.5)
    assert C.all_bounds(0.1, 0.1, 0.1) == (0.0, 0.1)
    assert C.all_bounds(0.6, 0.2, 0.1) == (0.0, 0.1)
    lo, hi = C.all_bounds(0.5, 0.5, 0.5)
    assert abs((0.375 - lo) / (hi - lo) - 0.5) < 1e-12
    assert abs((0.5 - lo) / (hi - lo) - 1.0) < 1e-12


def test_conditional_hand():
    tr2 = np.zeros((5, 5))
    tr2[0] = [2, 1, 1, 0, 0]      # L1 ALL -> L2: 2 ALL, 1 AM, 1 MB
    tr3 = np.zeros((5, 5))
    tr3[0] = [1, 0, 0, 0, 1]      # from L2 ALL: 1 ALL, 1 NONE (n=2)
    tr3[1] = [0, 2, 0, 0, 0]      # from L2 AM: 2 AM (n=2)
    tr3[2] = [0, 0, 1, 0, 1]      # from L2 MB: 1 MB, 1 NONE (n=2)
    out = C.conditional_from_transitions(tr2, tr3)
    assert abs(out['l2_all_given_l1_all'] - 0.5) < 1e-12
    assert abs(out['cond_refine'] - 0.5) < 1e-12
    # p3_split = (2*0.5 + 1*0 + 1*0.5)/4
    assert abs(out['p3_split_given_l1_all'] - 0.375) < 1e-12
    # l3_all = (2*0.5 + 1*0 + 1*0)/4 = 0.25
    assert abs(out['l3_all_given_l1_all'] - 0.25) < 1e-12


def test_raw_unit_conversion():
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 7)); y = X @ np.arange(7) * 0.01 + 0.05
    sc = StandardScaler().fit(X)
    rg = Ridge(alpha=0.1).fit(sc.transform(X), y)
    coef = rg.coef_ / sc.scale_
    intercept = float(rg.intercept_ - (coef * sc.mean_).sum())
    assert np.allclose(rg.predict(sc.transform(X)), intercept + X @ coef, atol=1e-12)


def test_matching_predicate():
    import match284 as M
    rows = [{'x8': {k: v for k, v in zip(M.TOL, [0.1, 5.0, 0.01, 0.05, 2.0, 200, 3.0])}},
            {'x8': {k: v for k, v in zip(M.TOL, [0.105, 5.2, 0.0105, 0.051, 2.1, 210, 3.1])}}]
    ch = [{'p_am': 0.05, 'p_mb': 0.05, 'p_ab': 0.02},
          {'p_am': 0.054, 'p_mb': 0.051, 'p_ab': 0.021}]
    assert M.matches(rows, ch, marginal=True) == [(0, 1)]
    rows[1]['x8']['l1_pos_hit'] = 0.2
    assert M.matches(rows, ch, marginal=True) == []


def test_chain_feature_identities():
    rng = np.random.default_rng(1)
    codes = rng.integers(0, 256, size=(C.__dict__.get('N', 24587), 3)).astype(np.int16)
    f = C.chain_features(codes)
    S = f['n_chain']
    assert S > 0
    assert 0.0 <= f['all1_chain'] <= 1.0
    assert f['all_lo'] <= f['all1_chain'] + 1e-12
    assert f['all1_chain'] <= f['all_hi'] + 1e-12
    assert 0.0 <= f['all_norm'] <= 1.0
    assert 0.0 <= f['cond_refine'] <= 1.0
    assert 0.0 <= f['p3_split_given_l1_all'] <= 1.0
    assert 0.0 <= f['l1_entropy'] <= 8.0


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn(); print('ok', name)
