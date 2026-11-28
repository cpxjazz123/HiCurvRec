"""Deterministic checks for the SID-only directional decoder (CONTRACT.md section 2)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('d280', 'results/research/issue280/decoder280.py')
d280 = importlib.util.module_from_spec(spec); spec.loader.exec_module(d280)


def codes_of(rows):
    c = np.zeros((len(rows), 3), dtype=np.int64)
    for i, r in enumerate(rows):
        c[i] = r
    return c


def test_tables_hand_counts():
    # 3 items: 0=[1,1,1], 1=[1,2,2], 2=[2,2,1]; edges 0->1 (count 3), 0->2 (count 1)
    codes = codes_of([[1, 1, 1], [1, 2, 2], [2, 2, 1]])
    ea = np.array([0, 0]); eb = np.array([1, 2]); ew = np.array([3.0, 1.0])
    tabs = d280.build_tables(codes, ea, eb, ew)
    lv1 = tabs[0]
    # level-1 numerator key = (s1=1, t0=0, c1=1) with count 3
    key = (1 * 256 + 0) * 256 + 1
    val, hit = d280.lookup(lv1['ukn'], lv1['cn'], np.array([key]))
    assert hit[0] and abs(val[0] - 3.0) < 1e-12
    # level-2 numerator: (s2=1*256+1=257, t1=1, c2=2) count 3
    key2 = (257 * 256 + 1) * 256 + 2
    val2, hit2 = d280.lookup(tabs[1]['ukn'], tabs[1]['cn'], np.array([key2]))
    assert hit2[0] and abs(val2[0] - 3.0) < 1e-12


def test_direction_asymmetry():
    # 0=[1,1,1], 1=[1,2,2], 2=[2,3,3]; edges 0->1 (5) and a decoy 2->1 (1).
    # Forward 0->1 gets non-zero level-2/3 terms because the source prefix (1,1) has outgoing edges;
    # the reverse 1->0 falls back to the marginal on both sides of every level (source prefix (1,2)
    # has no outgoing edges) -> exactly 0. The test asserts the asymmetry, not its sign (the Laplace
    # mass makes the sign of a ratio with c = T context-dependent).
    codes = codes_of([[1, 1, 1], [1, 2, 2], [2, 3, 3]])
    ea = np.array([0, 2]); eb = np.array([1, 1]); ew = np.array([5.0, 1.0])
    tabs = d280.build_tables(codes, ea, eb, ew)
    fwd = d280.score_pairs(tabs, codes, np.array([0]), np.array([1]), 1.0, 1e-3)
    rev = d280.score_pairs(tabs, codes, np.array([1]), np.array([0]), 1.0, 1e-3)
    assert abs(fwd[0] - rev[0]) > 0.1  # both finite; the reverse lacks the source-prefix terms


def test_permutation_invariance():
    codes = codes_of([[1, 1, 1], [1, 2, 2], [2, 2, 1], [2, 1, 3]])
    ea = np.array([0, 0, 3, 2]); eb = np.array([1, 2, 1, 3]); ew = np.array([3.0, 1.0, 2.0, 4.0])
    A = np.array([0, 2, 3]); B = np.array([1, 3, 0])
    base_tabs = d280.build_tables(codes, ea, eb, ew)
    base = d280.score_pairs(base_tabs, codes, A, B, 1.0, 1e-3)
    perm = np.array([2, 0, 1])  # bijection on level codes
    codes_p = codes.copy()
    codes_p[:, 0] = perm[codes[:, 0]]
    tabs_p = d280.build_tables(codes_p, ea, eb, ew)
    # scores are computed with the permuted codebook: rebuild with the same permutation applied to B's codes
    perm_codes = codes_p
    permuted = d280.score_pairs(tabs_p, perm_codes, A, B, 1.0, 1e-3)
    assert np.allclose(base, permuted, atol=1e-12)


def test_zero_support_is_zero_and_finite():
    codes = codes_of([[1, 1, 1], [1, 2, 2]])
    ea = np.array([0]); eb = np.array([1]); ew = np.array([1.0])
    tabs = d280.build_tables(codes, ea, eb, ew)
    s = d280.score_pairs(tabs, codes, np.array([0]), np.array([1]), 1.0, 1e-3)
    assert np.isfinite(s).all() and abs(float(s[0])) < 1e-9  # only self-support -> likelihood ratio 1


if __name__ == '__main__':
    test_tables_hand_counts()
    test_direction_asymmetry()
    test_permutation_invariance()
    test_zero_support_is_zero_and_finite()
    print('all decoder checks passed')
