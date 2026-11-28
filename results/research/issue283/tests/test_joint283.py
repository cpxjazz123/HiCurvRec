"""Hand-computed determinism tests for the Issue #283 joint definitions (CONTRACT §3/§4)."""
import sys
import numpy as np

sys.path.insert(0, '.')
from results.research.issue283 import joint283 as J


def _codes(rows):
    """rows: list of (ca, cm, cb) codeword triples at one level -> (ca,cm,cb) arrays."""
    arr = np.asarray(rows, dtype=np.int16)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def _counts(rows):
    ca, cm, cb = _codes(rows)
    pat = J.pattern_of(*J.eq_bits(ca, cm, cb))
    return np.bincount(pat, minlength=5)


def test_hand_example_equal_marginals():
    # D1: {ALL, ALL, NONE, NONE}; D2: {ALL, AM, MB, AB}
    d1 = [(0, 0, 0), (0, 0, 0), (0, 1, 2), (0, 1, 2)]
    d2 = [(0, 0, 0), (0, 0, 1), (1, 0, 0), (0, 1, 0)]
    c1, c2 = _counts(d1), _counts(d2)
    assert c1.tolist() == [2, 0, 0, 0, 2], c1
    assert c2.tolist() == [1, 1, 1, 1, 0], c2
    # pairwise rates ALL+AM+AB, ALL+MB+AB, ALL+AB are 0.5 in both
    for counts in (c1, c2):
        p = counts / counts.sum()
        p_am = p[0] + p[1]
        p_mb = p[0] + p[2]
        p_ab = p[0] + p[3]
        assert np.allclose([p_am, p_mb, p_ab], [0.5, 0.5, 0.5]), (p, p_am, p_mb, p_ab)
    # the non-marginal dof ALL differs exactly by 0.25
    assert abs(c1[0] / 4 - c2[0] / 4 - 0.25) < 1e-12


def test_partition_sums_and_refinement():
    rng = np.random.default_rng(0)
    seq = rng.integers(0, 4, size=(5000, 3, 3))  # rows, role, level -> cumulative prefixes
    levels = []
    cum = [np.ones(5000, dtype=bool) for _ in range(3)]
    for k in range(3):
        c = seq[:, :, k]
        eq = J.eq_bits(c[:, 0], c[:, 1], c[:, 2])
        cum = [cum[i] & eq[i] for i in range(3)]
        levels.append(J.pattern_of(*cum))
    for pat in levels:
        assert pat.min() >= 0 and pat.max() <= 4
    allowed = {0: {0, 1, 2, 3, 4}, 1: {1, 4}, 2: {2, 4}, 3: {3, 4}, 4: {4}}
    for a, b in zip(levels[0], levels[1]):
        assert int(b) in allowed[int(a)]
    for a, b in zip(levels[1], levels[2]):
        assert int(b) in allowed[int(a)]


def test_rename_invariance():
    rng = np.random.default_rng(1)
    rows = rng.integers(0, 8, size=(4000, 3))
    ca, cm, cb = _codes(rows)
    base = np.bincount(J.pattern_of(*J.eq_bits(ca, cm, cb)), minlength=5)
    for _ in range(5):
        perm = rng.permutation(8)
        ca2, cm2, cb2 = perm[ca], perm[cm], perm[cb]
        other = np.bincount(J.pattern_of(*J.eq_bits(ca2, cm2, cb2)), minlength=5)
        assert np.array_equal(base, other)


def test_lcp_and_feasibility():
    # a=m on 3 levels, m=b on 2 -> a=b on at least 2
    eq_am = np.array([True, True, True]); eq_mb = np.array([True, True, False])
    eq_ab = eq_am & eq_mb
    l_am = J.lcp_of([eq_am[0:1], eq_am[1:2], eq_am[2:3]])
    l_mb = J.lcp_of([eq_mb[0:1], eq_mb[1:2], eq_mb[2:3]])
    l_ab = J.lcp_of([eq_ab[0:1], eq_ab[1:2], eq_ab[2:3]])
    assert (int(l_am[0]), int(l_mb[0]), int(l_ab[0])) == (3, 2, 2)
    assert J.feasible_lcp_combo(3, 2, 2)
    assert not J.feasible_lcp_combo(3, 2, 1)
    assert len(J.FEASIBLE_LCP) == 22


def test_type_classification_hand():
    def code(**kw):
        b = {'in1': 0, 'in2': 0, 'out1': 0, 'out2': 0, 'e12': 0, 'e21': 0}
        b.update(kw)
        return ((b['out1'] << 5) | (b['out2'] << 4) | (b['in1'] << 3) |
                (b['e12'] << 2) | (b['in2'] << 1) | b['e21'])
    assert J.classify_code(code(in1=1, out2=1)) == 'chain'
    assert J.classify_code(code(in2=1, out1=1)) == 'revchain'
    assert J.classify_code(code(out1=1, out2=1)) == 'fork'
    assert J.classify_code(code(in1=1, in2=1)) == 'merge'
    assert J.classify_code(code(in1=1, out2=1, e12=1)) == 'ff'
    assert J.classify_code(code(in1=1, out2=1, e21=1)) == 'fb'
    assert J.classify_code(code(in1=1, out1=1, in2=1)) == 'recip_a_in'
    assert J.classify_code(code(in2=1, out2=1, out1=1)) == 'recip_b_out'
    assert J.classify_code(code(**{k: 1 for k in ('in1', 'in2', 'out1', 'out2', 'e12', 'e21')})) == 'recip_both_both'
    for c in range(64):
        assert J.classify_code(c) in J.CLASS_NAMES or J.classify_code(c) == 'infeasible'
    # symmetry flags
    assert J.SYMMETRIC[J.CLASS_ID['fork']] and J.SYMMETRIC[J.CLASS_ID['merge']]
    assert not J.SYMMETRIC[J.CLASS_ID['fork_e12']] and not J.SYMMETRIC[J.CLASS_ID['chain']]


def test_null_rates_hand():
    out = J.null_rates([3, 3], 6)
    assert abs(out['pair'] - 0.4) < 1e-12, out
    assert abs(out['triple'] - 0.1) < 1e-12, out


def test_canonical_table_matches_graphlet_module():
    from tools.graph_infra.graphlet import canonical_code
    for c in range(64):
        assert int(J.CANONICAL_CODE[c]) == canonical_code(c), c


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn(); print('ok', name)
