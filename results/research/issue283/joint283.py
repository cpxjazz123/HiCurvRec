"""Issue #283 core: role taxonomy, partitions, LCP, null expectations (CONTRACT.md)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path('.')
P = REPO / 'results/research/issue283'
ROWS = P / 'rows'
N = 24587
ORDERED_PAIRS = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


# ------------------------------------------------------------------ 6-bit codes
def bits_of(code: int) -> dict:
    """in1=a->m, in2=b->m, out1=m->a, out2=m->b, e12=a->b, e21=b->a (CONTRACT §2)."""
    return {'in1': (code >> 3) & 1, 'in2': (code >> 1) & 1, 'out1': (code >> 5) & 1,
            'out2': (code >> 4) & 1, 'e12': (code >> 2) & 1, 'e21': code & 1}


def _outer_suffix(e12: int, e21: int) -> str:
    return '' if (e12 == 0 and e21 == 0) else ('_e12' if (e12 and not e21) else
                                              ('_e21' if (e21 and not e12) else '_both'))


def classify_code(code: int) -> str:
    b = bits_of(code)
    in1, in2, out1, out2, e12, e21 = b['in1'], b['in2'], b['out1'], b['out2'], b['e12'], b['e21']
    if not ((in1 or out1) and (in2 or out2)):
        return 'infeasible'  # impossible under the undirected projection (CONTRACT §2)
    rec1, rec2 = in1 & out1, in2 & out2
    if rec1 and rec2:
        return 'recip_both' + _outer_suffix(e12, e21)
    if rec1 and not rec2:
        second = 'in' if in2 else 'out'
        return f'recip_a_{second}' + _outer_suffix(e12, e21)
    if rec2 and not rec1:
        second = 'in' if in1 else 'out'
        return f'recip_b_{second}' + _outer_suffix(e12, e21)
    if in1 and out2 and not in2 and not out1:
        if e12 and e21:
            return 'chain_both'
        if e12:
            return 'ff'
        if e21:
            return 'fb'
        return 'chain'
    if in2 and out1 and not in1 and not out2:
        if e12 and e21:
            return 'revchain_both'
        if e21:
            return 'ff_rev'
        if e12:
            return 'fb_rev'
        return 'revchain'
    if out1 and out2 and not in1 and not in2:
        return 'fork' + _outer_suffix(e12, e21)
    if in1 and in2 and not out1 and not out2:
        return 'merge' + _outer_suffix(e12, e21)
    return 'other'


def swap_ab_code(code: int) -> int:
    b = bits_of(code)
    # swap positions a and b
    return ((b['out2'] << 5) | (b['out1'] << 4) | (b['in2'] << 3) | (b['e21'] << 2) |
            (b['in1'] << 1) | b['e12'])


def _build_tables():
    from tools.graph_infra.graphlet import canonical_code, orbit_of_position
    names = sorted({classify_code(c) for c in range(64)} - {'infeasible'})
    if 'other' in names:
        raise AssertionError('taxonomy must cover all feasible codes')
    class_of = np.array([(names.index(classify_code(c)) if classify_code(c) != 'infeasible' else -1)
                         for c in range(64)], dtype=np.int8)
    canonical = np.array([canonical_code(c) for c in range(64)], dtype=np.int8)
    orbits = np.array([[orbit_of_position(c, pos) for pos in range(3)] for c in range(64)], dtype=np.int8)
    symmetric = np.zeros(len(names), dtype=bool)
    for ci, name in enumerate(names):
        codes = [c for c in range(64) if class_of[c] == ci]
        swapped = {classify_code(swap_ab_code(c)) for c in codes}
        # structurally symmetric: the class is closed under a<->b and the split rate is 1/1
        symmetric[ci] = (swapped == {name}) and all(
            classify_code(swap_ab_code(c)) == name for c in codes)
    family = []
    for name in names:
        if name.startswith('chain'):
            family.append('chain')
        elif name.startswith('revchain'):
            family.append('revchain')
        elif name in ('ff', 'fb', 'ff_rev', 'fb_rev'):
            family.append('triangle')
        elif name.startswith('fork') or name.startswith('merge'):
            family.append('wedge')
        elif name.startswith('recip'):
            family.append('reciprocal')
        else:
            family.append('other')
    return names, class_of, canonical, orbits, symmetric, family


CLASS_NAMES, CLASS_OF_CODE, CANONICAL_CODE, ORBIT_OF_CODE, SYMMETRIC, FAMILY = _build_tables()
CLASS_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
SYMMETRIC_IDS = {i for i, s in enumerate(SYMMETRIC) if s}
assert len(CLASS_NAMES) == len(set(CLASS_NAMES))


# ------------------------------------------------------------------ partitions
def eq_bits(ca, cm, cb):
    """Cumulative-prefix equality of the three role codewords at one level."""
    return (ca == cm), (cm == cb), (ca == cb)


def pattern_of(eq_am, eq_mb, eq_ab):
    """0 ALL, 1 AM, 2 MB, 3 AB, 4 NONE; transitive equality guaranteed by the prefixes."""
    out = np.full(len(eq_am), 4, dtype=np.int8)
    only_ab = eq_ab & ~eq_am & ~eq_mb
    only_mb = eq_mb & ~eq_am & ~eq_ab
    only_am = eq_am & ~eq_mb & ~eq_ab
    all3 = eq_am & eq_mb
    out[only_ab] = 3
    out[only_mb] = 2
    out[only_am] = 1
    out[all3] = 0
    return out


PATTERN_NAMES = ('ALL', 'AM', 'MB', 'AB', 'NONE')


def lcp_of(eq_levels):
    """eq_levels: 3 boolean arrays (pair equality at k=1,2,3). Returns int8 LCP in 0..3."""
    a, b, c = eq_levels
    return (a.astype(np.int8) + (a & b).astype(np.int8) + (a & b & c).astype(np.int8))


def feasible_lcp_combo(l_am, l_mb, l_ab) -> bool:
    return (l_ab >= min(l_am, l_mb)) and (l_am >= min(l_ab, l_mb)) and (l_mb >= min(l_am, l_ab))


FEASIBLE_LCP = sorted({(x, y, z) for x in range(4) for y in range(4) for z in range(4)
                       if feasible_lcp_combo(x, y, z)})


# ------------------------------------------------------------------ null expectations
def null_rates(group_sizes, total):
    """Uniform permutation expectation preserving group sizes (distinct items)."""
    g = np.asarray(group_sizes, dtype=np.float64)
    n = float(total)
    pair = float((g * (g - 1)).sum() / (n * (n - 1)))
    triple = float((g * (g - 1) * (g - 2)).sum() / (n * (n - 1) * (n - 2)))
    return {'pair': pair, 'triple': triple}


def codes_matrix(path):
    """24,587 x 3 int16 code matrix from one item_sids.json (prefix levels only)."""
    raw = json.loads(Path(path).read_text())
    out = np.zeros((N, 3), dtype=np.int16)
    for k, v in raw.items():
        i = int(k)
        out[i, 0] = int(v[0]); out[i, 1] = int(v[1]); out[i, 2] = int(v[2])
    return out
