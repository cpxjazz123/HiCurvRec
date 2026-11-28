"""Hand-computed checks for the pair-level directed type definitions (Issue #282, CONTRACT.md §1)."""
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

spec = importlib.util.spec_from_file_location('t282', 'results/research/issue282/types282.py')
t282 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t282)


def graph(edges):
    c_from, c_to = defaultdict(set), defaultdict(set)
    for a, b in edges:
        c_from[a].add(b); c_to[b].add(a)
    return c_from, c_to


def test_chain_revchain_merge_fork():
    # chain a=1 -> m=2 -> b=3
    # revchain b=3 -> m=4 -> a=1
    # merge a=1 -> m=5, b=3 -> m=5
    # fork m=6 -> a=1, m=6 -> b=3
    edges = [(1, 2), (2, 3), (3, 4), (4, 1), (1, 5), (3, 5), (6, 1), (6, 3)]
    cf, ct = graph(edges)
    d = t282.type_counts(1, 3, cf, ct)
    assert d['chain'] == 1, d
    assert d['revchain'] == 1, d
    assert d['merge'] == 1, d
    assert d['fork'] == 1, d


def test_multi_label_overlap():
    # the same pair can be chain+merge+fork simultaneously
    edges = [(1, 2), (2, 3), (1, 7), (3, 7), (8, 1), (8, 3)]
    cf, ct = graph(edges)
    d = t282.type_counts(1, 3, cf, ct)
    assert d['chain'] >= 1 and d['merge'] >= 1 and d['fork'] >= 1
    assert d['revchain'] == 0, d


def test_zero_support():
    cf, ct = graph([(1, 2)])
    d = t282.type_counts(9, 10, cf, ct)
    assert all(v == 0 for v in d.values())


if __name__ == '__main__':
    test_chain_revchain_merge_fork()
    test_multi_label_overlap()
    test_zero_support()
    print('all type checks passed')
