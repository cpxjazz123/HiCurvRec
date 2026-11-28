"""Known-answer checks for the higher-order pair scores (CONTRACT.md section 2)."""
import importlib.util
import numpy as np
import sys
from collections import defaultdict
from pathlib import Path

spec = importlib.util.spec_from_file_location('rl', 'results/research/issue279/rellevel279.py')
rl = importlib.util.module_from_spec(spec); spec.loader.exec_module(rl)


def graph(edges):
    c_from, c_to = defaultdict(dict), defaultdict(dict)
    for a, b, c in edges:
        c_from[a][b] = c
        c_to[b][a] = c
    succ = {k: set(v) for k, v in c_from.items()}
    pred = {k: set(v) for k, v in c_to.items()}
    return succ, pred, c_from, c_to


def test_two_hop_and_common_neighbours():
    succ, pred, cf, ct = graph([(1, 2, 2), (2, 3, 3), (1, 3, 1), (1, 4, 1), (5, 4, 1)])
    s = rl.pair_scores(1, 3, succ, pred, cf, ct)
    # two-hop 1 -> 2 -> 3 with counts 2*3 = 6; direct edge 1
    assert s['two_hop'] == 6.0 and s['direct'] == 1.0
    # succ(1)={2,3,4}, succ(3)={} -> no common successor; pred(1)={} pred(3)={1,2} -> no common pred
    assert s['common_succ'] == 0.0 and s['common_pred'] == 0.0
    s2 = rl.pair_scores(1, 5, succ, pred, cf, ct)
    # succ(1)={2,3,4}, succ(5)={4} -> common {4}; union {2,3,4} -> jaccard 1/3
    assert s2['common_succ'] == 1.0 and abs(s2['cs_jaccard'] - 1 / 3) < 1e-12
    s3 = rl.pair_scores(2, 3, succ, pred, cf, ct)
    # pred(2)={1}, pred(3)={1,2} -> common {1}; union {1,2} -> 1/2
    assert s3['common_pred'] == 1.0 and abs(s3['cp_jaccard'] - 0.5) < 1e-12


def test_norm_and_zero_edges():
    succ, pred, cf, ct = graph([(1, 2, 2), (2, 3, 3)])
    s = rl.pair_scores(1, 3, succ, pred, cf, ct)
    assert s['two_hop'] == 6.0
    assert abs(s['two_hop_norm'] - 6.0 / (np.sqrt(1 * 1) + 1.0)) < 1e-12
    s0 = rl.pair_scores(9, 10, succ, pred, cf, ct)
    assert s0['two_hop'] == 0.0 and s0['common_succ'] == 0.0 and s0['direct'] == 0.0


if __name__ == '__main__':
    import numpy as np
    test_two_hop_and_common_neighbours()
    test_norm_and_zero_edges()
    print('all higher-order checks passed')
