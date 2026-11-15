"""The batched scorer must reproduce the scalar one exactly.

The scalar path (``score_arm`` + a distance callable) is the definition; the batched path only exists
because a norm per pair costs far more in Python calls than in floating point.  These cases put both
paths on small synthetic instances whose distances are known, so a disagreement points at the batching
rather than at the data.
"""

import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("analysis_under_test", HERE.parent / "analysis.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def line_vectors(nodes):
    """One-dimensional embeddings: distance between item i and j equals |i - j|."""
    return np.array([[float(node)] for node in nodes])


def build(pattern, nodes, groups, draws):
    instance = {"nodes": tuple(nodes), **groups}
    return {pattern: [instance]}, {pattern: [list(draws)]}


def compare(pattern, nodes, groups, draws):
    patterns, nulls = build(pattern, nodes, groups, draws)
    vectors = line_vectors(range(0, 40))
    distance = lambda left, right: float(abs(vectors[left, 0] - vectors[right, 0]))
    slow = study.score_arm(patterns, nulls, {}, distance, 40)
    fast = study.score_arm_vectors(patterns, nulls, vectors, 40)
    for slow_family, fast_family in zip(slow, fast):
        for name in patterns:
            for side in (0, 1):
                a = np.asarray(slow_family[name][side], dtype=float)
                b = np.asarray(fast_family[name][side], dtype=float)
                assert np.allclose(a, b, equal_nan=True), (pattern, name, side, a, b)


def test_k5_matches():
    compare("K5", (0, 1, 2, 3, 4), {}, [[20, 21, 22, 23, 24], [25, 26, 27, 28, 29],
                                        [30, 31, 32, 33, 34]])


def test_near_far_matches():
    compare("K4_tail", (0, 1, 2, 10),
            {"core": [(0, 1), (0, 2), (1, 2)], "tail": [(3, 0), (3, 1), (3, 2)]},
            [[20, 21, 22, 30], [25, 26, 27, 33], [11, 12, 13, 31]])


def test_p5_matches():
    compare("P5_hops", (0, 1, 5, 9, 14),
            {"D1": [(0, 1)], "D2": [(0, 2)], "D3": [(0, 3)], "D4": [(0, 4)]},
            [[20, 22, 25, 28, 33], [21, 24, 27, 30, 35], [10, 14, 18, 22, 26]])


if __name__ == "__main__":
    test_k5_matches()
    test_near_far_matches()
    test_p5_matches()
    print("batched scoring matches the scalar path")
