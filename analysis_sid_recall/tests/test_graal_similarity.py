"""GRAAL's signature similarity, checked against the paper's own arithmetic.

Milenković & Pržulj define D_i = w_i |log2(u_i+1) - log2(v_i+1)| / log2(max(u_i,v_i)+2) and S = 1 - D.
For counts 1 and 0 with a unit weight that gives 1 - log2(2)/log2(3) = 0.3690..., which is the value the
review used to show that the earlier denominator was wrong; identical signatures must score exactly 1.
"""

import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "role_topology", HERE.parents[1] / "results" / "research" / "issue215" / "role_topology.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_zero_against_one_matches_the_paper():
    first = np.array([[1.0, 5.0]])
    second = np.array([[0.0, 5.0]])
    weights = np.array([1.0, 1.0])
    similarity = module.graal_similarity(first, second, weights)[0]
    expected = 1.0 - (np.log2(2.0) / np.log2(3.0)) / 2.0   # orbit 2 is identical and contributes 0
    assert abs(similarity - expected) < 1e-12
    single = module.graal_similarity(np.array([[1.0]]), np.array([[0.0]]), np.array([1.0]))[0]
    assert abs(single - (1.0 - np.log2(2.0) / np.log2(3.0))) < 1e-12


def test_identical_signatures_score_one():
    values = np.array([[3.0, 0.0, 7.0]])
    assert abs(module.graal_similarity(values, values, np.ones(3))[0] - 1.0) < 1e-12


if __name__ == "__main__":
    test_zero_against_one_matches_the_paper()
    test_identical_signatures_score_one()
    print("GRAAL similarity tests pass")
