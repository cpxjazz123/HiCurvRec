import tempfile
from pathlib import Path

import numpy as np
from distances import auc_neg_farther_than_pos, euclidean, load_pairs, ordering_error_rate
from neighborhoods import build_successors, topk_composition
from residuals import norm_statistics, residual_trace_from_model


def test_residual_levels_entering_quantizer():
    import torch
    class Quantizer:
        def __init__(self, value):
            self.value = value
        def __call__(self, residual):
            return residual * 0 + self.value, None, None, None
    class RQ:
        codebook_num = 3
        vq_layers = [Quantizer(1), Quantizer(2), Quantizer(3)]
    class Model:
        encoder = staticmethod(lambda x: x)
        rq = RQ()
    x = np.full((2, 4), 10.0, dtype=np.float32)
    trace = residual_trace_from_model(Model(), x)
    assert np.array_equal(trace[0, :, 0], [10, 9, 7])
    residual = torch.as_tensor(x)
    for level, quantizer in enumerate(Model.rq.vq_layers):
        assert np.array_equal(trace[:, level, :], residual.numpy())
        residual = residual - quantizer(residual)[0]


def test_norm_statistics_finite_nonnegative():
    sample = np.array([[[3., 4.], [0., 2.], [0., 1.]]], dtype=np.float32)
    norms = np.linalg.norm(sample, axis=-1)
    assert np.isfinite(sample).all()
    assert norms[0, 0] >= norms[0, 1] >= norms[0, 2]
    for row in norm_statistics(sample):
        assert all(np.isfinite(value) and value >= 0 for value in row.values() if isinstance(value, (int, float)))


def test_pair_loader_returns_file_arrays_unchanged():
    source = {"anchors": np.array([3, 1], dtype=np.int64), "positives": np.array([4, 2], dtype=np.int64),
              "negatives": np.array([[5, 6], [7, 8]], dtype=np.int64)}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "pairs.npz"
        np.savez(path, **source)
        loaded = load_pairs(path)
        for key, value in source.items():
            assert np.array_equal(loaded[key], value)


def test_pair_protocol_positive_excluded_from_negatives():
    from config import PRIMARY_PAIRS
    pairs = load_pairs(PRIMARY_PAIRS)
    for positive, negatives in zip(pairs["positives"], pairs["negatives"]):
        assert positive not in negatives


def test_euclidean_and_rms_ranking():
    x = np.array([[0., 0.], [3., 4.], [6., 8.]])
    d = euclidean(x)
    assert np.allclose(d, d.T) and np.allclose(np.diag(d), 0)
    rms = np.sqrt(np.mean(x * x))
    normalized = euclidean(x / rms)
    assert np.array_equal(np.argsort(d[0]), np.argsort(normalized[0]))
    assert not np.allclose(d, normalized)


def test_auc_ordering_and_ties():
    pos = np.array([1., 1.])
    separated = np.array([[3., 3.], [3., 3.]])
    inverted = np.array([[0., 0.], [0., 0.]])
    ties = np.array([[1., 1.], [1., 1.]])
    assert auc_neg_farther_than_pos(pos, separated) == 1.0
    assert ordering_error_rate(pos, separated) == 0.0
    assert auc_neg_farther_than_pos(pos, inverted) == 0.0
    assert ordering_error_rate(pos, inverted) == 1.0
    assert auc_neg_farther_than_pos(pos, ties) == 0.5
    assert ordering_error_rate(pos, ties) == 0.5
    assert auc_neg_farther_than_pos(np.array([1, 2]), np.array([3, 4])) == 1.0
    assert auc_neg_farther_than_pos(np.array([3, 4]), np.array([1, 2])) == 0.0
    assert auc_neg_farther_than_pos(np.array([1, 1]), np.array([1, 1])) == 0.5
    rng = np.random.default_rng(42)
    shuffled = rng.permutation(np.repeat([0, 1], 500))
    scores = np.arange(1000)
    assert abs(auc_neg_farther_than_pos(scores[shuffled == 0], scores[shuffled == 1]) - .5) < .08


def test_topk_composition_and_all_successors():
    ranked = np.array([2, 3, 4, 5])
    comp = topk_composition(ranked, {2, 3}, {8, 9}, 2)
    assert np.isclose(sum(comp), 1)
    assert np.isclose(topk_composition(np.array([2, 3]), {2, 3}, set(), 2)[0], 1)


def test_successor_multiplicity_uses_all_rows():
    pairs = {"anchors": np.array([1, 1, 2]), "positives": np.array([3, 4, 5]),
             "negatives": np.array([[6], [7], [8]])}
    successors = build_successors(pairs)
    assert successors[1] == {3, 4}


if __name__ == "__main__":
    tests = [test_residual_levels_entering_quantizer, test_norm_statistics_finite_nonnegative,
             test_pair_loader_returns_file_arrays_unchanged, test_pair_protocol_positive_excluded_from_negatives,
             test_euclidean_and_rms_ranking, test_auc_ordering_and_ties, test_topk_composition_and_all_successors,
             test_successor_multiplicity_uses_all_rows]
    failed = []
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as error:
            failed.append((test.__name__, error))
            print(f"FAIL {test.__name__}: {error}")
    print(f"{len(tests) - len(failed)} passed, {len(failed)} failed")
    raise SystemExit(bool(failed))
