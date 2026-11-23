"""Hand-checked asset-manifest and join tests (issue #245 C3/C19)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.graph_infra import assets  # noqa: E402


def graph(edges, n):
    rows = [a for a, _ in edges]
    cols = [b for _, b in edges]
    data = np.ones(len(edges), dtype=np.int64)
    csr = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    csr.sort_indices()
    return csr


def test_graph_stats(tmp_path=None):
    stats = assets.graph_stats(graph([(0, 1), (1, 0), (1, 2)], 4))
    assert stats["nodes"] == 4
    assert stats["directed_edges"] == 3
    assert stats["reciprocal_pairs"] == 1
    assert stats["isolated_nodes"] == 1, "node 3 has no edge"
    assert stats["components"] == 2


def test_sparse_round_trip(tmp_path):
    csr = graph([(0, 1), (1, 2)], 3)
    path = tmp_path / "g.npz"
    assets.save_sparse(path, csr)
    loaded = assets.load_sparse(path)
    assert (loaded != csr).nnz == 0


def test_manifest_verifies_and_detects_drift(tmp_path):
    target = tmp_path / "bank.npz"
    np.savez_compressed(target, pairs=np.array([[0, 1]]), scores=np.array([0.5]))
    manifest = tmp_path / "manifest.json"
    assets.write_manifest(manifest, [
        assets.entry("bank", target, schema="test.v1", purpose="unit test",
                     command="pytest", dependencies=["none"]),
    ], issue=245)
    assert assets.verify_manifest(manifest)["verified"] == ["bank"]
    np.savez_compressed(target, pairs=np.array([[0, 1]]), scores=np.array([0.6]))  # tamper
    try:
        assets.verify_manifest(manifest)
    except ValueError as error:
        assert "hash drift" in str(error)
    else:
        raise AssertionError("tampered asset passed verification")


def test_join_orders_by_bank_pairs(tmp_path=None):
    bank = np.array([[0, 1], [2, 3], [4, 5]])
    model_pairs = np.array([[4, 5], [0, 1], [2, 3]])
    scores = np.array([9.0, 1.0, 5.0])
    aligned = assets.join_scores(bank, model_pairs, scores)
    assert aligned.tolist() == [1.0, 5.0, 9.0]


def test_join_rejects_missing_duplicate_and_misaligned(tmp_path=None):
    bank = np.array([[0, 1], [2, 3]])
    for pairs, scores, message in (
        (np.array([[0, 1]]), np.array([1.0]), "count mismatch"),
        (np.array([[0, 1], [0, 1]]), np.array([1.0, 2.0]), "duplicate"),
        (np.array([[0, 1], [3, 2]]), np.array([1.0, 2.0]), "missing"),
    ):
        try:
            assets.join_scores(bank, pairs, scores)
        except ValueError as error:
            assert message in str(error), (message, str(error))
        else:
            raise AssertionError(f"join accepted a bad input: {message}")


def test_join_rejects_non_finite_scores(tmp_path=None):
    bank = np.array([[0, 1]])
    try:
        assets.join_scores(bank, np.array([[0, 1]]), np.array([np.nan]))
    except ValueError as error:
        assert "non-finite" in str(error)
    else:
        raise AssertionError("NaN score passed the join")


def main() -> int:
    import tempfile
    tests = [(name, value) for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    failures = []
    for name, function in tests:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[3]) as tmp:
            try:
                function(Path(tmp))
                print(f"PASS {name}")
            except Exception as error:  # noqa: BLE001
                failures.append((name, str(error)))
                print(f"FAIL {name}: {error}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
