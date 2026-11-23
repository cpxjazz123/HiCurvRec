"""Wiring checks for the production teacher (issue #244 C5/C6/C11)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import teacher  # noqa: E402

N = 8
EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 4)]


def graph():
    rows = [a for a, _ in EDGES]
    cols = [b for _, b in EDGES]
    data = np.ones(len(EDGES), dtype=np.int64)
    out_csr = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    out_csr.sort_indices()
    return out_csr, out_csr.T.tocsr()


def synthetic():
    rng = np.random.default_rng(0)
    semantic = rng.normal(size=(N, 16)).astype(np.float32)
    structural = rng.normal(size=(N, 6)).astype(np.float32)
    return semantic, structural


def build_model():
    semantic, structural = synthetic()
    model = teacher.TopologyDescriptor(semantic.shape[1], structural.shape[1]).to(teacher.DEVICE)
    model.eval()
    return model, semantic, structural


def test_descriptor_is_128_dims_from_two_scales():
    model, semantic, structural = build_model()
    out_csr, in_csr = graph()
    rng = np.random.default_rng(0)
    with torch.no_grad():
        one = model.encode_two_scales([0], out_csr, in_csr, rng, semantic, structural)
        single_scale = model.encode([teacher.topology.ego_induced(0, out_csr, in_csr, rng, radius=1)],
                                    semantic, structural)
    assert one.shape == (1, 128), one.shape
    assert single_scale.shape == (1, 64), single_scale.shape


def test_gate_consumes_128_not_64():
    gate = teacher.Gate()
    with torch.no_grad():
        weights = gate(torch.zeros(3, 128))
    assert weights.shape == (3, len(teacher.CURVATURES))
    assert torch.allclose(weights.sum(dim=1), torch.ones(3), atol=1e-6)


def test_direction_reversal_changes_the_descriptor():
    """Flipping every edge must move the 128-dim descriptor: direction is really used."""
    model, semantic, structural = build_model()
    out_csr, in_csr = graph()
    rng = np.random.default_rng(1)
    with torch.no_grad():
        forward = model.encode_two_scales([0], out_csr, in_csr, rng, semantic, structural)
        flipped_out = out_csr.T.tocsr()
        rng = np.random.default_rng(1)
        backward = model.encode_two_scales([0], flipped_out, flipped_out.T.tocsr(), rng,
                                           semantic, structural)
    assert forward.shape == backward.shape == (1, 128)
    assert not torch.allclose(forward, backward, atol=1e-6), "direction is ignored"


def test_teacher_uses_the_induced_sampler():
    """The production path must be the induced-edge sampler, not the #243 tree walker."""
    import inspect
    source = inspect.getsource(teacher.TopologyDescriptor.encode_two_scales)
    assert "topology.ego_induced" in source, source


def main() -> int:
    tests = [(name, value) for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    failures = []
    for name, function in tests:
        try:
            function()
            print(f"PASS {name}")
        except Exception as error:  # noqa: BLE001
            failures.append((name, str(error)))
            print(f"FAIL {name}: {error}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
