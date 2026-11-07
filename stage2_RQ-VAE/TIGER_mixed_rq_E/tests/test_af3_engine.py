"""The incremental engine must equal a full recomputation, exactly.

The per-prefix arm accepts or rejects a candidate using the distortion of a
*local* change. That is only sound if the local update touches every pair whose
readout or triangle term can have moved and no others -- so this test builds a
state, changes a set of items, applies the local update, then throws the state
away and recomputes everything from the same SIDs. The two must agree to machine
precision, pair by pair.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))

from model.af3_engine import StructuralState  # noqa: E402


def _synthetic(n_nodes: int = 60, seed: int = 3):
    """A small support with observed pairs, unobserved pairs and their targets."""
    rng = np.random.default_rng(seed)
    pairs = set()
    while len(pairs) < 260:
        a, b = rng.integers(0, n_nodes, 2)
        if a == b:
            continue
        pairs.add((int(min(a, b)), int(max(a, b))))
    pairs = np.array(sorted(pairs), dtype=np.int64)
    is_edge = rng.random(len(pairs)) < 0.6
    kappa = np.where(is_edge, rng.integers(-40, 5, len(pairs)).astype(np.float64), 0.0)
    return pairs, is_edge, kappa


def _make_state(seed: int = 3, n_nodes: int = 60):
    pairs, is_edge, kappa = _synthetic(n_nodes=n_nodes, seed=seed)
    state = StructuralState(pairs, is_edge, kappa, n_nodes, tau=0.3, intercept=-0.7, scale=12.0)
    return state


def test_a_local_update_equals_a_full_recomputation() -> str:
    """After changing a set of SIDs, the local update equals a full recompute."""
    state = _make_state()
    generator = torch.Generator().manual_seed(23)
    phi = torch.randn(state.n_nodes, 32, generator=generator, dtype=torch.float64)
    state.full(phi)
    changed = np.array([4, 17, 33, 41], dtype=np.int64)
    phi[changed] = phi[changed] * 0.5 + torch.randn(len(changed), 32,
                                                    generator=generator, dtype=torch.float64)
    local = state.update(phi, changed)
    locally_updated = state.state()
    fresh = StructuralState(state.pairs, state.is_edge, state.kappa, state.n_nodes,
                            tau=state.tau, intercept=state.intercept, scale=state.scale)
    reference = fresh.full(phi)
    for key, values in locally_updated.items():
        np.testing.assert_allclose(values, fresh.state()[key], rtol=0, atol=1e-12,
                                   err_msg=f"the local update diverged on {key}")
    if abs(local["distortion"] - reference["distortion"]) > 1e-12:
        raise AssertionError("the local distortion differs: "
                             + repr(local["distortion"]) + " vs " + repr(reference["distortion"]))
    return "four changed items: every cached pair and the distortion agree with a full run"


def test_an_unchanged_update_is_a_no_op() -> str:
    """Without a real change the update must move nothing."""
    state = _make_state(seed=5)
    generator = torch.Generator().manual_seed(31)
    phi = torch.randn(state.n_nodes, 32, generator=generator, dtype=torch.float64)
    state.full(phi)
    before = state.state()
    state.update(phi, np.array([7], dtype=np.int64))
    after = state.state()
    for key in before:
        np.testing.assert_allclose(after[key], before[key], rtol=0, atol=1e-12)
    return "an update with no changed item leaves the state untouched"


def test_the_engine_matches_the_reference_distortion() -> str:
    """The sparse engine and the reference path agree on the same state."""
    import torch as _torch

    from model import af3_structure as reference

    state = _make_state(seed=9)
    generator = _torch.Generator().manual_seed(37)
    phi = _torch.randn(state.n_nodes, 32, generator=generator)
    engine = state.full(phi)
    support = {"pairs": state.pairs,
               "is_edge": state.is_edge,
               "edges": int(state.is_edge.sum()),
               "unobserved": int((~state.is_edge).sum())}
    kappa = {"rows": state.pairs[state.is_edge, 0], "cols": state.pairs[state.is_edge, 1],
             "kappa": state.kappa[state.is_edge],
             "degree": np.zeros(state.n_nodes), "triangles": np.zeros(1)}
    wanted = reference.structural_distortion(
        phi, support, kappa, tau=state.tau, intercept=state.intercept,
        n_nodes=state.n_nodes, scale=state.scale)
    if abs(engine["distortion"] - wanted["distortion"]) > 1e-9:
        raise AssertionError("the engine and the reference disagree: "
                             + repr(engine["distortion"]) + " vs " + repr(wanted["distortion"]))
    return "the sparse engine reproduces the reference distortion"


def main() -> int:
    failures = 0
    for test in (
        test_a_local_update_equals_a_full_recomputation,
        test_an_unchanged_update_is_a_no_op,
        test_the_engine_matches_the_reference_distortion,
    ):
        try:
            print("PASS " + test.__name__ + ": " + test())
        except AssertionError as exc:
            failures += 1
            print("FAIL " + test.__name__ + ": " + str(exc))
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print("ERROR " + test.__name__ + ": " + type(exc).__name__ + ": " + str(exc))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

