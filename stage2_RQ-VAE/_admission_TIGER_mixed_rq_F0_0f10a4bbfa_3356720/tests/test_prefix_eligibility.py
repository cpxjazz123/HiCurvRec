"""Unsupported persisted local weights must stop a round, not pretend to inherit.

The synthetic token assignments isolate support eligibility from geometry. The
test executes the real outer-loop entry with its real structural cache; it does
not prescribe pruning, a changed acceptance policy, or a formal training run.
"""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

TREE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TREE))

from model import af3_selector, mixed_rq  # noqa: E402
from model.af3_engine import StructuralState  # noqa: E402
from model.mixed_rq import AlphaTable  # noqa: E402


def _fixture(kind, level, prefix):
    spec = importlib.util.spec_from_file_location(
        "issue139_prefix_eligibility_trainer", TREE / "train_rqvae.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)

    n_nodes = 40
    group_size = 31 if kind == "few_items" else 32
    tokens = torch.zeros((n_nodes, 3), dtype=torch.long)
    tokens[group_size:, 0] = 1
    pairs = np.array([(i, j) for i in range(n_nodes)
                      for j in range(i + 1, n_nodes)], dtype=np.int64)
    touches = (pairs[:, 0] < group_size) | (pairs[:, 1] < group_size)
    touching_positions = np.flatnonzero(touches)
    if kind in ("few_edges", "valid_edges64"):
        is_edge = np.zeros(len(pairs), dtype=bool)
        count = 63 if kind == "few_edges" else 64
        is_edge[touching_positions[:count]] = True
    elif kind in ("few_unobserved", "valid_unobserved64"):
        is_edge = np.ones(len(pairs), dtype=bool)
        count = 63 if kind == "few_unobserved" else 64
        is_edge[touching_positions[:count]] = False
    else:
        is_edge = (pairs[:, 0] + pairs[:, 1]) % 3 == 0

    # Explicit fixture checks, independent of the production eligibility helper.
    if kind == "few_items":
        assert group_size == 31
        assert is_edge[touches].sum() >= 64
        assert (~is_edge[touches]).sum() >= 64
    elif kind == "few_edges":
        assert group_size == 32 and is_edge[touches].sum() == 63
        assert (~is_edge[touches]).sum() >= 64
    elif kind == "few_unobserved":
        assert group_size == 32 and (~is_edge[touches]).sum() == 63
        assert is_edge[touches].sum() >= 64
    elif kind == "absent_prefix":
        assert not (tokens[:, :len(prefix)] == torch.tensor(prefix)).all(dim=1).any()
    elif kind in ("valid", "valid_edges64", "valid_unobserved64"):
        assert group_size == 32
        assert is_edge[touches].sum() >= 64
        assert (~is_edge[touches]).sum() >= 64
        if kind == "valid_edges64":
            assert is_edge[touches].sum() == 64
        elif kind == "valid_unobserved64":
            assert (~is_edge[touches]).sum() == 64
    else:
        raise AssertionError("unknown fixture")

    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for i, j in pairs[is_edge]:
        adjacency[i, j] = adjacency[j, i] = 1.0
    degree = adjacency.sum(axis=1)
    targets = np.array([
        4.0 - degree[i] - degree[j] + 3.0 * np.dot(adjacency[i], adjacency[j])
        if observed else 0.0 for (i, j), observed in zip(pairs, is_edge)])
    state = StructuralState(pairs, is_edge, targets, n_nodes,
                            tau=0.5, intercept=0.2, scale=1.0)
    table = AlphaTable()
    table.set(level, prefix, 0.75)
    rq = SimpleNamespace(levels=[SimpleNamespace(n_embed=8) for _ in range(3)],
                         alpha_table=table)
    raw = SimpleNamespace(encoder=lambda x: x, decoder=lambda x: x,
                          rq=rq, _structure_state=state)
    features = torch.arange(1, n_nodes * 4 + 1, dtype=torch.float64).reshape(n_nodes, 4)
    graph = {"n_nodes": n_nodes, "rows": pairs[is_edge, 0],
             "cols": pairs[is_edge, 1], "item_ids": np.arange(n_nodes, dtype=np.int64)}
    support = {"pairs": pairs, "is_edge": is_edge}
    return trainer, raw, features, (graph, support, targets), tokens


def _run_without_candidate_mutation(trainer, raw, features, bundle, tokens, arm, calls,
                                    check_candidate=None):
    def record_choice(*_args, **kwargs):
        calls.append((kwargs["level"], kwargs["prefix"]))
        if check_candidate is not None:
            check_candidate(kwargs)
        return {"accepted": None, "attempts": [], "distortion": 1.0}

    with patch.object(trainer, "ARM_VARIANT", arm), \
            patch.object(af3_selector, "choose_weight", record_choice), \
            patch.object(af3_selector, "_readout_of", lambda z, *_args: z), \
            patch.object(mixed_rq, "encode_tokens", lambda *_args, **_kwargs: tokens.clone()):
        return trainer.run_outer_loop(
            2000, raw, features, features, torch.device("cpu"), bundle,
            {"tau": 0.5, "intercept": 0.2}, lambda *_args, **_kwargs: None)


@pytest.mark.parametrize("arm", ["L", "E"])
@pytest.mark.parametrize("kind,level,prefix", [
    ("few_items", 2, (0,)),
    ("few_edges", 2, (0,)),
    ("few_unobserved", 3, (0, 0)),
    ("absent_prefix", 3, (7, 7)),
])
def test_stale_explicit_local_entry_stops_before_any_selection(kind, level, prefix, arm):
    trainer, raw, features, bundle, tokens = _fixture(kind, level, prefix)
    table = raw.rq.alpha_table
    before = copy.deepcopy(table.to_dict())
    calls = []
    with pytest.raises(RuntimeError, match="(?i)(support|prefix|inherit)"):
        _run_without_candidate_mutation(trainer, raw, features, bundle, tokens, arm, calls)
    assert calls == [], "eligibility was checked only after another coordinate was selected"
    assert raw.rq.alpha_table is table
    assert table.to_dict() == before, "the preflight silently pruned or changed persisted weights"


@pytest.mark.parametrize("arm", ["L", "E"])
@pytest.mark.parametrize("kind", ["valid", "valid_edges64", "valid_unobserved64"])
def test_supported_explicit_entry_does_not_disable_the_outer_loop(arm, kind):
    trainer, raw, features, bundle, tokens = _fixture(kind, 2, (0,))
    before = copy.deepcopy(raw.rq.alpha_table.to_dict())
    calls = []
    result = _run_without_candidate_mutation(
        trainer, raw, features, bundle, tokens, arm, calls)
    assert (1, ()) in calls and (2, (0,)) in calls
    assert raw.rq.alpha_table.to_dict() == before
    assert result["alpha_table"] == before


@pytest.mark.parametrize("arm", ["L", "E"])
def test_ancestor_candidate_cannot_activate_an_unsupported_existing_local_entry(arm):
    trainer, raw, features, bundle, tokens = _fixture("valid", 2, (0,))
    before = copy.deepcopy(raw.rq.alpha_table.to_dict())
    changed = tokens.clone()
    changed[0, 0] = 1
    assert (tokens[:, 0] == 0).sum() == 32
    assert (changed[:, 0] == 0).sum() == 31
    checked = []

    def check_candidate(kwargs):
        if kwargs["level"] != 1:
            return
        checked.append(True)
        # All other guards are held equal or improve in this fixture. The
        # persistent (level=2, prefix=(0,)) alpha is the only lost eligibility.
        feasible, reason = kwargs["evaluate_feasibility"](changed, {"distortion": 0.1})
        assert not feasible, "an ancestor candidate kept a now-unsupported explicit child active"
        assert any(word in reason.lower() for word in ("support", "prefix", "inherit"))

    calls = []
    _run_without_candidate_mutation(trainer, raw, features, bundle, tokens,
                                    arm, calls, check_candidate=check_candidate)
    assert len(checked) == 1
    assert raw.rq.alpha_table.to_dict() == before
