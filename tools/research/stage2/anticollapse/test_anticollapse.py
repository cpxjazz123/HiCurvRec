"""The floor's guarantee, the patcher's safety, and a run against both real quantisers.

The last two tests are the ones that matter for "works with any mechanism": they copy an
actual tree -- the frozen baseline (Euclidean VQLayer) and the F0 mixed quantiser -- patch
it, and run the patched class in a fresh interpreter on the cost matrix that collapsed every
arm in this investigation (three cheap codewords, five unreachable).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anticollapse import (InfeasibleFloor, UnknownTree, apply_floor,  # noqa: E402
                          constrained_assignment, verify_tree)

BASELINE = REPO / "stage2_RQ-VAE" / "baseline" / "TIGER_RQ-VAE"
MIXED = REPO / "results" / "research" / "qda" / "trees" / "F0_seed42"
SCRATCH = REPO / "results" / "research" / "qda" / "anticollapse_check"


def _collapsed(items=4096, codewords=256, live=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    cost = torch.rand(items, codewords, generator=generator) + 10.0
    cost[:, :live] = torch.rand(items, live, generator=generator)
    return cost


def test_floor_rescues_a_collapsed_cost():
    cost = _collapsed()
    assignment, report = constrained_assignment(cost, 1)
    counts = torch.bincount(assignment, minlength=cost.shape[1])
    assert int(counts.min()) == 1
    assert report["moves"] == 248


def test_floor_is_free_when_the_assignment_already_satisfies_it():
    generator = torch.Generator().manual_seed(1)
    cost = torch.rand(4096, 256, generator=generator)
    assignment, report = constrained_assignment(cost, 1)
    assert report["moves"] == 0 and torch.equal(assignment, torch.argmin(cost, dim=-1))


def test_infeasible_batch_raises():
    with pytest.raises(InfeasibleFloor):
        constrained_assignment(torch.rand(100, 256), 1)


@pytest.fixture()
def scratch():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    made = []
    def make(name: str, source: Path) -> Path:
        target = SCRATCH / name
        if target.exists():
            shutil.rmtree(target)
        # the frozen baseline ships a dangling dataset symlink; it is not part of the patch
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"),
                        symlinks=True, ignore_dangling_symlinks=True)
        made.append(target)
        return target
    yield make
    for path in made:
        shutil.rmtree(path, ignore_errors=True)


def test_patched_baseline_survives_a_collapsed_cost(scratch):
    tree = scratch("euclid", BASELINE)
    report = apply_floor(tree)
    assert report["changed"] is True and report["family"] == "euclid"
    verdict = verify_tree(tree)
    assert verdict["ok"] is True, verdict
    assert verdict["used"] == verdict["codewords"] == 8


def test_patched_mixed_tree_survives_a_collapsed_cost(scratch):
    tree = scratch("mixed", MIXED)
    report = apply_floor(tree)
    assert report["changed"] is True and report["family"] == "mixed"
    verdict = verify_tree(tree)
    assert verdict["ok"] is True, verdict


def test_patching_is_idempotent(scratch):
    tree = scratch("twice", BASELINE)
    first = apply_floor(tree)
    body = (tree / first["module"]).read_text(encoding="utf-8")
    second = apply_floor(tree)
    assert first["changed"] is True and second["changed"] is False
    assert (tree / first["module"]).read_text(encoding="utf-8") == body


def test_an_unrecognised_quantiser_fails_closed(scratch):
    tree = scratch("unknown", MIXED)
    original = (tree / "model" / "mixed_rq.py").read_text(encoding="utf-8")
    (tree / "model" / "mixed_rq.py").write_text(
        original.replace("return torch.argmin(distances, dim=-1)",
                         "return torch.argmin(distances, dim=-2)"),
        encoding="utf-8")
    with pytest.raises(UnknownTree):
        apply_floor(tree)


def test_a_tree_without_a_quantiser_fails_closed(scratch):
    empty = SCRATCH / "empty"
    empty.mkdir(parents=True, exist_ok=True)
    (empty / "model").mkdir(exist_ok=True)
    try:
        with pytest.raises(UnknownTree):
            apply_floor(empty)
    finally:
        shutil.rmtree(empty, ignore_errors=True)
