"""Regression tests for the collective-call fault that collapsed every run.

The fault: the collision decision broadcast ran inside ``if rank == 0``. Rank 0
entered an NCCL collective the other ranks skipped, the communicator went out of
step, and the gradient all-reduces that followed mixed buffers from different
steps. The symptom was exact: the partition was healthy at epoch 50 (collision
0.314, 19,770 unique triples) and collapsed to 0.999 within three hundred epochs,
with the mechanism never activated. A plain copy of the frozen baseline on the
same four cards held 0.17 through epoch 1400.

Two guards, because a deadlock cannot be asserted directly:

* the static guard reads every Stage 2 trainer and refuses a ``dist.<collective>``
  call inside a branch conditioned on the rank;
* the functional guard runs the fixed pattern on two gloo ranks and proves the
  group stays usable afterwards (both ranks take the same decision and a following
  all-reduce returns the sum).

Run:  python3 -m tools.research.stage2.tests.test_collectives
      (the functional guard runs only where torch is importable)
"""
from __future__ import annotations

import ast
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tools.research.stage2 import contract

COLLECTIVES = frozenset({
    "broadcast", "broadcast_object_list", "all_reduce", "all_gather",
    "all_gather_object", "reduce", "scatter", "gather", "barrier",
})


def _rank_branch(node: ast.If) -> bool:
    """Whether a branch is conditioned on the rank."""
    text = ast.unparse(node.test)
    return "rank" in text.lower() and "0" in text


def find_rank_branch_collectives(source: str) -> list[tuple[int, str, str]]:
    """Every (line, collective, condition) a trainer enters alone."""
    hits = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not _rank_branch(node):
            continue
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr in COLLECTIVES):
                hits.append((sub.lineno, sub.func.attr, ast.unparse(node.test)))
    return hits


def test_the_guard_catches_the_original_fault() -> str:
    """The check must flag the exact shape that collapsed every run."""
    source = """
def main(rank, world_size):
    accepted = 0
    if rank == 0:
        with torch.no_grad():
            tokens = model.get_indices(embeddings)
        decision = collision(tokens)
        accepted = 1 if decision == "accept" else 0
        if world_size > 1:
            dist.broadcast(accepted, 0)
    if world_size > 1:
        dist.barrier()
"""
    hits = find_rank_branch_collectives(source)
    assert [hit[1] for hit in hits] == ["broadcast"], f"guard missed the fault: {hits}"
    return "the detector flags dist.broadcast inside if rank == 0"


def test_no_collective_runs_inside_a_rank_branch() -> str:
    """Every collective must be entered by every rank."""
    root = Path(contract.resolve_root()).resolve()
    offenders = []
    trees = sorted((root / "stage2_RQ-VAE").glob("*/train_rqvae.py"))
    if not trees:
        return "no trainer to scan under stage2_RQ-VAE"
    for path in trees:
        for line, name, condition in find_rank_branch_collectives(
                path.read_text(encoding="utf-8")):
            offenders.append(f"{path.relative_to(root)}:{line} dist.{name} "
                             f"under {condition!r}")
    assert not offenders, "collective inside a rank branch: " + "; ".join(offenders[:6])
    return f"{len(trees)} trainers scanned, no collective inside a rank branch"


def _collective_worker(rank: int, world_size: int, rendezvous: str, out_dir: str) -> None:
    import torch
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method="file://" + rendezvous,
                            rank=rank, world_size=world_size)
    decision = torch.tensor(1 if rank == 0 else 0)
    if world_size > 1:
        dist.broadcast(decision, 0)
    total = torch.tensor(float(rank + 1))
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    Path(out_dir, f"rank{rank}.txt").write_text(
        f"{int(decision.item())} {float(total.item())}", encoding="utf-8")
    dist.destroy_process_group()


def test_a_collective_decision_keeps_the_group_usable() -> str:
    """The fixed pattern: every rank joins the broadcast, then all-reduces."""
    try:
        import torch.multiprocessing as mp
    except Exception:                                          # noqa: BLE001
        return "skipped: torch is not importable with this interpreter"
    with tempfile.TemporaryDirectory() as work:
        rendezvous = os.path.join(work, "rendezvous")
        mp.spawn(_collective_worker, args=(2, rendezvous, work), nprocs=2, join=True)
        values = [Path(work, f"rank{rank}.txt").read_text().split() for rank in (0, 1)]
    decisions = {int(value[0]) for value in values}
    totals = {float(value[1]) for value in values}
    assert decisions == {1}, f"ranks disagreed on the decision: {decisions}"
    assert totals == {3.0}, f"the all-reduce after the decision was not in step: {totals}"
    return "two gloo ranks agreed and kept the communicator in step"


def main() -> int:
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    failures = []
    for test in tests:
        try:
            print("PASS", test.__name__ + ":", test(), flush=True)
        except AssertionError as exc:
            failures.append(test.__name__ + ": " + str(exc))
            print("FAIL", test.__name__ + ":", exc, flush=True)
        except Exception as exc:                               # noqa: BLE001
            failures.append(test.__name__ + ": " + repr(exc))
            print("ERROR", test.__name__ + ":", repr(exc), flush=True)
    if failures:
        print("TESTS_FAIL", failures, flush=True)
        return 1
    print("TESTS_PASS", len(tests), "collective invariants hold", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
