"""Track F source-conditioned exclusion masks: RED before GREEN.

Issue #87 names the defect precisely: the existing ``exclusion_mask`` ignores ``target_ids`` and
applies a GLOBAL ``torch.isin`` over successor values, which can admit a known positive for its own
source and can wrongly exclude an item that is a successor of somebody else.

These tests are written against the DEFECTIVE behaviour first. Each one must fail before the fix
and pass after it, and each case is independent so that a mask which only happens to be right on
the aggregate cannot pass.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
for entry in (str(REPO_ROOT / "tools"),):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import r7_f_soft_separation as F  # noqa: E402

FAILURES: list[str] = []
CHECKS = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(f"{name}: {detail}")


def graph(n_items: int, edges: list[tuple[int, int]]) -> csr_matrix:
    """The same shape the host trainer builds: entry (i, j) is True iff j follows i."""
    rows = [edge[0] for edge in edges]
    cols = [edge[1] for edge in edges]
    matrix = csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)),
                        shape=(n_items, n_items), dtype=bool)
    matrix.sum_duplicates()
    matrix.data[:] = True
    return matrix


def test_legacy_exclusion_mask_exhibits_both_named_defects() -> None:
    """RED against the SHIPPED #85 implementation, which is the defect issue #87 describes.

    Kept as a standing regression test: if this ever passes, the legacy helper is no longer the
    one under test and the fix should be re-examined rather than assumed.
    """
    # item 2 is a successor of item 0 ONLY. Item 5 is the observed target of BOTH sources.
    candidates = torch.tensor([[1, 2], [0, 5]])
    # the legacy helper's own signature: a flat successor tensor, not the host's CSR
    legacy = F.exclusion_mask(
        source_ids=torch.tensor([1, 0]), target_ids=torch.tensor([5, 5]),
        candidates=candidates, successors=torch.tensor([[0, 2]]))
    kept = legacy.reshape(-1).tolist()
    check("the observed target is excluded for its own source",
          kept[1] is False, f"row 1 kept={kept[1]}: target_ids was ignored, so the true positive "
                            f"was admitted as an unrelated candidate")
    check("a successor of a DIFFERENT source is not excluded",
          kept[0] is True, f"row 0 kept={kept[0]}: the global successor filter dropped an "
                           f"allowed candidate because item 2 follows item 0, not item 1")


def test_the_global_successor_filter_is_the_defect() -> None:
    """RED: a global filter drops a candidate that is a successor of a DIFFERENT source.

    Item 2 is a successor of item 0. For source 1 it is an ordinary unknown candidate and must
    survive; the global filter removes it anyway.
    """
    graph_matrix = graph(8, [(0, 2)])
    candidates = torch.tensor([[1, 2]])
    mask = F.source_conditioned_mask(
        source_ids=torch.tensor([1]), target_ids=torch.tensor([5]),
        candidates=candidates, successors=graph_matrix)
    check("a successor of another source is NOT excluded for this source",
          bool(mask.reshape(-1)[0]), "the global successor filter removed an allowed candidate")


def test_a_known_positive_for_this_source_is_excluded() -> None:
    """RED: the old mask never looked at target_ids at all, so the true target survived."""
    graph_matrix = graph(8, [])
    candidates = torch.tensor([[0, 5]])
    mask = F.source_conditioned_mask(
        source_ids=torch.tensor([0]), target_ids=torch.tensor([5]),
        candidates=candidates, successors=graph_matrix)
    check("this source's observed target is excluded",
          not bool(mask.reshape(-1)[0]), "the true target was admitted as an unrelated candidate")


def test_duplicate_targets_are_all_excluded() -> None:
    """Several rows can share one target; every one of them must be excluded, not the first."""
    graph_matrix = graph(8, [])
    candidates = torch.tensor([[0, 5], [2, 5], [3, 5]])
    mask = F.source_conditioned_mask(
        source_ids=torch.tensor([0, 2, 3]), target_ids=torch.tensor([5, 5, 5]),
        candidates=candidates, successors=graph_matrix)
    check("every row whose candidate is an observed target is excluded",
          not bool(mask.reshape(-1).any()), str(mask.reshape(-1).tolist()))


def test_known_successor_of_the_same_source_is_excluded() -> None:
    """The false-negative case the host's own CSR exists for."""
    graph_matrix = graph(8, [(0, 3), (0, 6)])
    candidates = torch.tensor([[0, 3], [0, 6], [0, 7]])
    mask = F.source_conditioned_mask(
        source_ids=torch.tensor([0]), target_ids=torch.tensor([1]),
        candidates=candidates, successors=graph_matrix)
    kept = mask.reshape(-1).tolist()
    check("known successors of this source are excluded and an unknown one is kept",
          kept == [False, False, True], str(kept))


def test_self_pairs_are_excluded() -> None:
    graph_matrix = graph(8, [])
    candidates = torch.tensor([[4, 4]])
    mask = F.source_conditioned_mask(
        source_ids=torch.tensor([4]), target_ids=torch.tensor([1]),
        candidates=candidates, successors=graph_matrix)
    check("a self pair is excluded", not bool(mask.reshape(-1)[0]), "self pair survived")


def test_source_equal_to_its_own_target_is_still_excluded() -> None:
    """source == target is a degenerate positive; it must not become an unrelated pair."""
    graph_matrix = graph(8, [])
    candidates = torch.tensor([[2, 2]])
    mask = F.source_conditioned_mask(
        source_ids=torch.tensor([2]), target_ids=torch.tensor([2]),
        candidates=candidates, successors=graph_matrix)
    check("a source that is its own target is excluded", not bool(mask.reshape(-1)[0]),
          "degenerate self pair survived")


def test_empty_valid_set_is_reported_not_silently_zero() -> None:
    """Every candidate excluded must be reported as an empty set, never scored as a clean zero."""
    graph_matrix = graph(8, [(0, 3)])
    candidates = torch.tensor([[0, 3]])
    result = F.valid_pair_indices(
        source_ids=torch.tensor([0]), target_ids=torch.tensor([3]),
        candidates=candidates, successors=graph_matrix)
    check("an all-excluded candidate set yields an empty valid list",
          result["valid"].shape[0] == 0, str(result["valid"].shape))
    check("counts are reported: candidate, valid and empty",
          result["candidate_count"] == 1 and result["valid_count"] == 0
          and result["empty_count"] == 1, str(result))


def test_counts_are_logged_at_every_call() -> None:
    """candidate_count / valid_count / empty_count must be available per step, not inferred."""
    graph_matrix = graph(8, [(0, 3)])
    candidates = torch.tensor([[0, 3], [0, 5], [1, 2]])
    result = F.valid_pair_indices(
        source_ids=torch.tensor([0, 0, 1]), target_ids=torch.tensor([5, 5, 2]),
        candidates=candidates, successors=graph_matrix)
    check("counts partition the candidate set",
          result["candidate_count"] == result["valid_count"] + result["empty_count"],
          str(result))


def test_candidate_positions_are_not_confused_with_item_ids() -> None:
    """A candidate is an ITEM pair. The mask must index the graph by item id, not by row index."""
    graph_matrix = graph(8, [(2, 6)])       # item 6 follows item 2
    candidates = torch.tensor([[2, 6], [2, 4]])
    mask = F.source_conditioned_mask(
        source_ids=torch.tensor([2]), target_ids=torch.tensor([4]),
        candidates=candidates, successors=graph_matrix)
    kept = mask.reshape(-1).tolist()
    check("the graph is indexed by item id, not by candidate row",
          kept == [False, False], str(kept))


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001
            FAILURES.append(f"{test.__name__} raised {type(exc).__name__}: {exc}")
    print(f"r7_f_masks: {CHECKS} checks, {len(FAILURES)} failures")
    for failure in FAILURES:
        print(f"  FAIL {failure}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
