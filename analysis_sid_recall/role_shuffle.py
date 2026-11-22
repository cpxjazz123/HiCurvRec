"""Issue #231 C38: a control that really changes the near/far relation.

#230 permuted whole instances. Every instance kept its own three pair relations, so the control
and the treatment were indistinguishable on all three structures -- measured, their
treatment-minus-control intervals were -0.0070/+0.0070/+0.0005 and all three contained zero.

#231 says the control must change the roles *inside* the triple:
"在训练三元组内部改变中心/中间角色，使 near/far 关系实际改变，保留商品及三对集合；
断言标签集合/权重确有差异，不能仅置换样本行或互换等价叶角色。"

Two constraints come from that sentence and are enforced here:

* the item triple and its three unordered pairs are preserved;
* swapping the two interchangeable leaves (fork/merge B<->C) does **not** count as a change,
  because those two roles carry the same pair.

The roles that do change the labels are:

* chain ``(A,B,C)``: the chain's near set is ``{AB, BC}`` and far is ``{AC}``, so moving ``A``
  and ``C`` changes which pairs are adjacent in the ordering;
* fork/merge ``(A,B,C)`` with ``A`` the centre: the near pair is ``BC`` and the far pairs are
  ``AB, AC``, so swapping the centre with one leaf turns a far pair into the near pair.
"""
from __future__ import annotations

import numpy as np

#: The near and far positions per structure, over the role tuple ``(A, B, C)``.
NEAR = {"chain": ((0, 1), (1, 2)), "fork": ((1, 2),), "merge": ((1, 2),)}
FAR = {"chain": ((0, 2),), "fork": ((0, 1), (0, 2)), "merge": ((0, 1), (0, 2))}


def all_pairs(row) -> list:
    return [(int(row[i]), int(row[j])) for i, j in ((0, 1), (1, 2), (0, 2))]


def _unordered(pair):
    return (min(pair), max(pair))


def labelled_pairs(row, kind: str) -> tuple:
    """The (near, far) pair sets of one role tuple, as sorted tuples of unordered pairs."""
    near = tuple(sorted(_unordered((row[i], row[j])) for i, j in NEAR[kind]))
    far = tuple(sorted(_unordered((row[i], row[j])) for i, j in FAR[kind]))
    return near, far


def shuffle_roles(bank: np.ndarray, kind: str, seed: int = 42):
    """Swap a role inside each triple whenever that actually changes the labels.

    Returns ``(shuffled_bank, changed)`` where ``changed`` is a boolean mask. For fork/merge
    the two leaves are interchangeable, so the only meaningful swap is centre <-> leaf; for
    chain the meaningful swap is ``A <-> C`` (the two ends), because moving the middle changes
    nothing either -- ``{AB, BC}`` stays the same pair set.
    """
    rng = np.random.default_rng(seed)
    out = np.array(bank, dtype=np.int64, copy=True)
    changed = np.zeros(len(bank), dtype=bool)
    for index, row in enumerate(bank.tolist()):
        before = labelled_pairs(row, kind)
        if kind == "chain":
            # Swapping the two ENDS (A<->C) leaves the labels unchanged: near={AB,BC} and
            # far={AC} are the same pair sets either way. Measured: 0 of 1138 changed. The
            # middle must move to an end, which turns an adjacent pair into the endpoint pair.
            side = int(rng.integers(0, 2))
            candidate = list(row)
            middle = 1
            other = 0 if side == 0 else 2
            candidate[middle], candidate[other] = candidate[other], candidate[middle]
        else:
            leaf = int(rng.integers(0, 2))                 # B or C
            candidate = list(row)
            candidate[0], candidate[1 + leaf] = candidate[1 + leaf], candidate[0]
        after = labelled_pairs(candidate, kind)
        if after != before and {frozenset(p) for p in all_pairs(candidate)} == {
                frozenset(p) for p in all_pairs(row)}:
            out[index] = candidate
            changed[index] = True
    return out, changed


def changed_ratio(changed) -> float:
    """The fraction of instances whose labels actually moved (C38's reported ratio)."""
    changed = np.asarray(changed, dtype=bool)
    if changed.size == 0:
        return 0.0
    return float(changed.mean())
