"""Frozen definitions for the Issue254 paired diagnostic.

Every rule the analysis uses is here as a pure function so that the report,
the tables and the tests can never disagree about a definition.  The contract
is frozen before the numbers are read (see results/research/issue254/CONTRACT.md).
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

#: The two finished generations under comparison, in the order used by every table.
ARMS = ('issue252', 'issue253')

#: The reported cutoffs. They are the ones the frozen Stage3 protocol emits.
CUTOFFS = (5, 10)

#: Search beam has 20 tuples; only the first 10 selected items were persisted.
SEARCH_BEAM = 20
SAVED_CANDIDATES = 10

#: Group names for a paired comparison at one cutoff.
BOTH = 'both_hit'
ONLY_252 = 'only_issue252_hit'      # regression: the arm that lost the user
ONLY_253 = 'only_issue253_hit'      # improvement: the arm that won the user
NEITHER = 'neither_hit'


def hit(rank: Optional[int], cutoff: int) -> bool:
    """A target counts as hit at ``cutoff`` when its zero-based rank is inside it."""
    return rank is not None and 0 <= rank < cutoff


def ndcg_contribution(rank: Optional[int], cutoff: int) -> float:
    """The single-relevant-item NDCG contribution, zero when the target is outside."""
    if not hit(rank, cutoff):
        return 0.0
    return 1.0 / math.log2(rank + 2.0)


def truncated_rank(rank: Optional[int], saved: int = SAVED_CANDIDATES) -> int:
    """One-based display rank, or a lower bound with a separate censoring mask."""
    return saved + 1 if rank is None else rank + 1


def group(rank_252: Optional[int], rank_253: Optional[int], cutoff: int) -> str:
    """The paired group of one user at one cutoff."""
    hit_252 = hit(rank_252, cutoff)
    hit_253 = hit(rank_253, cutoff)
    if hit_252 and hit_253:
        return BOTH
    if hit_252:
        return ONLY_252
    if hit_253:
        return ONLY_253
    return NEITHER


def group_counts(pairs: Iterable[Tuple[Optional[int], Optional[int]]], cutoff: int) -> Dict[str, int]:
    """How many users fall into each paired group at one cutoff."""
    counts = {BOTH: 0, ONLY_252: 0, ONLY_253: 0, NEITHER: 0}
    for rank_252, rank_253 in pairs:
        counts[group(rank_252, rank_253, cutoff)] += 1
    return counts


def net_change(counts: Dict[str, int], population: int) -> float:
    """The paired net recall change: (improvements - regressions) / population.

    The denominator is the joined user population of the comparison, never the
    number of users in either arm alone.
    """
    if population <= 0:
        raise ValueError('paired net change needs a positive population')
    return (counts[ONLY_253] - counts[ONLY_252]) / population


def recall(pairs: Sequence[Tuple[Optional[int], Optional[int]]], cutoff: int, arm: int) -> float:
    """The per-arm recall at one cutoff, recomputed from the paired ranks."""
    if not pairs:
        raise ValueError('recall needs at least one paired user')
    hits = sum(1 for pair in pairs if hit(pair[arm], cutoff))
    return hits / len(pairs)


def ndcg(pairs: Sequence[Tuple[Optional[int], Optional[int]]], cutoff: int, arm: int) -> float:
    """The per-arm NDCG at one cutoff, recomputed from the paired ranks."""
    if not pairs:
        raise ValueError('ndcg needs at least one paired user')
    total = sum(ndcg_contribution(pair[arm], cutoff) for pair in pairs)
    return total / len(pairs)


def prefix_levels(sid_a: Sequence[int], sid_b: Sequence[int]) -> int:
    """How many leading SID levels two items share (0 = no shared level)."""
    shared = 0
    for left, right in zip(sid_a[:3], sid_b[:3]):
        if left != right:
            break
        shared += 1
    return shared


def per_item_contributions(rows: Iterable[dict], cutoff: int) -> List[dict]:
    """Aggregate the paired ledger to one row per target item.

    ``rows`` carry ``target_item``, ``rank_issue252``, ``rank_issue253``.
    The returned rows carry the user counts per group, the net contribution to
    the global recall change and the hit rates, so that the sum of the net
    contributions over the items equals the global net change exactly.
    """
    buckets: Dict[int, Dict[str, int]] = {}
    for row in rows:
        item = int(row['target_item'])
        bucket = buckets.setdefault(item, {BOTH: 0, ONLY_252: 0, ONLY_253: 0, NEITHER: 0})
        bucket[group(row['rank_issue252'], row['rank_issue253'],
                     cutoff)] += 1
    population = sum(sum(bucket.values()) for bucket in buckets.values())
    out = []
    for item in sorted(buckets):
        bucket = buckets[item]
        users = sum(bucket.values())
        out.append({
            'target_item': item,
            'cutoff': cutoff,
            'users': users,
            'both_hit': bucket[BOTH],
            'only_issue252_hit': bucket[ONLY_252],
            'only_issue253_hit': bucket[ONLY_253],
            'neither_hit': bucket[NEITHER],
            'regressions': bucket[ONLY_252],
            'improvements': bucket[ONLY_253],
            'net_contribution': (bucket[ONLY_253] - bucket[ONLY_252]) / population,
            'hit_rate_issue252': (bucket[BOTH] + bucket[ONLY_252]) / users,
            'hit_rate_issue253': (bucket[BOTH] + bucket[ONLY_253]) / users,
        })
    return out
