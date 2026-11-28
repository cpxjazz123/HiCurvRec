"""Issue #287 round 2 (pre-registered): the same-anchor behavioural pair index.

For every centre m of the frozen triplet table the two outer nodes b and c form

  * a **closed** triangle (bc_dist_sq == 1): b and c are directly connected, so (b, c) is an
    observed behavioural relation; and
  * an **open** wedge (bc_dist_sq == 4): b and c are at distance two, so (b, c) is an unobserved pair
    (a potential false negative, recorded as such).

For every anchor (b or c) the candidate pairs are therefore

  * positive: an anchor's **direct neighbour** (the edge exists), and
  * negative: an anchor's **distance-two co-neighbour** (no edge).

The curvature only screens **inside** a comparability group - the anchor, the unordered multiset of
the two centre-edge direction flags and the anchor's degree bucket - exactly as in Issue #274: the
positive candidate with the highest raw curvature and the negative candidate with the lowest raw
curvature are kept, and a group whose extremum is tied is dropped (no arbitrary tie-break).

Nothing in this module trains; it produces the frozen sample set the candidate reads.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path('.').resolve()
OUT = REPO / 'results/research/issue287'
SHARDS = [f'shard_{i:03d}.npz' for i in range(13)]
HOLDOUT_MODULUS = 5
DEGREE_BUCKETS = 10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def edge_keys(path: Path) -> np.ndarray:
    blob = np.load(path)
    indptr = blob['indptr']; indices = blob['indices']
    nodes = len(indptr) - 1
    rows = np.repeat(np.arange(nodes), np.diff(indptr))
    return np.unique(rows.astype(np.int64) * nodes + indices.astype(np.int64))


def has_edge(keys: np.ndarray, nodes: int, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    probe = src.astype(np.int64) * nodes + dst.astype(np.int64)
    pos = np.searchsorted(keys, probe)
    pos = np.clip(pos, 0, len(keys) - 1)
    return keys[pos] == probe


def extremes(groups: np.ndarray, curvature: np.ndarray, others: np.ndarray, take_max: bool):
    """The per-group extremum with the #274 tie rule (a tied extremum drops the group).

    Vectorised: the group is sorted by (group, curvature); the first row of a group is its extremum
    and the group is kept only when that extremum value occurs exactly once inside the group.
    """
    order = np.lexsort((curvature if take_max else -curvature, groups))
    g = groups[order]; v = curvature[order]; o = others[order]
    first = np.ones(len(g), dtype=bool)
    first[1:] = g[1:] != g[:-1]
    starts = np.flatnonzero(first)
    ends = np.append(starts[1:], len(g))
    best_value = v[starts]
    repeated = np.repeat(best_value, ends - starts)
    equal = (v == repeated).astype(np.int64)
    tie_counts = np.add.reduceat(equal, starts)
    keep = tie_counts == 1
    return g[starts][keep], o[starts][keep], best_value[keep]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    nodes = 24587
    forward = edge_keys(REPO / 'assets/train/visible_forward.npz')
    reverse = edge_keys(REPO / 'assets/train/visible_reverse.npz')
    und = np.load(REPO / 'assets/train/visible_undirected.npz')
    degree = np.diff(und['indptr']).astype(np.int64)
    bounds = np.quantile(degree, np.linspace(0.1, 0.9, DEGREE_BUCKETS - 1))
    bucket = np.clip(np.searchsorted(bounds, degree, side='right'), 0, DEGREE_BUCKETS - 1).astype(np.int64)
    pos_best: dict[int, tuple[int, float]] = {}
    neg_best: dict[int, tuple[int, float]] = {}
    rows_total = 0
    for name in SHARDS:
        blob = np.load(REPO / 'assets/shards' / name)
        m = blob['centre'].astype(np.int64); b = blob['b'].astype(np.int64); c = blob['c'].astype(np.int64)
        dist = blob['bc_dist_sq'].astype(np.int64)
        curv = blob['curvature'].astype(np.float64)
        rows_total += len(m)
        dir_mb = has_edge(forward, nodes, m, b).astype(np.int64) + 2 * has_edge(reverse, nodes, b, m).astype(np.int64)
        dir_mc = has_edge(forward, nodes, m, c).astype(np.int64) + 2 * has_edge(reverse, nodes, c, m).astype(np.int64)
        multi = np.minimum(dir_mb, dir_mc) * 4 + np.maximum(dir_mb, dir_mc)
        closed = dist == 1
        opened = dist == 4
        for role_anchor, role_other, mask, take_max, store in (
                (b, c, closed, True, pos_best), (c, b, closed, True, pos_best),
                (b, c, opened, False, neg_best), (c, b, opened, False, neg_best)):
            if not mask.any():
                continue
            anchors = role_anchor[mask]; others = role_other[mask]
            keys = (anchors * (16 * DEGREE_BUCKETS) + multi[mask] * DEGREE_BUCKETS + bucket[anchors])
            g, o, cv = extremes(keys, curv[mask], others, take_max)
            for key, other, value in zip(g.tolist(), o.tolist(), cv.tolist()):
                store[key] = (other, value)
    common = sorted(set(pos_best) & set(neg_best))
    anchor = np.array([key // (16 * DEGREE_BUCKETS) for key in common], dtype=np.int64)
    pos = np.array([pos_best[key][0] for key in common], dtype=np.int64)
    neg = np.array([neg_best[key][0] for key in common], dtype=np.int64)
    pos_curv = np.array([pos_best[key][1] for key in common], dtype=np.float64)
    neg_curv = np.array([neg_best[key][1] for key in common], dtype=np.float64)
    holdout = (anchor % HOLDOUT_MODULUS) == 0
    path = OUT / 'pairs_same_anchor.npz'
    np.savez(path, anchor=anchor, pos=pos, neg=neg, pos_curvature=pos_curv, neg_curvature=neg_curv,
             is_holdout=holdout)
    manifest = {
        'schema': 'genrec.issue287.same_anchor_pairs/1',
        'path': str(path), 'sha256': sha256(path),
        'rules': {
            'positive': 'the anchor\'s direct neighbour: the closed triangle (bc_dist_sq == 1) outer pair',
            'negative': 'the anchor\'s distance-two co-neighbour: the open wedge (bc_dist_sq == 4) outer pair',
            'comparability_group': 'the anchor, the unordered multiset of the two centre-edge direction '
                                   'flags and the anchor\'s degree decile',
            'curvature_role': 'offline selection inside the group only; it never enters the loss, the '
                              'margin or a weight',
            'tie_rule': 'a group whose extremum curvature is tied is dropped',
            'holdout': f'anchor % {HOLDOUT_MODULUS} == 0',
            'false_negative_note': 'an unobserved pair is not an irrelevant pair; the negative label is '
                                    'a relative-association label, not a proof of irrelevance',
        },
        'counts': {'rows_scanned': int(rows_total), 'groups_positive': len(pos_best),
                   'groups_negative': len(neg_best), 'groups_common': len(common),
                   'anchors': int(len(np.unique(anchor))),
                   'train_groups': int((~holdout).sum()), 'holdout_groups': int(holdout.sum()),
                   'items_touched': int(len(np.unique(np.concatenate([anchor, pos, neg]))))},
        'inputs': {'visible_forward': sha256(REPO / 'assets/train/visible_forward.npz'),
                   'visible_reverse': sha256(REPO / 'assets/train/visible_reverse.npz'),
                   'visible_undirected': sha256(REPO / 'assets/train/visible_undirected.npz')},
    }
    (OUT / 'pairs_same_anchor.manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps(manifest['counts'], indent=1))


if __name__ == '__main__':
    main()
