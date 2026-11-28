"""Issue #287 round 5: the same-anchor index with a **non-adjacent** behavioural positive.

Rounds 1-4 measured a monotone trade-off on the (P, N, C) plane: the stronger the ranking term, the
higher P and N and the lower C; the issue's three thresholds ask for a point *off* that curve.  The
measured reason is that the Issue #274-style positive pair is **adjacent** (a closed triangle's outer
pair), so pulling it together drags a whole direct neighbourhood into one L1 bucket.

This index therefore uses, for every anchor ``a``:

  * the **positive** candidate: an item ``x`` at distance two from ``a`` that shares a neighbour with
    it (a behavioural co-occurrence: both interacted with the same item) but is **not adjacent** to it
    - the open-wedge outer pair of Issue #274, used here as the *positive*;
  * the **negative** candidate: an item ``y`` at distance two from ``a`` through a **different** centre,
    i.e. a relative association that is not the positive's context.

The curvature only screens inside a comparability group (the anchor, the unordered multiset of the two
centre-edge direction flags, the anchor's degree decile): the highest-curvature candidate is the
positive and the lowest-curvature candidate from another centre is the negative; a tied extremum drops
the group (Issue #274's rule).  Nothing here trains; it produces the frozen sample set.
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
NODES = 24587


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


def has_edge(keys: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    probe = src.astype(np.int64) * NODES + dst.astype(np.int64)
    pos = np.clip(np.searchsorted(keys, probe), 0, len(keys) - 1)
    return keys[pos] == probe


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    forward = edge_keys(REPO / 'assets/train/visible_forward.npz')
    reverse = edge_keys(REPO / 'assets/train/visible_reverse.npz')
    und = np.load(REPO / 'assets/train/visible_undirected.npz')
    degree = np.diff(und['indptr']).astype(np.int64)
    bounds = np.quantile(degree, np.linspace(0.1, 0.9, DEGREE_BUCKETS - 1))
    bucket = np.clip(np.searchsorted(bounds, degree, side='right'), 0, DEGREE_BUCKETS - 1).astype(np.int64)

    keys_l: list[np.ndarray] = []; others_l: list[np.ndarray] = []
    centres_l: list[np.ndarray] = []; curvs_l: list[np.ndarray] = []
    rows_total = 0
    for name in SHARDS:
        blob = np.load(REPO / 'assets/shards' / name)
        m = blob['centre'].astype(np.int64); b = blob['b'].astype(np.int64); c = blob['c'].astype(np.int64)
        rows_total += len(m)
        dist = blob['bc_dist_sq'].astype(np.int64); curv = blob['curvature'].astype(np.float64)
        keep = dist == 4
        if not keep.any():
            continue
        mm, bb, cc, kk = m[keep], b[keep], c[keep], curv[keep]
        dir_mb = has_edge(forward, mm, bb).astype(np.int64) + 2 * has_edge(reverse, bb, mm).astype(np.int64)
        dir_mc = has_edge(forward, mm, cc).astype(np.int64) + 2 * has_edge(reverse, cc, mm).astype(np.int64)
        multi = np.minimum(dir_mb, dir_mc) * 4 + np.maximum(dir_mb, dir_mc)
        for anchor, other in ((bb, cc), (cc, bb)):
            keys_l.append(anchor * (16 * DEGREE_BUCKETS) + multi * DEGREE_BUCKETS + bucket[anchor])
            others_l.append(other); centres_l.append(mm); curvs_l.append(kk)
    key = np.concatenate(keys_l); other = np.concatenate(others_l)
    centre = np.concatenate(centres_l); curv = np.concatenate(curvs_l)
    del keys_l, others_l, centres_l, curvs_l

    gid, inverse = np.unique(key, return_inverse=True)
    order_hi = np.lexsort((-curv, inverse))
    order_lo = np.lexsort((curv, inverse))
    counts = np.bincount(inverse, minlength=len(gid))
    starts = np.zeros(len(gid), dtype=np.int64); np.cumsum(counts[:-1], out=starts[1:])
    first_hi = order_hi[starts]
    first_lo = order_lo[starts]
    second_lo = np.full(len(gid), -1, dtype=np.int64)
    second_lo[counts > 1] = order_lo[starts[counts > 1] + 1]

    pos_other = other[first_hi]; pos_centre = centre[first_hi]; pos_curv = curv[first_hi]
    neg_other = other[first_lo]; neg_centre = centre[first_lo]; neg_curv = curv[first_lo]
    clash = neg_centre == pos_centre
    alt = second_lo
    use_alt = clash & (alt >= 0)
    alt_other = np.where(use_alt, other[np.maximum(alt, 0)], neg_other)
    alt_centre = np.where(use_alt, centre[np.maximum(alt, 0)], neg_centre)
    alt_curv = np.where(use_alt, curv[np.maximum(alt, 0)], neg_curv)
    keep = (~clash) | use_alt
    keep &= alt_centre != pos_centre
    keep &= alt_other != pos_other
    keep &= pos_other != neg_other
    keep &= (pos_curv > alt_curv) | (~use_alt)     # a tied extremum drops the group

    anchor = (gid[keep] // (16 * DEGREE_BUCKETS)).astype(np.int64)
    pos = pos_other[keep].astype(np.int64)
    neg = alt_other[keep].astype(np.int64)
    pos_c = pos_curv[keep]; neg_c = alt_curv[keep]
    order = np.argsort(anchor, kind='stable')
    anchor, pos, neg, pos_c, neg_c = anchor[order], pos[order], neg[order], pos_c[order], neg_c[order]
    holdout = (anchor % HOLDOUT_MODULUS) == 0

    path = OUT / 'pairs_same_anchor_r5.npz'
    np.savez(path, anchor=anchor, pos=pos, neg=neg, pos_curvature=pos_c, neg_curvature=neg_c,
             is_holdout=holdout)
    manifest = {
        'schema': 'genrec.issue287.same_anchor_r5/1',
        'path': str(path), 'sha256': sha256(path),
        'rules': {
            'positive': 'a distance-two item that shares a neighbour with the anchor (behavioural '
                        'co-occurrence) and is not adjacent to it',
            'negative': 'a distance-two item whose centre differs from the positive\'s centre',
            'comparability_group': 'the anchor, the unordered multiset of the two centre-edge '
                                   'direction flags and the anchor\'s degree decile',
            'curvature_role': 'offline selection inside the group only; never in the loss, the margin '
                              'or a weight',
            'holdout': f'anchor % {HOLDOUT_MODULUS} == 0',
            'false_negative_note': 'an unobserved pair is a relative-association label, not a proof '
                                    'of irrelevance',
        },
        'counts': {'rows_scanned': int(rows_total), 'groups': int(len(gid)),
                   'groups_kept': int(keep.sum()),
                   'anchors': int(len(np.unique(anchor))),
                   'train_groups': int((~holdout).sum()), 'holdout_groups': int(holdout.sum()),
                   'items_touched': int(len(np.unique(np.concatenate([anchor, pos, neg]))))},
    }
    (OUT / 'pairs_same_anchor_r5.manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps(manifest['counts'], indent=1))


if __name__ == '__main__':
    main()
