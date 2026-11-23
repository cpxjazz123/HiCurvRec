"""The graph-side curvature target K_G for issue #238.

Only train interactions enter (the canonical behaviour graph is built from train.parquet), and the
target is the geodesic four-point sectional curvature proxy of #236:

    K_G(m; b, c) = mean_a [ d(a,m)^2 + d(b,c)^2/4 - (d(a,b)^2 + d(a,c)^2)/2 ]

with `m` adjacent to both `b` and `c` and `d_G(b,c) = 2`, so the middle term is exactly 1.
The same reference set `A` is used for every triplet, which is what makes the values comparable
across triplets; an unreachable `a` is dropped from the mean rather than entering as a zero
distance — a zero would silently claim "a is m", which is the opposite of the truth.

Hub domination is bounded by `max_per_node`: a node contributes at most that many triplets, so a
hub with 2 220 neighbours cannot outvote the rest of the catalogue.
"""
from __future__ import annotations

from collections import deque

import numpy as np

#: Triplets whose |K_G| is below this are reported as the "near zero" stratum.
NEAR_ZERO = 1e-3


def reference_nodes(n_items: int, n_refs: int = 64, seed: int = 42) -> np.ndarray:
    """A fixed, reproducible sample of nodes used as `a` for every triplet."""
    rng = np.random.default_rng(seed)
    n_refs = int(min(n_refs, n_items))
    return np.sort(rng.choice(np.arange(n_items), n_refs, replace=False))


def _neighbours(graph) -> list[np.ndarray]:
    return [graph.indices[graph.indptr[i]:graph.indptr[i + 1]] for i in range(graph.shape[0])]


def graph_distance_pair(graph, source: int, target: int, limit: int = 8) -> int:
    """Breadth-first distance with a small depth limit; `-1` when unreachable within it."""
    if source == target:
        return 0
    seen = {int(source)}
    frontier = deque([(int(source), 0)])
    while frontier:
        node, depth = frontier.popleft()
        if depth >= limit:
            continue
        start, end = graph.indptr[node], graph.indptr[node + 1]
        for nb in graph.indices[start:end]:
            nb = int(nb)
            if nb == target:
                return depth + 1
            if nb not in seen:
                seen.add(nb)
                frontier.append((nb, depth + 1))
    return -1


def build_triplets(graph, max_per_node: int = 8, seed: int = 42, attempts: int = 40) -> dict:
    """Collect `(m; b, c)` with `m` adjacent to both and `d_G(b,c) = 2`.

    Candidate pairs are sampled from m's neighbourhood and *verified* against the adjacency lists,
    so no triplet is accepted on a probabilistic argument.  At most `max_per_node` per m.
    """
    rng = np.random.default_rng(seed)
    neighbours = _neighbours(graph)
    ms: list[int] = []
    bs: list[int] = []
    cs: list[int] = []
    for m in range(graph.shape[0]):
        ring = neighbours[m]
        if ring.size < 2:
            continue
        seen_pairs: set[tuple[int, int]] = set()
        taken = 0
        for _ in range(attempts):
            if taken >= max_per_node:
                break
            b, c = rng.choice(ring, 2, replace=False)
            b, c = int(b), int(c)
            key = (min(b, c), max(b, c))
            if key in seen_pairs:                # the same unordered pair is one triplet, not many
                continue
            set_b = set(int(x) for x in neighbours[b])
            set_c = set(int(x) for x in neighbours[c])
            if c in set_b:                       # adjacent -> d = 1, not a geodesic triplet
                continue
            if not (set_b & set_c):              # no common neighbour -> d >= 3
                continue
            seen_pairs.add(key)
            ms.append(m)
            bs.append(b)
            cs.append(c)
            taken += 1
    return {"m": np.asarray(ms, np.int64), "b": np.asarray(bs, np.int64),
            "c": np.asarray(cs, np.int64)}


def graph_distances(graph, refs: np.ndarray) -> np.ndarray:
    """`(len(refs), n_nodes)` unweighted BFS distances; `inf` where unreachable."""
    n = graph.shape[0]
    out = np.full((len(refs), n), np.inf, dtype=np.float64)
    neighbours = _neighbours(graph)
    for row, source in enumerate(refs):
        source = int(source)
        out[row, source] = 0.0
        frontier = deque([source])
        while frontier:
            node = frontier.popleft()
            nxt = out[row, node] + 1.0
            for nb in neighbours[node]:
                if np.isinf(out[row, nb]):
                    out[row, nb] = nxt
                    frontier.append(int(nb))
    return out


def sectional_curvature(dists: np.ndarray, triplets: dict, refs: np.ndarray) -> np.ndarray:
    """`K_G` per triplet, averaging only the references that reach all three nodes."""
    n_t = len(triplets["m"])
    out = np.full(n_t, np.nan, dtype=np.float64)
    for t in range(n_t):
        m, b, c = int(triplets["m"][t]), int(triplets["b"][t]), int(triplets["c"][t])
        dm = dists[:, m]
        db = dists[:, b]
        dc = dists[:, c]
        ok = np.isfinite(dm) & np.isfinite(db) & np.isfinite(dc)
        if not ok.any():
            continue                      # no legal reference: the triplet is unusable, not zero
        d_bc = 2.0                        # by construction of the triplet
        out[t] = float(np.mean(dm[ok] ** 2 + d_bc ** 2 / 4.0
                               - (db[ok] ** 2 + dc[ok] ** 2) / 2.0))
    return out


def strata(k_g: np.ndarray) -> dict:
    """Positive / negative / near-zero split, frozen before training."""
    finite = np.isfinite(k_g)
    return {
        "positive": finite & (k_g > NEAR_ZERO),
        "negative": finite & (k_g < -NEAR_ZERO),
        "near_zero": finite & (np.abs(k_g) <= NEAR_ZERO),
        "unusable": ~finite,
    }


def midpoint_deviation(coords, triplets: dict, distance_fn=None) -> dict:
    """Diagnostic only: the three local distances and their ratio.  Never a loss term."""
    import torch

    if distance_fn is None:
        def distance_fn(x, y):
            return torch.linalg.norm(x - y, dim=-1)

    m = torch.as_tensor(triplets["m"], dtype=torch.long)
    b = torch.as_tensor(triplets["b"], dtype=torch.long)
    c = torch.as_tensor(triplets["c"], dtype=torch.long)
    coords = torch.as_tensor(coords, dtype=torch.float32)
    d_bm = distance_fn(coords[b], coords[m])
    d_mc = distance_fn(coords[m], coords[c])
    d_bc = distance_fn(coords[b], coords[c])
    ratio = d_bc / (d_bm + d_mc).clamp_min(1e-12)
    return {"d_bm": d_bm, "d_mc": d_mc, "d_bc": d_bc, "ratio": ratio}
