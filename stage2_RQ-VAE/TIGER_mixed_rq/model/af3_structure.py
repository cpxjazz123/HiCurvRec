"""The frozen AF3 target, the fixed support, the shared readout and the distortion.

This module is the *outer* half of the mechanism: it never trains anything and
never touches the optimizer. It answers one question only -- given a complete
set of hard SIDs, how much of the training graph's local structure (degree and
triangles) does the quantized readout still explain -- and it answers it with
quantities that are frozen before any arm runs:

* the graph is the training co-occurrence graph, undirected and de-duplicated;
* the target ``kappa_ij = 4 - d_i - d_j + 3 T_ij`` uses the degree and the
  triangle count of that graph, never of a SID;
* the support ``S`` is fixed once (the edges, the semantic k-nearest candidates
  and a degree-stratified sample of unobserved pairs), and ``p = 0`` outside it;
* the readout ``1 - cos(phi_q)`` is shared by every arm, with ``tau`` and the
  intercept ``b`` calibrated once on the F0 snapshot.

The distortion is the class-balanced Huber of the Bernoulli expectation of the
degree-and-triangle target. Under the per-edge Bernoulli readout the expectation
of ``kappa`` for a pair is ``p (2 - sum_k p_ik - sum_k p_jk + 3 sum_k p_ik p_jk)``;
the constant is 2 because the conditional edge (i, j) is already counted once in
each of the two degrees. This is a modelling assumption about the readout, not a
claim that user graphs are independent.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

#: The semantic neighbour count of the support's candidate edges.
SEMANTIC_CANDIDATES = 16
#: The seed of every sampled set. One seed: the support is frozen, not resampled.
SUPPORT_SEED = 42
#: The numeric floor of the temperature.
TAU_FLOOR = 1e-6
#: The numeric tolerance of the intercept calibration and of an accepted step.
CALIBRATION_TOL = 1e-8


def load_graph(path: Path) -> dict:
    """The frozen training graph: undirected, de-duplicated, weighted by AF3."""
    payload = np.load(Path(path))
    rows = np.asarray(payload["rows"], dtype=np.int64)
    cols = np.asarray(payload["cols"], dtype=np.int64)
    weights = np.asarray(payload["af3"], dtype=np.float64)
    n_nodes = int(max(int(payload["nodes"]) if "nodes" in payload.files else 0,
                      int(rows.max()) + 1, int(cols.max()) + 1))
    return {"rows": rows, "cols": cols, "weights": weights, "n_nodes": n_nodes,
            "af3_median": float(payload["median"]), "af3_p90": float(payload["p90"])}


def _neighbours(rows: np.ndarray, cols: np.ndarray, n_nodes: int) -> list[np.ndarray]:
    """Sorted adjacency, undirected."""
    pairs = np.concatenate([np.stack([rows, cols], 1), np.stack([cols, rows], 1)], 0)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    pairs = pairs[order]
    keep = np.ones(len(pairs), dtype=bool)
    keep[1:] = np.any(pairs[1:] != pairs[:-1], axis=1)
    pairs = pairs[keep]
    counts = np.bincount(pairs[:, 0], minlength=n_nodes)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    flat = pairs[:, 1]
    return [flat[starts[i]:starts[i] + counts[i]] for i in range(n_nodes)]


def target_kappa(graph: dict) -> dict:
    """``kappa_ij = 4 - d_i - d_j + 3 T_ij`` for every observed edge."""
    rows, cols = graph["rows"], graph["cols"]
    n_nodes = graph["n_nodes"]
    adjacency = _neighbours(rows, cols, n_nodes)
    degree = np.array([len(a) for a in adjacency], dtype=np.float64)
    triangles = np.zeros(len(rows), dtype=np.float64)
    for index, (i, j) in enumerate(zip(rows, cols)):
        a, b = adjacency[i], adjacency[j]
        if a.size == 0 or b.size == 0:
            continue
        start = np.searchsorted(b, a)
        start = np.clip(start, 0, b.size - 1)
        triangles[index] = float(np.sum(b[start] == a))
    kappa = 4.0 - degree[rows] - degree[cols] + 3.0 * triangles
    return {"rows": rows, "cols": cols, "kappa": kappa, "degree": degree,
            "triangles": triangles, "adjacency": adjacency}


def build_support(graph: dict, kappa: dict, embeddings: np.ndarray | None = None,
                  k: int = SEMANTIC_CANDIDATES, seed: int = SUPPORT_SEED) -> dict:
    """The fixed support: the edges, the semantic candidates and the stratified sample.

    Every unobserved group has the size of the edge set, drawn from the TRAIN
    degree strata, so the two classes carry the same weight in the distortion.
    The result records what it actually built, including the per-stratum counts.
    """
    rows, cols = graph["rows"], graph["cols"]
    n_nodes = graph["n_nodes"]
    observed = set(int(value) for value in rows * n_nodes + cols)
    pairs = [(int(i), int(j)) for i, j in zip(rows, cols)]
    seen = set(int(i) * n_nodes + int(j) for i, j in pairs)

    semantic_added = 0
    if embeddings is not None:
        # Only the training items are nodes of the graph: the validation and
        # test items have embeddings too, and a candidate edge that touches one
        # of them would put held-out items into the support.
        matrix = np.asarray(embeddings, dtype=np.float32)[:n_nodes]
        norm = np.linalg.norm(matrix, axis=1, keepdims=True).clip(1e-8)
        unit = matrix / norm
        neighbour_index = []
        chunk = 2048
        for start in range(0, unit.shape[0], chunk):
            block = unit[start:start + chunk] @ unit.T
            block[np.arange(block.shape[0]), np.arange(start, start + block.shape[0])] = -2.0
            neighbour_index.append(np.argpartition(-block, k, axis=1)[:, :k])
        neighbours = np.concatenate(neighbour_index, 0)
        for item in range(unit.shape[0]):
            for partner in neighbours[item]:
                a, b = int(item), int(partner)
                if a == b:
                    continue
                key = min(a, b) * n_nodes + max(a, b)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append((min(a, b), max(a, b)))
                semantic_added += 1

    rng = np.random.default_rng(seed)
    degree = kappa["degree"]
    strata = np.digitize(degree, np.quantile(degree, [0.25, 0.5, 0.75]))
    target = len(rows)
    per_stratum = np.bincount(strata, minlength=4)
    drawn = 0
    strata_counts = {}
    attempts = 0
    max_attempts = 80 * target
    while drawn < target and attempts < max_attempts:
        item = int(rng.integers(0, n_nodes))
        partner = int(rng.integers(0, n_nodes))
        attempts += 1
        if item == partner:
            continue
        a, b = (item, partner) if item < partner else (partner, item)
        key = a * n_nodes + b
        if key in seen:
            continue
        seen.add(key)
        pairs.append((a, b))
        drawn += 1
        bucket = int(strata[a])
        strata_counts[bucket] = strata_counts.get(bucket, 0) + 1

    pair_array = np.array(pairs, dtype=np.int64)
    is_edge = np.zeros(len(pair_array), dtype=bool)
    edge_index = {}
    for index, (i, j) in enumerate(zip(rows, cols)):
        edge_index[(int(i), int(j))] = index
    for position, (i, j) in enumerate(pair_array):
        is_edge[position] = (int(i), int(j)) in edge_index
    return {
        "pairs": pair_array,
        "is_edge": is_edge,
        "edges": int(is_edge.sum()),
        "unobserved": int((~is_edge).sum()),
        "semantic_added": semantic_added,
        "sampled_unobserved": drawn,
        "attempts": attempts,
        "strata_counts": strata_counts,
        "strata_population": {str(i): int(value) for i, value in enumerate(per_stratum)},
        "seed": seed,
        "k": k,
    }


def readout_delta(phi: torch.Tensor, pairs: np.ndarray, *, device=None) -> torch.Tensor:
    """``1 - cos(phi_i, phi_j)`` for every supported pair."""
    index = torch.as_tensor(pairs.T, dtype=torch.long, device=device or phi.device)
    left = phi[index[0]]
    right = phi[index[1]]
    # The norm denominator is clamped exactly as the spec says.
    denominator = (left.norm(dim=-1) * right.norm(dim=-1)).clamp_min(1e-8)
    cosine = (left * right).sum(-1) / denominator
    return 1.0 - cosine


def readout_probability(delta: torch.Tensor, tau: float, intercept: float) -> torch.Tensor:
    """``p = sigmoid(b - delta / tau)``."""
    return torch.sigmoid(intercept - delta / max(float(tau), TAU_FLOOR))


def calibrate(phi: torch.Tensor, support: dict, *, device=None):
    """The frozen ``(tau, b)``: the IQR of delta and the intercept that matches density.

    ``tau`` is the inter-quartile range of the readout on the whole support; the
    intercept is the single scalar that makes the mean predicted edge
    probability equal the true edge density ``|E| / |S|``. Both are computed on
    one snapshot and then shared, never re-fitted per arm.
    """
    delta = readout_delta(phi, support["pairs"], device=device)
    quantiles = torch.quantile(delta, torch.tensor([0.25, 0.75], dtype=delta.dtype,
                                                   device=delta.device))
    tau = float(max(float(quantiles[1] - quantiles[0]), TAU_FLOOR))
    density = float(support["edges"]) / float(len(support["pairs"]))

    def mean_probability(intercept: float) -> float:
        return float(readout_probability(delta, tau, intercept).mean())

    low, high = -60.0, 60.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        if mean_probability(middle) < density:
            low = middle
        else:
            high = middle
        if high - low < CALIBRATION_TOL:
            break
    intercept = 0.5 * (low + high)
    return {"tau": tau, "intercept": intercept,
            "mean_probability": mean_probability(intercept),
            "density": density}


def soft_structure(support: dict, probability: torch.Tensor, n_nodes: int):
    """Per-node soft degree over the support, with ``p = 0`` outside it."""
    p = probability.detach().cpu().numpy()
    pairs = support["pairs"]
    degree = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(degree, pairs[:, 0], p)
    np.add.at(degree, pairs[:, 1], p)
    return degree, p


def uncertainty_term(pairs: np.ndarray, p: np.ndarray, n_nodes: int) -> np.ndarray:
    """``2 - soft_degree_i - soft_degree_j + 3 sum_k p_ik p_jk`` for every pair.

    The triangle sum runs over the support only, because ``p = 0`` outside it.
    """
    degree = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(degree, pairs[:, 0], p)
    np.add.at(degree, pairs[:, 1], p)
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n_nodes)]
    for position, (i, j) in enumerate(pairs):
        adjacency[i].append((int(j), float(p[position])))
        adjacency[j].append((int(i), float(p[position])))
    dictionaries = []
    for entries in adjacency:
        entries.sort()
        dictionaries.append(dict(entries))
    # The conditional edge (i, j) is already counted in both degrees, so the
    # degree sums here exclude the pair itself: with binary probabilities this
    # gives mu = 4 - d_i - d_j + 3 T_ij, the target, exactly.
    own = p
    term = 2.0 - (degree[pairs[:, 0]] - own) - (degree[pairs[:, 1]] - own)
    for position in range(len(pairs)):
        i, j = int(pairs[position, 0]), int(pairs[position, 1])
        left, right = dictionaries[i], dictionaries[j]
        if len(left) > len(right):
            left, right = right, left
        total = 0.0
        for partner, value in left.items():
            other = right.get(partner)
            if other is not None:
                total += value * other
        term[position] += 3.0 * total
    return term


def huber(value: np.ndarray, delta: float = 1.0) -> np.ndarray:
    absolute = np.abs(value)
    return np.where(absolute <= delta, 0.5 * value * value, delta * (absolute - 0.5 * delta))


def structural_distortion(phi: torch.Tensor, support: dict, kappa: dict, *,
                          tau: float, intercept: float, n_nodes: int,
                          scale: float | None = None, device=None) -> dict:
    """``D_AF3``: the class-balanced Huber of the expectation surrogate."""
    delta = readout_delta(phi, support["pairs"], device=device)
    probability = readout_probability(delta, tau, intercept)
    _, p = soft_structure(support, probability, n_nodes)
    term = uncertainty_term(support["pairs"], p, n_nodes)
    predicted = p * term
    is_edge = support["is_edge"]
    edge_kappa = np.zeros(len(p), dtype=np.float64)
    lookup = {}
    for index, (i, j) in enumerate(zip(kappa["rows"], kappa["cols"])):
        lookup[(int(i), int(j))] = float(kappa["kappa"][index])
    for position, (i, j) in enumerate(support["pairs"]):
        value = lookup.get((int(i), int(j)))
        if value is not None:
            edge_kappa[position] = value
    if scale is None:
        scale = float(max(1.0, float(np.subtract(*np.quantile(kappa["kappa"], [0.75, 0.25])))))
    edge_half = float(huber((predicted[is_edge] - edge_kappa[is_edge]) / scale).mean()) if is_edge.any() else 0.0
    other_half = float(huber(predicted[~is_edge] / scale).mean()) if (~is_edge).any() else 0.0
    return {"distortion": 0.5 * edge_half + 0.5 * other_half,
            "edge_half": edge_half, "unobserved_half": other_half,
            "scale": scale,
            "probability": probability.detach().cpu().numpy(),
            "predicted": predicted}
