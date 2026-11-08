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
    with np.load(Path(path), allow_pickle=False) as payload:
        if "nodes" not in payload.files:
            raise ValueError("AF3 cache must declare its graph-position row count")
        rows, cols = np.asarray(payload["rows"]), np.asarray(payload["cols"])
        nodes = np.asarray(payload["nodes"])
        if nodes.ndim != 0 or not np.issubdtype(nodes.dtype, np.integer):
            raise ValueError("AF3 node count must be an integer scalar")
        n_nodes = int(nodes)
        _validate_graph_edges(rows, cols, n_nodes)
        weights = np.asarray(payload["af3"], dtype=np.float64)
        if weights.shape != rows.shape or not np.isfinite(weights).all():
            raise ValueError("AF3 weights must be finite and aligned with graph edges")
        median, p90 = float(payload["median"]), float(payload["p90"])
        if not np.isfinite([median, p90]).all():
            raise ValueError("AF3 summary statistics must be finite")
        return {"rows": rows.astype(np.int64), "cols": cols.astype(np.int64),
                "weights": weights, "n_nodes": n_nodes,
                "af3_median": median, "af3_p90": p90}


def _validate_graph_edges(rows, cols, n_nodes):
    if (rows.ndim != 1 or cols.shape != rows.shape or not len(rows)
            or not np.issubdtype(rows.dtype, np.integer)
            or not np.issubdtype(cols.dtype, np.integer) or n_nodes <= 0):
        raise ValueError("graph requires non-empty aligned integer endpoint vectors")
    if np.any(rows < 0) or np.any(cols >= n_nodes) or np.any(rows >= cols):
        raise ValueError("graph requires in-range canonical pairs with row < col")
    keys = rows.astype(np.int64) * int(n_nodes) + cols.astype(np.int64)
    if len(np.unique(keys)) != len(keys):
        raise ValueError("duplicate graph edges would change AF3 targets")


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

    Requested random non-edges total |E|, with TRAIN-degree quartile quotas
    proportional to node populations (integer largest remainder; lower-bin
    tie break). This makes stratification real rather than merely labelling
    uniform draws afterwards. Underfilled strata are recorded, not reallocated.
    The new support must be reviewed/refrozen before formal training.
    """
    rows, cols = graph["rows"], graph["cols"]
    n_nodes = graph["n_nodes"]
    _validate_graph_edges(rows, cols, n_nodes)
    pairs = [(int(i), int(j)) for i, j in zip(rows, cols)]
    seen = set(int(i) * n_nodes + int(j) for i, j in pairs)

    semantic_added = 0
    effective_k = min(max(int(k), 0), n_nodes - 1)
    if embeddings is not None:
        # The caller must index by the VERIFIED graph-position -> item map.
        # Truncating catalog embeddings to n_nodes silently selects wrong items.
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or len(matrix) != n_nodes or not np.isfinite(matrix).all():
            raise ValueError("support embeddings must be finite and exactly graph-row aligned")
        norm = np.linalg.norm(matrix, axis=1, keepdims=True).clip(1e-8)
        unit = matrix / norm
        neighbour_index = []
        chunk = 2048
        for start in range(0, unit.shape[0], chunk):
            block = unit[start:start + chunk] @ unit.T
            block[np.arange(block.shape[0]), np.arange(start, start + block.shape[0])] = -2.0
            # Stable sorting makes equal-cosine ties choose the lower node ID.
            neighbour_index.append(np.argsort(-block, axis=1, kind="stable")[:, :effective_k])
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
    quotas = (target * per_stratum) // n_nodes
    priority = sorted(range(4), key=lambda bucket:
                      (-int((target * per_stratum[bucket]) % n_nodes), bucket))
    for bucket in priority[:target - int(quotas.sum())]:
        quotas[bucket] += 1
    drawn = 0
    strata_counts = {str(bucket): 0 for bucket in range(4)}
    anchors = []
    attempts = 0
    for bucket in range(4):
        population = np.flatnonzero(strata == bucket)
        quota, tried = int(quotas[bucket]), 0
        while strata_counts[str(bucket)] < quota and tried < 80 * quota:
            item = int(population[rng.integers(len(population))])
            partner = int(rng.integers(n_nodes))
            tried += 1
            attempts += 1
            if item == partner:
                continue
            a, b = (item, partner) if item < partner else (partner, item)
            key = a * n_nodes + b
            if key in seen:
                continue
            seen.add(key)
            pairs.append((a, b))
            anchors.append(item)
            drawn += 1
            strata_counts[str(bucket)] += 1

    # The order of the support is part of its identity: it is hashed into the
    # shared readout's provenance, and an order that came from set iteration
    # differs between processes even when every pair is the same. Sorting by
    # (left, right) makes the support reproducible from its inputs alone, which
    # is what "a fixed support" has to mean.
    pair_array = np.array(sorted(set(pairs)), dtype=np.int64)
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
        "sampled_unobserved_requested": target,
        "sampled_anchors": np.asarray(anchors, dtype=np.int64),
        "sampling_policy": "train-degree-proportional-largest-remainder-v1",
        "strata_targets": {str(bucket): int(quotas[bucket]) for bucket in range(4)},
        "attempts": attempts,
        "strata_counts": strata_counts,
        "strata_population": {str(i): int(value) for i, value in enumerate(per_stratum)},
        "seed": seed,
        "k": k,
        "effective_k": effective_k,
    }


def readout_delta(phi: torch.Tensor, pairs: np.ndarray, *, device=None) -> torch.Tensor:
    """``1 - cos(phi_i, phi_j)`` for every supported pair."""
    values = phi.to(device=device or phi.device, dtype=torch.float64)
    if not torch.isfinite(values).all():
        raise ValueError("the readout contains NaN or infinity")
    pieces = []
    # Do not allocate two support_size by embedding_dim endpoint matrices.
    for start in range(0, len(pairs), 65536):
        index = torch.as_tensor(pairs[start:start + 65536].T,
                                dtype=torch.long, device=values.device)
        left, right = values[index[0]], values[index[1]]
        denominator = (left.norm(dim=-1) * right.norm(dim=-1)).clamp_min(1e-8)
        pieces.append(1.0 - (left * right).sum(-1) / denominator)
    return torch.cat(pieces) if pieces else values.new_empty(0)


def readout_probability(delta: torch.Tensor, tau: float, intercept: float) -> torch.Tensor:
    """``p = sigmoid(b - delta / tau)``."""
    return torch.sigmoid(intercept - delta / max(float(tau), TAU_FLOOR))


@torch.no_grad()
def calibrate(phi: torch.Tensor, support: dict, *, device=None):
    """The frozen ``(tau, b)``: the IQR of delta and the intercept that matches density.

    ``tau`` is the inter-quartile range of the readout on the whole support; the
    intercept is the single scalar that makes the mean predicted edge
    probability equal the true edge density ``|E| / |S|``. Both are computed on
    one snapshot and then shared, never re-fitted per arm.
    """
    delta = readout_delta(phi, support["pairs"], device=device)
    if not len(delta):
        raise ValueError("cannot calibrate an empty support")
    quantiles = torch.quantile(delta, torch.tensor([0.25, 0.75], dtype=delta.dtype,
                                                   device=delta.device))
    tau = float(max(float(quantiles[1] - quantiles[0]), TAU_FLOOR))
    density = float(support["edges"]) / float(len(support["pairs"]))
    if not 0.0 < density < 1.0:
        raise ValueError("a finite calibration intercept requires both adjacency classes")

    def mean_probability(intercept: float) -> float:
        return float(readout_probability(delta, tau, intercept).mean())

    # If shift=delta/tau, min(shift)+logit(density) brackets the root
    # from below and max(shift)+logit(density) brackets it from above.
    # A fixed [-60,60] interval fails, e.g. delta=1 and tau=1e-6.
    shift = delta / tau
    logit = float(np.log(density) - np.log1p(-density))
    low, high = float(shift.min()) + logit, float(shift.max()) + logit
    for _ in range(200):
        middle = 0.5 * (low + high)
        measured = mean_probability(middle)
        if abs(measured - density) <= CALIBRATION_TOL:
            break
        if measured < density:
            low = middle
        else:
            high = middle
    intercept = middle
    measured = mean_probability(intercept)
    if not np.isfinite(intercept) or abs(measured - density) > CALIBRATION_TOL:
        raise RuntimeError("calibration failed to match the TRAIN edge density")
    return {"tau": tau, "intercept": intercept,
            "mean_probability": measured, "density": density,
            "delta_iqr": float(quantiles[1] - quantiles[0]),
            "temperature_at_floor": bool(tau == TAU_FLOOR)}


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

