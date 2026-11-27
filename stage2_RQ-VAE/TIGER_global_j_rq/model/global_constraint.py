"""Issue #271: global behavioural-distance constraint with a non-negative dual.

The global ruler is the Euclidean member of the frozen #269 coordinate map:
``w = q / (2 S*)`` and the kappa-stereographic distance carries a factor
``2`` (``d -> 2 ||w_i - w_j||`` as ``kappa -> 0``), so a perfect fit to a graph
distance ``g`` gives ``||q_i - q_j|| / (S* g) = 1``.  ``S*`` is the scale of the
frozen #269 kappa cache; the common Euclidean ruler is therefore ``d = ||q_i -
q_j|| / S*`` and any extra factor is a double scaling.  The local and global
terms share that ruler and the same hard ``q``.

The pair table is a fixed, centre-uniform stratified sample of the ordered
pairs of the visible training graph.  For a centre ``i`` and every distance
layer ``d`` one partner is drawn uniformly; the layer count is the weight, so
``sum_i sum_d count(i,d) * err(i, partner)`` is an unbiased estimate of the
ordered-pair sum, and the graph is undirected so that equals the unordered-pair
mean.  The sample is built once, stored as a result artifact, and its coverage
is written next to it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from tools.iteration.depgraph import file_sha  # noqa: F401  (identity helper)

SCHEMA = "genrec.issue271.global_pairs/1"
GRAPH = "assets/train/visible_undirected.npz"
KAPPA_CACHE = "results/stage2_RQ-VAE/TIGER_calibrated_kappa_rq/kappa_cache_s0p068284"
N_ITEMS = 24587
PAIR_CENTRES = 2048
PAIR_SEED = 271
RULER_FACTOR = 1.0
TRAIN_PAIRS_PER_STEP = 512
DUAL_STEP = 0.1
DUAL_EVERY = 50
DUAL_CAP = 10.0
# Frozen after calibrate_threshold.py; literals are the contract, not defaults.
# Frozen by results/research/issue271/calibrate_threshold.py before training:
# full-graph J on the frozen TIGER checkpoint, epsilon = max(1e-4, 2*bootstrap SE,
# |sample-full|) of the fixed pair estimator, and J_max = J_TIGER + epsilon.
J_TIGER = 0.739634552674212
EPSILON = 0.04230598283915425
J_MAX = 0.7819405355133663


def repository_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "stage2_RQ-VAE").is_dir() and (parent / "tools/stage2/server.py").is_file():
            return parent
    raise RuntimeError("cannot locate the repository root")


REPO = repository_root()


def load_graph():
    import scipy.sparse as sp

    z = np.load(REPO / GRAPH, allow_pickle=False)
    if set(z.files) != {"indptr", "indices", "data", "shape"}:
        raise ValueError("unexpected graph archive keys")
    shape = tuple(int(v) for v in z["shape"])
    graph = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=shape)
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    if graph.shape != (N_ITEMS, N_ITEMS):
        raise ValueError("graph shape is not the frozen catalogue")
    return graph


def kappa_scale() -> float:
    global _KAPPA_SCALE
    if _KAPPA_SCALE is None:
        manifest = json.loads((REPO / KAPPA_CACHE / "manifest.json").read_text())
        _KAPPA_SCALE = float(manifest["scale"])
    return _KAPPA_SCALE


_KAPPA_SCALE = None


def _parse_graph_distances(graph, rows):
    from scipy.sparse.csgraph import shortest_path

    return shortest_path(graph, unweighted=True, directed=False, indices=np.asarray(rows))


def build_pair_table(graph, *, centres: int = PAIR_CENTRES, seed: int = PAIR_SEED):
    """Fixed centre-uniform stratified pairs with layer-count weights."""
    rng = np.random.default_rng(seed)
    nodes = np.flatnonzero(np.asarray(graph.sum(axis=1)).ravel() > 0)
    chosen = np.sort(rng.choice(nodes, min(centres, len(nodes)), replace=False))
    ii, jj, gg, ww = [], [], [], []
    chunk = 256
    for start in range(0, len(chosen), chunk):
        block = chosen[start:start + chunk]
        distances = _parse_graph_distances(graph, block)
        for row, centre in enumerate(block):
            values = distances[row]
            finite = np.flatnonzero(np.isfinite(values))
            finite = finite[finite != centre]
            if finite.size == 0:
                continue
            layers = values[finite].astype(np.int64)
            for distance in np.unique(layers):
                members = finite[layers == distance]
                partner = int(rng.choice(members))
                ii.append(int(centre))
                jj.append(partner)
                gg.append(int(distance))
                ww.append(int(members.size))
    table = {
        "i": np.asarray(ii, dtype=np.int64),
        "j": np.asarray(jj, dtype=np.int64),
        "g": np.asarray(gg, dtype=np.float64),
        "weight": np.asarray(ww, dtype=np.float64),
        "stratum": np.asarray(gg, dtype=np.int64),
    }
    if len(table["i"]) == 0:
        raise ValueError("empty pair table")
    unique = len({(int(a), int(b)) for a, b in zip(table["i"], table["j"])})
    receipt = {
        "schema": SCHEMA,
        "source_graph": GRAPH,
        "source_graph_sha256": file_sha(REPO / GRAPH),
        "kappa_cache": KAPPA_CACHE,
        "kappa_scale": kappa_scale(),
        "ruler_factor": RULER_FACTOR,
        "seed": seed,
        "centres": int(len(chosen)),
        "pairs": int(len(table["i"])),
        "unique_pairs": int(unique),
        "duplicate_fraction": float(1 - unique / len(table["i"])),
        "coverage": {str(int(d)): int(c) for d, c in zip(*np.unique(table["stratum"], return_counts=True))},
        "weight_sum": float(table["weight"].sum()),
        "ordered_pairs_estimate": float(table["weight"].sum()),
        "sample_fraction": float(table["weight"].sum() / (N_ITEMS * (N_ITEMS - 1))),
        "note": "centre-uniform ordered pairs; weight is the layer count, so the weighted mean "
                "estimates the ordered-pair mean; the graph is undirected.",
    }
    return table, receipt


def save_pair_table(root: Path, table, receipt) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    np.savez(root / "global_pairs.npz", **table)
    path = root / "global_pairs.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return root / "global_pairs.npz"


def load_pair_table(root: Path):
    root = Path(root)
    z = np.load(root / "global_pairs.npz")
    table = {key: z[key] for key in ("i", "j", "g", "weight", "stratum")}
    receipt = json.loads((root / "global_pairs.json").read_text())
    if receipt["source_graph_sha256"] != file_sha(REPO / GRAPH):
        raise ValueError("pair table was built from a different graph")
    return table, receipt


def dual_step(mu: float, estimate: float, *, step: float = DUAL_STEP, cap: float = DUAL_CAP,
              j_max: float | None = None) -> float:
    """Non-negative dual ascent on J - J_max; satisfies the KKT sign rule."""
    if not np.isfinite(estimate):
        raise ValueError("non-finite J estimate")
    bound = J_MAX if j_max is None else float(j_max)
    if not np.isfinite(bound):
        raise ValueError("global J_max is not frozen")
    value = max(0.0, mu + step * (estimate - bound))
    return min(cap, value)


def sample_pairs(table, *, count: int, rng: np.random.Generator):
    n = len(table["i"])
    index = rng.choice(n, min(count, n), replace=False)
    return {key: value[index] for key, value in table.items()}


def weighted_j(q: torch.Tensor, sample: dict) -> torch.Tensor:
    """J on the hard q for one pair sample; differentiable through q."""
    device = q.device
    i = torch.as_tensor(sample["i"], dtype=torch.long, device=device)
    j = torch.as_tensor(sample["j"], dtype=torch.long, device=device)
    g = torch.as_tensor(sample["g"], dtype=q.dtype, device=device)
    w = torch.as_tensor(sample["weight"], dtype=q.dtype, device=device)
    delta = q[i] - q[j]
    # linalg.vector_norm has a zero subgradient at the coincident point; a
    # naive sqrt(0) would produce a NaN gradient for identical q.
    distance = torch.linalg.vector_norm(delta, dim=-1)
    ratio = distance / (RULER_FACTOR * kappa_scale() * g)
    error = (ratio - 1.0).square()
    return (w * error).sum() / w.sum()


def full_j(q: np.ndarray, graph, *, chunk: int = 256, bins=(1, 2, 3, 4, 6, 1 << 30)):
    """Exact full-graph J and distance-stratum tail report for a hard q matrix."""
    q = np.asarray(q, dtype=np.float64)
    n_items = q.shape[0]
    if graph.shape != (n_items, n_items):
        raise ValueError("q does not cover the graph catalogue")
    scale = RULER_FACTOR * kappa_scale()
    sums = {label: [0.0, 0] for label in range(len(bins) - 1)}
    total = [0.0, 0]
    for start in range(0, n_items, chunk):
        rows = np.arange(start, min(start + chunk, n_items))
        distances = _parse_graph_distances(graph, rows)
        dq = np.sqrt(np.maximum(
            (q[rows] ** 2).sum(1)[:, None] + (q ** 2).sum(1)[None, :]
            - 2.0 * q[rows] @ q.T, 0.0))
        valid = np.isfinite(distances) & (distances > 0)
        ratio = dq / (scale * np.where(valid, distances, 1.0))
        error = (ratio - 1.0) ** 2
        upper = rows[:, None] < np.arange(n_items)[None, :]
        mask = valid & upper
        total[0] += float(error[mask].sum())
        total[1] += int(mask.sum())
        for b in range(len(bins) - 1):
            band = mask & (distances >= bins[b]) & (distances < bins[b + 1])
            sums[b][0] += float(error[band].sum())
            sums[b][1] += int(band.sum())
    if total[1] == 0:
        raise ValueError("no valid pairs")
    strata = {}
    for b in range(len(bins) - 1):
        label = "%d-%s" % (bins[b], "inf" if bins[b + 1] >= 1 << 30 else bins[b + 1] - 1)
        strata[label] = {"sum": sums[b][0], "pairs": sums[b][1],
                         "mean_j": sums[b][0] / sums[b][1] if sums[b][1] else None,
                         "share": sums[b][1] / total[1]}
    return {"mean_j": total[0] / total[1], "pairs": total[1], "sum": total[0],
            "scale": scale, "strata": strata}


def load_hard_q(sid_path: Path, checkpoint_path: Path, *, levels: int = 3) -> np.ndarray:
    """q = sum of the exported hard codes' codewords; the first three levels only."""
    sid = json.loads(Path(sid_path).read_text())
    codes = np.asarray([sid[str(i)] for i in range(N_ITEMS)], dtype=np.int64)[:, :levels]
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    books = []
    for level in range(levels):
        key = (f"rq.vq_layers.{level}.embed.weight" if f"rq.vq_layers.{level}.embed.weight" in state
               else f"rq.levels.{level}.embed.weight")
        books.append(state[key].detach().cpu().numpy().astype(np.float64))
    q = np.zeros((N_ITEMS, books[0].shape[1]), dtype=np.float64)
    for level, book in enumerate(books):
        q += book[codes[:, level]]
    return q
