"""The frozen local-structure descriptor of the TRAIN behaviour graph (issue #149).

The descriptor is the only behaviour signal this mechanism reads. It is built
once, before training, from the TRAIN co-occurrence graph -- the same graph, with
the same construction and the same AF3 convention, as the frozen cache of issue
#136 -- and it is written to disk with its SHA-256. Training and inference both
read that table, so the same item always gets the same descriptor (CLAUDE.md §17).

Eight dimensions per item, all measured on the TRAIN graph:

======  =====================================================================
 name    definition
======  =====================================================================
 d       log1p of the degree
 T       log1p of the number of triangles containing the node
 af3m    mean of AF3 over the incident edges
 af3s    ``q90 - q10`` of AF3 over the incident edges
 tri     share of incident edges that lie in at least one triangle
 open    ``1 - tri``, the share of incident edges that lie in none
 hent    normalised Shannon entropy of the neighbour-degree distribution
 xcom    share of neighbours in a different community
======  =====================================================================

Every dimension is standardised with the TRAIN mean and standard deviation of
the *observed* entries. A dimension that cannot be measured is recorded as
``NaN`` and flagged in the mask -- never filled with a measured-looking zero.
The gate consumes ``0.0`` for a flagged entry, which after standardisation is
exactly the population mean; that routing rule is declared in the math spec, and
the counts travel with the table.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

#: The dimension names, in the order the gate consumes them.
DESCRIPTOR_NAMES = (
    "degree_log",
    "triangles_log",
    "af3_mean",
    "af3_spread",
    "triangle_edge_share",
    "open_edge_share",
    "neighbour_degree_entropy",
    "cross_community_share",
)
DESCRIPTOR_DIMS = len(DESCRIPTOR_NAMES)

#: Where the frozen table and its manifest live, relative to the repository root.
TABLE_PATH_RELATIVE = ("results", "research", "issue149", "structure_table.npz")
#: The frozen AF3 cache of issue #136, which the rebuild has to reproduce exactly.
AF3_CACHE_RELATIVE = ("results", "research", "issue121", "community_curvature",
                      "train_edge_af3.npz")
TRAIN_RELATIVE = ("results", "stage0_build_parquet", "train.parquet")
EMBEDDING_RELATIVE = ("stage1_GeneEmbedding", "output", "sentence_t5.npy")

#: The weight a consecutive pair accumulates in the weighted graph, exactly as
#: the frozen construction counts it; the binary graph only asks whether it is
#: positive, so the value never enters a descriptor.
TRAIN_WEIGHT_STEP = 0.5
#: Louvain modularity resolution. One is the definition's default, not a tuned
#: value: the community labels only decide which neighbours count as crossing.
LOUVAIN_RESOLUTION = 1.0
#: One seed for the community labelling and for every frozen sample.
SEED = 42
#: Budget caps of the frozen sets the auxiliary terms consume. They are a cost
#: bound of one epoch, sampled once, and never selected on a result.
TRIANGLE_CAP = 200_000
PAIR_CAP = 200_000


# --------------------------------------------------------------------- graph
def train_graph(root: Path | str) -> dict:
    """The TRAIN behaviour graph, rebuilt and verified against the frozen cache.

    A row's history plus its target is one training sequence and consecutive
    items in it are adjacent -- the construction the hierarchy audit and the AF3
    measurement both use. The rebuild is checked, edge for edge, against the
    frozen ``train_edge_af3.npz`` of issue #136: if the two disagree the table is
    not built, because a descriptor of a different graph is not this mechanism's
    input.
    """
    import pandas as pd
    import scipy.sparse as sp

    root = Path(root)
    frame = pd.read_parquet(root.joinpath(*TRAIN_RELATIVE), columns=["history", "target"])
    weights: dict[tuple[int, int], float] = {}
    for history, target in zip(frame["history"].tolist(), frame["target"].tolist()):
        values = [int(item) for item in history]
        if target is not None:
            values.append(int(target))
        for first, second in zip(values[:-1], values[1:]):
            if first == second:
                continue
            key = (first, second) if first < second else (second, first)
            weights[key] = weights.get(key, 0.0) + TRAIN_WEIGHT_STEP
    nodes = np.asarray(sorted({item for edge in weights for item in edge}), dtype=np.int64)
    index = {int(item): position for position, item in enumerate(nodes)}
    rows = np.fromiter((index[first] for first, _ in weights), dtype=np.int64, count=len(weights))
    cols = np.fromiter((index[second] for _, second in weights), dtype=np.int64, count=len(weights))
    data = np.fromiter(weights.values(), dtype=np.float64, count=len(weights))
    weighted = sp.coo_matrix((data, (rows, cols)), shape=(len(nodes), len(nodes))).tocsr()
    binary = (weighted != 0).astype(np.int8).tocsr()
    binary = binary.maximum(binary.T)
    binary.setdiag(0)
    binary.eliminate_zeros()
    full_rows, full_cols = binary.nonzero()
    keep = full_rows < full_cols
    edge_rows, edge_cols = full_rows[keep], full_cols[keep]
    order = np.lexsort((edge_cols, edge_rows))
    edge_rows, edge_cols = edge_rows[order], edge_cols[order]
    degree = np.asarray(binary.sum(axis=1)).ravel().astype(np.int64)
    triangles_per_edge = _triangles_per_edge(binary, edge_rows, edge_cols)
    af3 = 4.0 - degree[edge_rows] - degree[edge_cols] + 3.0 * triangles_per_edge
    cache = root.joinpath(*AF3_CACHE_RELATIVE)
    cache_status = "NOT_CHECKED"
    if cache.is_file():
        frozen = np.load(cache)
        if (int(frozen["nodes"]) != nodes.size or int(frozen["edges"]) != edge_rows.size
                or not np.array_equal(frozen["rows"].astype(np.int64), edge_rows)
                or not np.array_equal(frozen["cols"].astype(np.int64), edge_cols)):
            raise ValueError("the TRAIN graph rebuild does not reproduce the frozen AF3 cache")
        if not np.array_equal(frozen["af3"].astype(np.float64), af3):
            raise ValueError("the AF3 values of the rebuild disagree with the frozen cache")
        cache_status = "MATCHED"
    return {"nodes": nodes, "index": index, "binary": binary, "rows": edge_rows,
            "cols": edge_cols, "degree": degree, "triangles_per_edge": triangles_per_edge,
            "af3": af3, "af3_cache": str(cache), "af3_cache_status": cache_status}


def _triangles_per_edge(binary, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """``|N(u) & N(v)|`` for every undirected edge, in the edge order given."""
    size = int(binary.shape[0])
    integer = binary.astype(np.int32)
    product = (integer @ integer).multiply(integer).tocoo()
    keep = product.row < product.col
    key = product.row[keep].astype(np.int64) * size + product.col[keep].astype(np.int64)
    counts = np.rint(product.data[keep]).astype(np.int64)
    target = rows.astype(np.int64) * size + cols.astype(np.int64)
    positions = np.searchsorted(target, key)
    if positions.size and (positions.max() >= target.size
                           or not np.array_equal(target[positions], key)):
        raise RuntimeError("the triangle pass returned an edge outside the graph")
    per_edge = np.zeros(rows.size, dtype=np.int64)
    per_edge[positions] = counts
    return per_edge


def communities(binary, seed: int = SEED, resolution: float = LOUVAIN_RESOLUTION) -> np.ndarray:
    """Louvain labels on the TRAIN graph, seeded so the labels are reproducible."""
    import networkx as nx
    graph = nx.from_scipy_sparse_array(binary.tocsr())
    groups = nx.community.louvain_communities(graph, weight=None, resolution=resolution,
                                              seed=seed)
    labels = np.full(int(binary.shape[0]), -1, dtype=np.int64)
    for label, members in enumerate(groups):
        labels[np.fromiter(sorted(members), dtype=np.int64, count=len(members))] = label
    if (labels < 0).any():
        raise RuntimeError("the community pass left a node unlabelled")
    return labels


def _neighbours(adjacency, node: int) -> np.ndarray:
    if hasattr(adjacency, "indptr"):
        start, stop = int(adjacency.indptr[node]), int(adjacency.indptr[node + 1])
        return np.asarray(adjacency.indices[start:stop], dtype=np.int64)
    return np.nonzero(np.asarray(adjacency)[node])[0].astype(np.int64)


def _edges(adjacency) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(adjacency, "indptr"):
        rows, cols = adjacency.nonzero()
    else:
        rows, cols = np.nonzero(np.asarray(adjacency))
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    keep = rows < cols
    rows, cols = rows[keep], cols[keep]
    order = np.lexsort((cols, rows))
    return rows[order], cols[order]


def _node_descriptor(node: int, adjacency, triangles_per_edge: dict, af3: dict,
                     community: np.ndarray, n_nodes: int | None = None) -> np.ndarray:
    """The eight raw dimensions of one node; unmeasured entries are ``NaN``."""
    values = np.full(DESCRIPTOR_DIMS, np.nan, dtype=np.float64)
    neighbours = _neighbours(adjacency, node)
    degree = int(neighbours.size)
    if degree == 0:
        return values
    size = int(n_nodes if n_nodes is not None else np.asarray(community).size)
    incident_triangles = np.empty(degree, dtype=np.float64)
    incident_af3 = np.empty(degree, dtype=np.float64)
    for position, other in enumerate(neighbours.tolist()):
        key = (node, other) if node < other else (other, node)
        incident_triangles[position] = float(triangles_per_edge[key])
        incident_af3[position] = float(af3[key])
    values[0] = np.log1p(degree)
    values[1] = np.log1p(incident_triangles.sum() / 2.0)
    values[2] = incident_af3.mean()
    values[3] = np.quantile(incident_af3, 0.9) - np.quantile(incident_af3, 0.1)
    values[4] = float((incident_triangles > 0).mean())
    values[5] = 1.0 - values[4]
    degrees = np.asarray([int(_neighbours(adjacency, other).size) for other in neighbours.tolist()])
    if degree >= 2:
        _, counts = np.unique(degrees, return_counts=True)
        probability = counts.astype(np.float64) / degree
        values[6] = float(-(probability * np.log(probability)).sum() / np.log(degree))
    values[7] = float((community[neighbours] != community[node]).mean())
    return values


def descriptor_matrix(adjacency, degree: np.ndarray, triangles_per_edge: dict, af3: dict,
                      community: np.ndarray, nodes_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Every node's raw descriptor, plus the mask of the dimensions measured."""
    raw = np.full((int(nodes_len), DESCRIPTOR_DIMS), np.nan, dtype=np.float64)
    for node in range(int(nodes_len)):
        if int(degree[node]) == 0:
            continue
        raw[node] = _node_descriptor(node, adjacency, triangles_per_edge, af3, community,
                                     n_nodes=int(nodes_len))
    observed = np.isfinite(raw)
    return raw, observed


def standardise(raw: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score every dimension with the observed TRAIN entries of that dimension.

    An unobserved entry becomes ``0.0`` -- the population mean in standardised
    units -- and stays flagged in ``observed``. A dimension with no spread maps
    every observed entry to ``0.0`` instead of dividing by zero.
    """
    values = np.zeros_like(raw, dtype=np.float64)
    mean = np.zeros(DESCRIPTOR_DIMS, dtype=np.float64)
    std = np.zeros(DESCRIPTOR_DIMS, dtype=np.float64)
    for dim in range(raw.shape[1]):
        column = raw[observed[:, dim], dim]
        if column.size == 0:
            raise ValueError(f"descriptor dimension {DESCRIPTOR_NAMES[dim]} has no observed entry")
        mean[dim] = float(column.mean())
        std[dim] = float(column.std())
        scale = std[dim] if std[dim] > 0 else 1.0
        values[observed[:, dim], dim] = (column - mean[dim]) / scale
    return values, mean, std


# -------------------------------------------------------------- frozen sets
def frozen_sets(adjacency, community: np.ndarray, seed: int = SEED,
                triangle_cap: int = TRIANGLE_CAP, pair_cap: int = PAIR_CAP) -> dict:
    """The triangles, the cross-community edges and the same-community pairs.

    One deterministic sample of the TRAIN graph, frozen with the table: the
    auxiliary terms consume these every epoch, so they must not move. The two
    pair sets are the real edges of the graph, split into crossing and
    within-community edges, so the bridge hinge compares like with like.
    """
    rows, cols = _edges(adjacency)
    rng = np.random.default_rng(seed)
    order = rng.permutation(rows.size)
    triangles: dict[tuple[int, int, int], None] = {}
    for position in order.tolist():
        first, second = int(rows[position]), int(cols[position])
        common = np.intersect1d(_neighbours(adjacency, first), _neighbours(adjacency, second),
                                assume_unique=True)
        for third in common.tolist():
            triangles[tuple(sorted((first, second, int(third))))] = None
            if len(triangles) >= int(triangle_cap):
                break
        if len(triangles) >= int(triangle_cap):
            break
    triangle_array = np.asarray(sorted(triangles), dtype=np.int64).reshape(-1, 3)

    labels = np.asarray(community, dtype=np.int64)
    # The two sides of the bridge hinge are the same kind of object: real TRAIN
    # edges, split by whether they cross a community. A cross edge is compared
    # with an in-community edge, so the term asks that the coarse prefixes keep
    # community structure without forcing a real bridge into one group.
    crossing = labels[rows] != labels[cols]
    cross = np.stack([rows[crossing], cols[crossing]], axis=1)
    within = np.stack([rows[~crossing], cols[~crossing]], axis=1)
    if cross.shape[0] > int(pair_cap):
        chosen = np.sort(rng.permutation(cross.shape[0])[:int(pair_cap)])
        cross = cross[chosen]
    if within.shape[0] > int(pair_cap):
        chosen = np.sort(rng.permutation(within.shape[0])[:int(pair_cap)])
        within = within[chosen]
    return {"triangles": triangle_array, "cross_pairs": cross, "same_pairs": within}


# -------------------------------------------------------------------- build
def table_path(root: Path | str) -> Path:
    return Path(root).joinpath(*TABLE_PATH_RELATIVE)


def manifest_path(path: Path | str) -> Path:
    path = Path(path)
    return path.with_suffix(".json")


def sha256_of(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(root: Path | str) -> dict:
    """Every array of the frozen table, plus the counts a reader has to see."""
    root = Path(root)
    graph = train_graph(root)
    labels = communities(graph["binary"])
    edges = {(int(row), int(col)): (float(tri), float(value))
             for row, col, tri, value in zip(graph["rows"], graph["cols"],
                                             graph["triangles_per_edge"], graph["af3"])}
    triangles_per_edge = {key: pair[0] for key, pair in edges.items()}
    af3 = {key: pair[1] for key, pair in edges.items()}
    raw, observed = descriptor_matrix(graph["binary"], graph["degree"], triangles_per_edge,
                                      af3, labels, nodes_len=graph["nodes"].size)
    values, mean, std = standardise(raw, observed)
    sets = frozen_sets(graph["binary"], labels)

    item_count = int(np.load(root.joinpath(*EMBEDDING_RELATIVE), mmap_mode="r").shape[0])
    node_ids = graph["nodes"]
    if node_ids.size and int(node_ids.max()) >= item_count:
        raise ValueError("the TRAIN graph names an item outside the embedding rows")
    full_values = np.zeros((item_count, DESCRIPTOR_DIMS), dtype=np.float32)
    full_raw = np.full((item_count, DESCRIPTOR_DIMS), np.nan, dtype=np.float32)
    full_observed = np.zeros((item_count, DESCRIPTOR_DIMS), dtype=bool)
    full_community = np.full(item_count, -1, dtype=np.int64)
    full_values[node_ids] = values.astype(np.float32)
    full_raw[node_ids] = raw.astype(np.float32)
    full_observed[node_ids] = observed
    full_community[node_ids] = labels

    missing = ~full_observed.any(axis=1)
    metadata = {
        "issue": 149,
        "mechanism": "AF3-guided structure-preserving mixed-curvature quantization",
        "descriptor_names": list(DESCRIPTOR_NAMES),
        "n_items": item_count,
        "n_nodes": int(node_ids.size),
        "n_edges": int(graph["rows"].size),
        "n_communities": int(np.unique(labels).size),
        "louvain_resolution": LOUVAIN_RESOLUTION,
        "seed": SEED,
        "triangle_cap": TRIANGLE_CAP,
        "pair_cap": PAIR_CAP,
        "triangles": int(sets["triangles"].shape[0]),
        "cross_pairs": int(sets["cross_pairs"].shape[0]),
        "same_pairs": int(sets["same_pairs"].shape[0]),
        "items_without_structure": int(missing.sum()),
        "undefined_dimensions": {name: int((~full_observed[:, dim]).sum())
                                 for dim, name in enumerate(DESCRIPTOR_NAMES)},
        "standardisation": {"mean": [float(value) for value in mean],
                            "std": [float(value) for value in std]},
        "degree": {"min": int(graph["degree"].min()), "max": int(graph["degree"].max()),
                   "mean": float(graph["degree"].mean())},
        "af3_cache": graph["af3_cache"],
        "af3_cache_status": graph["af3_cache_status"],
        "train_parquet": str(root.joinpath(*TRAIN_RELATIVE)),
        "train_parquet_sha256": sha256_of(root.joinpath(*TRAIN_RELATIVE)),
        "embedding_sha256": sha256_of(root.joinpath(*EMBEDDING_RELATIVE)),
        "input_rule": ("TRAIN split only; the validation and test splits and any future "
                       "behaviour are never read"),
    }
    return {"values": full_values, "raw": full_raw, "observed": full_observed,
            "mean": mean, "std": std, "item_ids": node_ids, "community": full_community,
            "triangles": sets["triangles"].astype(np.int32),
            "cross_pairs": sets["cross_pairs"].astype(np.int32),
            "same_pairs": sets["same_pairs"].astype(np.int32),
            "metadata": metadata}


def write_table(root: Path | str) -> dict:
    """Write the table and its manifest; the manifest pins the table's SHA-256."""
    root = Path(root)
    payload = build(root)
    path = table_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        values=payload["values"], raw=payload["raw"], observed=payload["observed"],
        mean=payload["mean"], std=payload["std"], item_ids=payload["item_ids"],
        community=payload["community"], triangles=payload["triangles"],
        cross_pairs=payload["cross_pairs"], same_pairs=payload["same_pairs"],
        metadata=np.asarray(json.dumps(payload["metadata"], sort_keys=True)),
    )
    manifest = {"table": str(path), "table_sha256": sha256_of(path),
                "metadata": payload["metadata"]}
    manifest_path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True),
                                   encoding="utf-8")
    return manifest


def load(path: Path | str) -> dict:
    """Read the table (no SHA check; ``verify`` is the pinned read)."""
    with np.load(Path(path), allow_pickle=False) as payload:
        loaded = {key: payload[key] for key in payload.files}
    loaded["metadata"] = json.loads(str(loaded["metadata"]))
    loaded["observed"] = loaded["observed"].astype(bool)
    return loaded


def verify(path: Path | str) -> dict:
    """Read the manifest and refuse a table whose bytes moved."""
    path = Path(path)
    manifest = json.loads(manifest_path(path).read_text(encoding="utf-8"))
    digest = sha256_of(path)
    if manifest["table_sha256"] != digest:
        raise ValueError("the frozen structure table does not match its manifest: "
                         + digest + " != " + manifest["table_sha256"])
    return manifest
