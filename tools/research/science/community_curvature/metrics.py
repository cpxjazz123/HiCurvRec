"""The measurements that say whether the learned communities are real.

Four measurements are reported, and they are the ones the experiment's question
needs:

* **curvature separation** -- how much more community mass a high-AF3 pair
  shares than a low-AF3 pair of the same anchor;
* **community occupancy** -- how many of the 64 communities are actually used,
  and how concentrated the usage is;
* **community transitivity** -- whether the items of a community form a tight
  subgraph (the induced subgraph's transitivity);
* **curvature gap** -- the AF3 curvature of the within-community edges minus
  that of the cross-community edges, computed with the frozen
  ``af3.gap_metrics`` on the original graph.

Two further readings are reported because they are the ones this repository
already tied to the Stage-3 recall: the group (quotient) transitivity, and the
same AF3 gap measured on the independent validation graph.

Every number comes with the random-label null it has to beat: a random
assignment of the same size is measured the same way.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .data import Graph, af3_module, quotient_module


def _hard_labels(pi: np.ndarray) -> np.ndarray:
    """The community of every item: the argmax of its soft distribution."""
    return pi.argmax(axis=1).astype(np.int64)


def _entropy(usage: np.ndarray) -> float:
    usage = usage[usage > 0]
    if usage.size == 0:
        return 0.0
    return float(-(usage * np.log(usage)).sum())


def occupancy(pi: np.ndarray) -> dict:
    """How many communities are used and how concentrated the usage is.

    The share of a community is the fraction of items whose hard assignment
    (the argmax of the soft distribution) falls on it, so the shares sum to one
    and a collapsed model puts every item on one community.
    """
    labels = _hard_labels(pi)
    counts = np.bincount(labels, minlength=pi.shape[1]).astype(float)
    share = counts / max(labels.size, 1)
    usage = pi.mean(axis=0)
    order = np.sort(usage)[::-1]
    return {
        "n_communities": int(pi.shape[1]),
        "used_communities": int((usage > 0).sum()),
        "effective_communities": float(np.exp(_entropy(usage))),
        "communities_above_1pct": int((share >= 0.01).sum()),
        "communities_above_5pct": int((share >= 0.05).sum()),
        "largest_usage_share": float(share.max()),
        "top5_usage_share": float(order[:5].sum()),
        "usage_entropy": _entropy(usage),
    }


def separation(pi: np.ndarray, pairs, seed: int = 42, replicates: int = 2000) -> dict:
    """S_high - S_low over the anchors, with a bootstrap interval over anchors."""
    anchor = pi[pairs.anchor]
    s_high = (anchor * pi[pairs.high]).sum(axis=1)
    s_low = (anchor * pi[pairs.low]).sum(axis=1)
    difference = s_high - s_low
    rng = np.random.default_rng(seed)
    n = difference.size
    if n == 0:
        return {"pairs": 0, "s_high": None, "s_low": None, "separation": None,
                "ci_low": None, "ci_high": None}
    draws = rng.integers(0, n, size=(replicates, n))
    means = difference[draws].mean(axis=1)
    return {
        "pairs": int(n),
        "s_high": float(s_high.mean()),
        "s_low": float(s_low.mean()),
        "separation": float(difference.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "bootstrap_replicates": int(replicates),
    }


def _transitivity(adjacency: sp.csr_matrix) -> tuple[float | None, int, int]:
    """3 * triangles / wedges of a graph, the convention of the AF3 pipeline."""
    if adjacency.shape[0] < 2:
        return None, 0, 0
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    wedges = float((degree * (degree - 1.0)).sum() / 2.0)
    if wedges <= 0:
        return None, 0, 0
    integer = adjacency.astype(np.int32)
    # A o A^2 counts every triangle six times, once per ordered pair of its edges
    triangles = float(integer.multiply(integer @ integer).sum() / 6.0)
    return (3.0 * triangles / wedges if wedges else None), int(triangles), int(wedges)


def community_transitivity(graph: Graph, labels: np.ndarray) -> dict:
    """The induced subgraph's transitivity of every community that has one."""
    values, weights = [], []
    for community in np.unique(labels):
        members = np.nonzero(labels == community)[0]
        if members.size < 2:
            continue
        inner = graph.adjacency[members][:, members]
        if inner.nnz == 0:
            continue
        value, triangles, wedges = _transitivity(inner)
        if value is None:
            continue
        values.append(value)
        weights.append(wedges)
    if not values:
        return {"eligible_communities": 0, "mean": None, "edge_weighted_mean": None}
    values_array = np.asarray(values, dtype=float)
    weights_array = np.asarray(weights, dtype=float)
    return {
        "eligible_communities": int(values_array.size),
        "mean": float(values_array.mean()),
        "edge_weighted_mean": float(np.average(values_array, weights=weights_array))
        if weights_array.sum() > 0 else None,
    }


def group_transitivity(graph: Graph, labels: np.ndarray, n_communities: int) -> dict:
    """The transitivity of the quotient graph: communities as its nodes."""
    quotient = quotient_module().quotient_graph(graph.adjacency, labels, n_communities)
    value, triangles, wedges = _transitivity(quotient)
    return {"quotient_edges": int(quotient.nnz // 2), "triangles": triangles,
            "wedges": wedges, "transitivity": value}


def curvature_gap(graph: Graph, curvature: dict, labels: np.ndarray) -> dict:
    """The AF3 curvature gap of the assignment on the graph's own edges."""
    af3 = af3_module()
    rows, cols = graph.edges
    return af3.gap_metrics(rows, cols, curvature["af3"], labels)


def null_measurements(graph: Graph, curvature: dict, n_communities: int,
                      replicates: int = 20, seed: int = 42) -> dict:
    """The same measurements on random assignments: the bar the model must clear."""
    rng = np.random.default_rng(seed)
    gaps, transitivity, groups = [], [], []
    for _ in range(replicates):
        labels = rng.integers(0, n_communities, size=graph.n_nodes).astype(np.int64)
        gap = curvature_gap(graph, curvature, labels)
        if gap.get("gap") is not None:
            gaps.append(gap["gap"])
        value = community_transitivity(graph, labels).get("mean")
        if value is not None:
            transitivity.append(value)
        group = group_transitivity(graph, labels, n_communities)["transitivity"]
        if group is not None:
            groups.append(group)
    def summary(values):
        if not values:
            return {"mean": None, "p95": None}
        array = np.asarray(values, dtype=float)
        return {"mean": float(array.mean()), "p95": float(np.quantile(array, 0.95)),
                "replicates": int(array.size)}
    return {"curvature_gap": summary(gaps), "community_transitivity": summary(transitivity),
            "group_transitivity": summary(groups)}
