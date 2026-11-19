"""Graphlet structural supervisor: a triplet predictor that tells triangles from wedges.

    L_graphlet = BCE( f_theta(z_a, z_b, z_c), y )      y = 1 triangle, 0 open wedge

``f_theta`` is a small permutation-invariant head over the three **encoder** outputs.
It is not additive over edges: alongside a per-node term it carries the pairwise
products ``z_a*z_b + z_a*z_c + z_b*z_c``, so a purely edge-additive model cannot
reproduce it -- the ablation that removes the interaction term is what tells us the
triad information matters.

The gradients of the classification error reach the encoder, so the item vectors
must carry what distinguishes the two local structures.  The head is trained jointly
and discarded at export: Stage 3 still consumes the three-code SID.

Wedges are degree-matched to the triangles on all three nodes' degree buckets, and
the graph comes from the project's single undirected graph after its sha256 is
checked against ``metadata.json`` (CLAUDE.md 89b).
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCHEMA_PREFIX = "genrec.behavior_graph"
#: Share of the total loss the classification term takes at the first step (0 = off).
GRAPHLET_SHARE = 0.1
#: Triangles and wedges drawn per step; both classes are always present.
PAIRS_PER_STEP = 256
#: Matched (triangle, wedge) pairs frozen once at startup.
PAIR_POOL = 8192
#: Steps whose reconstruction/loss medians calibrate the weight.
CALIBRATION_STEPS = 20
EPS = 1e-12


def repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "stage2_RQ-VAE").is_dir() and (parent / "results").is_dir():
            return parent
    raise RuntimeError("repository root not found above " + __file__)


GRAPH_DIR = repo_root() / "results" / "research" / "behavior_graph"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_graph(folder: Path | str | None = None) -> dict:
    folder = Path(folder) if folder is not None else GRAPH_DIR
    metadata_path = folder / "metadata.json"
    graph_path = folder / "graph.npz"
    if not metadata_path.is_file() or not graph_path.is_file():
        raise RuntimeError("behaviour graph missing under %s; run "
                           "analysis_sid_recall/build_behavior_graph.py" % folder)
    metadata = json.loads(metadata_path.read_text())
    if not str(metadata.get("schema", "")).startswith(SCHEMA_PREFIX):
        raise RuntimeError("graph metadata schema is not %s.*: %r"
                           % (SCHEMA_PREFIX, metadata.get("schema")))
    policy = metadata.get("policy") or {}
    if policy.get("single_undirected_graph") is not True or policy.get("directed_graph_stored") is not False:
        raise RuntimeError("graph metadata policy must declare one undirected graph and no "
                           "stored directed copy; got %r" % (policy,))
    record = (metadata.get("outputs") or {}).get("graph.npz")
    if not record:
        raise RuntimeError("graph metadata lists no sha256 for graph.npz")
    digest = _sha256(graph_path)
    if digest != record.get("sha256"):
        raise RuntimeError("graph.npz sha256 %s does not match metadata %s; rebuild the graph"
                           % (digest[:16], str(record.get("sha256"))[:16]))
    with np.load(graph_path, allow_pickle=False) as payload:
        rows = np.asarray(payload["rows"], dtype=np.int64)
        cols = np.asarray(payload["cols"], dtype=np.int64)
        triangles = np.asarray(payload["triangles"], dtype=np.int64)
        af3 = np.asarray(payload["af3"], dtype=np.float64)
        af3_median = float(payload["median"])
        af3_p90 = float(payload["p90"])
        n_nodes = int(payload["nodes"])
    degree = np.bincount(np.concatenate([rows, cols]), minlength=n_nodes)
    return {"rows": rows, "cols": cols, "triangles": triangles, "af3": af3,
            "af3_median": af3_median, "af3_p90": af3_p90,
            "degree": degree, "n_nodes": n_nodes}


def _degree_bucket(value: int) -> int:
    for limit in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000):
        if value <= limit:
            return limit
    return 10_000


class TripletSampler:
    """Matched (triangle, open wedge) pairs, frozen once."""

    def __init__(self, graph: dict):
        self.degree = graph["degree"]
        neighbours = defaultdict(set)
        for left, right in zip(graph["rows"].tolist(), graph["cols"].tolist()):
            neighbours[left].add(right)
            neighbours[right].add(left)
        self.adjacency = {node: np.asarray(sorted(values), dtype=np.int64)
                          for node, values in neighbours.items()}
        buckets = defaultdict(list)
        for node, values in neighbours.items():
            if values:
                buckets[_degree_bucket(len(values))].append(node)
        self.buckets = {key: np.asarray(values, dtype=np.int64) for key, values in buckets.items()}
        self.seeds = np.asarray([(int(a), int(b)) for a, b, count in
                                 zip(graph["rows"], graph["cols"], graph["triangles"]) if count > 0],
                                dtype=np.int64)
        self.pool = None

    def _is_edge(self, left: int, right: int) -> bool:
        values = self.adjacency.get(left)
        return values is not None and bool(np.isin(right, values))

    def sample(self, generator: torch.Generator, count: int = PAIRS_PER_STEP):
        """Draw `count` triangles and `count` matched wedges."""
        if self.pool is None:
            self._build_pool()
        triangles, wedges = self.pool
        rng = np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,), generator=generator).item()))
        index = rng.integers(0, len(triangles), size=min(count, len(triangles)))
        return triangles[index], wedges[index]

    def _build_pool(self, target: int = PAIR_POOL):
        rng = np.random.default_rng(0)
        triangles, wedges = [], []
        order = rng.permutation(len(self.seeds))
        for index in order:
            if len(triangles) >= target:
                break
            left, right = self.seeds[index]
            common = np.intersect1d(self.adjacency[left], self.adjacency[right])
            if not len(common):
                continue
            third = int(common[rng.integers(0, len(common))])
            members = [int(left), int(right), third]
            centre = max(members, key=lambda node: self.degree[node])
            leaves = [node for node in members if node != centre]
            wedge = self._matched_wedge(centre, leaves, rng)
            if wedge is None:
                continue
            triangles.append((centre, leaves[0], leaves[1]))
            wedges.append(wedge)
        if not triangles:
            raise RuntimeError("no matched triangle/wedge pair could be built from the graph")
        self.pool = (np.asarray(triangles, dtype=np.int64).reshape(-1, 3),
                     np.asarray(wedges, dtype=np.int64).reshape(-1, 3))

    def _matched_wedge(self, centre: int, leaves, rng):
        for _ in range(60):
            pool = self.buckets.get(_degree_bucket(int(self.degree[centre])))
            if pool is None:
                return None
            candidate = int(pool[rng.integers(0, len(pool))])
            values = self.adjacency.get(candidate)
            if values is None or len(values) < 2:
                continue
            picks = values[rng.integers(0, len(values), size=8)]
            chosen = [int(node) for node in picks if int(node) != candidate]
            for first in chosen:
                for second in chosen:
                    if first >= second:
                        continue
                    if _degree_bucket(int(self.degree[first])) != _degree_bucket(int(self.degree[leaves[0]])):
                        continue
                    if _degree_bucket(int(self.degree[second])) != _degree_bucket(int(self.degree[leaves[1]])):
                        continue
                    if self._is_edge(first, second):
                        continue
                    return (candidate, first, second)
        return None


def _af3_matrices(graph: dict):
    """Sparse (kappa, support) matrices, built once and cached on the graph dict."""
    cache = graph.get("_af3_cache")
    if cache is None:
        import scipy.sparse as sparse
        rows, cols = graph["rows"], graph["cols"]
        af3 = np.asarray(graph["af3"], dtype=np.float64)
        n = graph["n_nodes"]
        # The edge list stores each undirected edge once, so both directions go in.
        both_rows = np.concatenate([rows, cols])
        both_cols = np.concatenate([cols, rows])
        kappa = sparse.csr_matrix((np.concatenate([af3, af3]), (both_rows, both_cols)), shape=(n, n))
        support = sparse.csr_matrix((np.ones(len(both_rows), dtype=bool), (both_rows, both_cols)),
                                    shape=(n, n))
        cache = (kappa, support, float(graph["af3_median"]),
                 max(float(graph["af3_p90"]) - float(graph["af3_median"]), 1e-6))
        graph["_af3_cache"] = cache
    return cache


def _mean_kappa(kappa, support, triples: np.ndarray) -> np.ndarray:
    first, second, third = triples[:, 0], triples[:, 1], triples[:, 2]
    values = np.stack([np.asarray(kappa[first, second]).ravel(),
                       np.asarray(kappa[first, third]).ravel(),
                       np.asarray(kappa[second, third]).ravel()])
    present = np.stack([np.asarray(support[first, second]).ravel(),
                        np.asarray(support[first, third]).ravel(),
                        np.asarray(support[second, third]).ravel()])
    total = (values * present).sum(axis=0)
    count = present.sum(axis=0)
    return np.where(count > 0, total / np.maximum(count, 1), 0.0)


def af3_target(graph: dict, triangles: np.ndarray, wedges: np.ndarray) -> np.ndarray:
    """Mean AF3 curvature over the edges each triple has, standardised by the graph.

    A triangle contributes three edges, an open wedge two.  The two sparse lookups
    are built once: a per-triple Python loop over 315k edges made an epoch five times
    slower.
    """
    kappa, support, median, scale = _af3_matrices(graph)
    tri = (_mean_kappa(kappa, support, triangles) - median) / scale
    wed = (_mean_kappa(kappa, support, wedges) - median) / scale
    return np.concatenate([tri, wed]).astype(np.float32)


class TripletPredictor(nn.Module):
    """Permutation-invariant triangle/wedge classifier over three item vectors.

    ``interaction=False`` removes the pairwise-product branch, which is the ablation
    that shows whether the triad information (and not an additive edge proxy) is what
    the head uses.
    """

    def __init__(self, dim: int = 32, hidden: int = 64, interaction: bool = True,
                 af3: bool = False):
        super().__init__()
        self.interaction = interaction
        self.af3 = af3
        self.node = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU())
        self.pair = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU()) if interaction else None
        width = hidden * 2 if interaction else hidden
        self.head = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 2 if af3 else 1))

    def forward(self, first: torch.Tensor, second: torch.Tensor, third: torch.Tensor):
        nodes = self.node(torch.stack([first, second, third], dim=1)).mean(dim=1)
        if self.pair is not None:
            product = (first * second + first * third + second * third) / 3.0
            nodes = torch.cat([nodes, self.pair(product)], dim=1)
        out = self.head(nodes)
        if self.af3:
            return out[:, 0], out[:, 1]
        return out.squeeze(1)


def graphlet_loss(logits_triangle: torch.Tensor, logits_wedge: torch.Tensor) -> torch.Tensor:
    """Balanced binary cross-entropy over the two classes."""
    targets_triangle = torch.ones_like(logits_triangle)
    targets_wedge = torch.zeros_like(logits_wedge)
    return (F.binary_cross_entropy_with_logits(logits_triangle, targets_triangle)
            + F.binary_cross_entropy_with_logits(logits_wedge, targets_wedge)) / 2.0


def af3_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Huber on the standardised curvature, delta = one inter-quantile step."""
    return F.huber_loss(prediction, target, delta=1.0)


def calibrate_share(recon: float, term: float, share: float = GRAPHLET_SHARE) -> float:
    share = float(share)
    return share * float(recon) / max((1.0 - share) * float(term), EPS)


def accuracy(logits_triangle: torch.Tensor, logits_wedge: torch.Tensor) -> float:
    """(logits > 0) is the triangle call; a balanced accuracy over both classes."""
    correct = (logits_triangle > 0).float().mean() + (logits_wedge <= 0).float().mean()
    return float(correct / 2.0)
