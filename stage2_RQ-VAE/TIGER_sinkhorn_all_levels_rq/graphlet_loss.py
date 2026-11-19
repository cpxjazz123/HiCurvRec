"""Graphlet supervision on the L1 soft assignment.

    L_graphlet = -log softmax( S1(triangle) / tau , S1(wedge_k) / tau )
    S1(a,b,c)  = sum_k p_a(k) p_b(k) p_c(k)        (k over the 256 level-1 codewords)

``p_i`` is the level-1 soft assignment of item ``i``: the quantiser's own distance
matrix, centred the way ``VQLayer.center_distance`` centres it, passed through
``softmax(-d / SK_EPSILON)``.  A triangle's three items sharing one codeword gives
``S1 = 1``; three disjoint assignments give ``S1 = 0``.

The negative of a triangle is a **degree-matched open wedge** (a centre with two
adjacent leaves that are not adjacent to each other), matched on all three nodes'
degree buckets, so the pair differs only in closure.

Nothing is hand-tuned:

* ``tau`` is the measured initial gap between the mean ``S1`` of triangles and of
  wedges, so the logit gap starts at one nat;
* ``lambda`` is calibrated from the first step's reconstruction loss so the term is
  ``GRAPHLET_SHARE`` of the total there, and the term and weight are written to the
  metrics stream every epoch;
* the graph is the project's single undirected graph and its sha256 is checked
  against ``metadata.json`` before a triplet is sampled (CLAUDE.md 89b).

The exported identifier is still the quantiser's; ``l1_agreement`` reports how
often ``argmax`` of the soft assignment equals the exported level-1 code, so the
proxy can be audited rather than assumed.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

SCHEMA_PREFIX = "genrec.behavior_graph"
#: Share of the total loss the graphlet term takes at the first step (0.0 = off).
GRAPHLET_SHARE = 0.1
#: Triplets per step; each triangle is paired with one matched open wedge.
TRIPLETS_PER_STEP = 256
#: Matched (triangle, wedge) pairs drawn once at startup; steps sample from the pool.
PAIR_POOL = 4096
#: Negatives per anchor: the matched wedge plus the rest of the batch's wedges.
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
    """Edges and degrees, after the metadata contract verifies."""
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
        n_nodes = int(payload["nodes"])
    degree = np.bincount(np.concatenate([rows, cols]), minlength=n_nodes)
    return {"rows": rows, "cols": cols, "triangles": triangles,
            "degree": degree, "n_nodes": n_nodes}


def _degree_bucket(value: int) -> int:
    for limit in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000):
        if value <= limit:
            return limit
    return 10_000


class TripletSampler:
    """Closed triangles and degree-matched open wedges."""

    def __init__(self, graph: dict):
        self.degree = graph["degree"]
        self.n_nodes = graph["n_nodes"]
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
        closed = [(int(a), int(b)) for a, b, count in
                  zip(graph["rows"], graph["cols"], graph["triangles"]) if count > 0]
        self.seeds = np.asarray(closed, dtype=np.int64) if closed else np.zeros((0, 2), np.int64)
        self.pool = None

    def _is_edge(self, left: int, right: int) -> bool:
        values = self.adjacency.get(left)
        return values is not None and bool(np.isin(right, values))

    def sample(self, generator: torch.Generator, count: int = TRIPLETS_PER_STEP):
        """Draw from the frozen pool of matched pairs (built once)."""
        if self.pool is None:
            self._build_pool()
        rng = np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,), generator=generator).item()))
        index = rng.integers(0, len(self.pool[0]), size=min(count, len(self.pool[0])))
        return self.pool[0][index], self.pool[1][index]

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
        """A centre from the same degree bucket whose two adjacent leaves are not adjacent."""
        for _ in range(60):
            pool = self.buckets.get(_degree_bucket(int(self.degree[centre])))
            if pool is None:
                return None
            candidate = int(pool[rng.integers(0, len(pool))])
            values = self.adjacency.get(candidate)
            if values is None or len(values) < 2:
                continue
            picks = values[rng.integers(0, len(values), size=12)]
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


def soft_assignment(distances: torch.Tensor, epsilon: float) -> torch.Tensor:
    """The quantiser's distances, centred the way ``VQLayer`` centres them, as a softmax."""
    maximum = distances.max()
    minimum = distances.min()
    middle = (maximum + minimum) / 2
    amplitude = (maximum - middle) + 1e-5
    centred = (distances - middle) / amplitude
    return torch.softmax(-centred / epsilon, dim=1)


def shared_probability(first: torch.Tensor, second: torch.Tensor, third: torch.Tensor) -> torch.Tensor:
    """S1: the chance that all three items take the same level-1 codeword."""
    return (first * second * third).sum(dim=1)


def graphlet_infonce(triangle_s1: torch.Tensor, wedge_s1: torch.Tensor,
                     temperature: float) -> torch.Tensor:
    """Triangles must score higher than the matched wedges and the batch's other wedges."""
    logits = torch.cat([(triangle_s1 / temperature).unsqueeze(1),
                        (wedge_s1 / temperature).unsqueeze(1)], dim=1)
    return torch.nn.functional.cross_entropy(
        logits, torch.zeros(len(logits), dtype=torch.long, device=logits.device))


def calibrate_temperature(triangle_mean: float, wedge_mean: float) -> float:
    """The temperature that makes the initial logit gap one nat."""
    return max(float(triangle_mean) - float(wedge_mean), 1e-3)


def calibrate_share(recon: float, term: float, share: float = GRAPHLET_SHARE) -> float:
    """The weight that makes the graphlet term ``share`` of the total at this step."""
    share = float(share)
    return share * float(recon) / max((1.0 - share) * float(term), EPS)


def l1_agreement(codebook: torch.Tensor, latent: torch.Tensor, hard_codes: np.ndarray,
                 epsilon: float) -> float:
    """How often argmax of the soft assignment equals the exported level-1 code."""
    distances = ((latent[:, None, :] - codebook[None, :, :]) ** 2).sum(dim=2)
    soft = soft_assignment(distances, epsilon)
    predicted = soft.argmax(dim=1).detach().cpu().numpy()
    return float((predicted == hard_codes).mean())
