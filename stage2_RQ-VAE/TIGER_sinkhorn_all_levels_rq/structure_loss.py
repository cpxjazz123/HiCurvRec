"""Graph contrastive structure: neighbours are not just near, they are nearer than controls.

    L_contrast = -log  exp(s(z_i, z_j)/tau)
                    / (exp(s(z_i, z_j)/tau) + sum_k exp(s(z_i, z_k)/tau))

``j`` is a behaviour-graph neighbour (positive), ``k`` runs over degree-matched
non-neighbours (negatives).  Cosine similarity is used, so the term is bounded
and cannot be minimised by shrinking every vector to zero.  Compared with the
unbounded squared-distance pull this replaces, it carries both attraction and
separation -- the earlier pull drove the codebook to collapse
(``assignment_floor.CodebookCollapse`` at epoch ~150).

Nothing here is tuned by hand:

* ``tau`` is set once from the first step's own logits, so the positive/negative
  logit gap is exactly one nat at initialisation, and is then fixed;
* ``lambda`` is set once from the first step's reconstruction loss, so the
  contrastive term is a fixed share of the total there (``GRAPH_SHARE``);
* negatives are degree-matched non-edges, drawn from the same degree bucket as the
  anchor and rejected if the pair is an edge.

The edges, degrees and triangle counts come from the project's single undirected
graph, and ``graph.npz``'s sha256 is checked against ``metadata.json`` before an
edge is read (CLAUDE.md 89b).
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

SCHEMA_PREFIX = "genrec.behavior_graph"
#: Share of the total loss the contrastive term takes at the first step (0.0 = off).
GRAPH_SHARE = 0.1
#: Anchors per step and negatives per anchor.
ANCHOR_SAMPLES = 1024
NEGATIVES_PER_ANCHOR = 8
#: Draws before an adjacent candidate is kept as a negative (then the slot is dropped).
NEGATIVE_RETRIES = 6
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
    """Edges, degrees and triangle counts, after the metadata contract verifies."""
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


def _neighbour_table(graph: dict):
    """CSR neighbour lists, so one neighbour per anchor is a vectorised draw."""
    rows, cols = graph["rows"], graph["cols"]
    left = np.concatenate([rows, cols])
    right = np.concatenate([cols, rows])
    order = np.lexsort((right, left))
    left, right = left[order], right[order]
    indptr = np.zeros(graph["n_nodes"] + 1, dtype=np.int64)
    np.add.at(indptr, left + 1, 1)
    indptr = np.cumsum(indptr)
    return indptr, right


class NegativeSampler:
    """Neighbours as positives, degree-matched non-neighbours as negatives."""

    def __init__(self, graph: dict):
        import scipy.sparse as sparse

        self.graph = graph
        self.degree = graph["degree"]
        self.n_nodes = graph["n_nodes"]
        self.indptr, self.neighbours = _neighbour_table(graph)
        rows, cols = graph["rows"], graph["cols"]
        left = np.concatenate([rows, cols])
        right = np.concatenate([cols, rows])
        self.adjacency = sparse.csr_matrix(
            (np.ones(len(left), dtype=bool), (left, right)),
            shape=(self.n_nodes, self.n_nodes))
        buckets = defaultdict(list)
        for node in range(self.n_nodes):
            if self.degree[node] > 0:
                buckets[_degree_bucket(int(self.degree[node]))].append(node)
        self.buckets = {key: np.asarray(values, dtype=np.int64) for key, values in buckets.items()}
        self.anchors = np.asarray([node for node in range(self.n_nodes) if self.degree[node] > 0],
                                  dtype=np.int64)
        self.anchor_bucket = np.asarray([_degree_bucket(int(self.degree[node]))
                                         for node in self.anchors], dtype=np.int64)

    def _neighbour(self, anchors: np.ndarray, rng) -> np.ndarray:
        start = self.indptr[anchors]
        length = self.indptr[anchors + 1] - start
        offset = (rng.random(len(anchors)) * length).astype(np.int64)
        return self.neighbours[start + offset]

    def _is_edge(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.asarray(self.adjacency[left, right]).ravel().astype(bool)

    def sample(self, generator: torch.Generator, count: int = ANCHOR_SAMPLES,
               negatives: int = NEGATIVES_PER_ANCHOR):
        """Anchors, one neighbour each, and degree-matched non-neighbour negatives."""
        rng = np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,), generator=generator).item()))
        index = rng.choice(len(self.anchors), size=min(count, len(self.anchors)), replace=False)
        anchors = self.anchors[index]
        buckets = self.anchor_bucket[index]
        positive = self._neighbour(anchors, rng)
        picked = np.full((len(anchors), negatives), -1, dtype=np.int64)
        filled = np.zeros(len(anchors), dtype=np.int64)
        for _ in range(negatives * NEGATIVE_RETRIES):
            open_rows = np.nonzero(filled < negatives)[0]
            if not len(open_rows):
                break
            candidates = np.empty(len(open_rows), dtype=np.int64)
            open_buckets = buckets[open_rows]
            for bucket in np.unique(open_buckets):
                mask = open_buckets == bucket
                pool = self.buckets[int(bucket)]
                candidates[mask] = pool[rng.integers(0, len(pool), size=int(mask.sum()))]
            bad = (candidates == anchors[open_rows]) | self._is_edge(anchors[open_rows], candidates)
            accepted = open_rows[~bad]
            if len(accepted):
                picked[accepted, filled[accepted]] = candidates[~bad]
                filled[accepted] += 1
        return anchors, positive, picked


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a / a.norm(dim=1, keepdim=True).clamp_min(EPS)
    b = b / b.norm(dim=1, keepdim=True).clamp_min(EPS)
    return (a * b).sum(dim=1)


def contrastive_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor,
                     temperature: float) -> torch.Tensor:
    """InfoNCE with one positive and one row of negatives per anchor.

    ``negative`` is ``(M, K, D)``; entries whose rows are entirely missing are
    dropped by the caller.
    """
    similarity = cosine(anchor, positive) / temperature
    negatives = torch.einsum("md,mkd->mk", anchor / anchor.norm(dim=1, keepdim=True).clamp_min(EPS),
                             negative / negative.norm(dim=2, keepdim=True).clamp_min(EPS))
    logits = torch.cat([similarity.unsqueeze(1), negatives / temperature], dim=1)
    return torch.nn.functional.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long,
                                                                 device=logits.device))


def calibrate_temperature(positive_similarity: float, negative_similarity: float) -> float:
    """The temperature that makes the initial logit gap one nat.

    The logit is ``similarity / tau``, so the gap is one nat exactly when
    ``tau`` equals the measured similarity gap.
    """
    gap = float(positive_similarity) - float(negative_similarity)
    return max(gap, 1e-3)


def calibrate_share(recon: float, graph: float, share: float = GRAPH_SHARE) -> float:
    """The weight that makes the contrastive term ``share`` of the total at this step."""
    share = float(share)
    return share * float(recon) / max((1.0 - share) * float(graph), EPS)
