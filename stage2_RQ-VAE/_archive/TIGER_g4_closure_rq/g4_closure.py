"""Issue #216: four arms over the matched 4-cycle / branch pool.

    A Edge         plain edge InfoNCE (degree-matched non-edges as negatives)
    B Context-Pair pairwise InfoNCE on the frozen (b, c) pairs: y=1 ranks above y=0
    C Context-Add  triplet classifier, node-only aggregation (no pairwise products)
    D Context-Interact same head capacity plus explicit pairwise interactions

Every arm draws from the same frozen pool, uses the same encoder position (z), the
same seed and the same calibration rule: the auxiliary weight is set once on the
first 20 steps so that the auxiliary term's **gradient norm on the encoder** is 10%
of the RQ-VAE term's gradient norm (the issue's budget definition, not the loss
share).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

G4_ARM = "D"
SHARE = 0.10
CALIBRATION_STEPS = 20
PAIRS_PER_STEP = 512
EPS = 1e-12

ARM_EDGE = "A"
ARM_PAIR = "B"
ARM_ADD = "C"
ARM_INTERACT = "D"


def repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "stage2_RQ-VAE").is_dir() and (parent / "results").is_dir():
            return parent
    raise RuntimeError("repository root not found above " + __file__)


POOL_DIR = repo_root() / "results" / "research" / "g4_closure"
GRAPH_DIR = repo_root() / "results" / "research" / "behavior_graph"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pool(split: str = "train") -> dict:
    """The frozen closure/branch pool for one split, with its audit files checked."""
    path = POOL_DIR / f"pairs_{split}.npz"
    coverage_path = POOL_DIR / "coverage.json"
    if not coverage_path.is_file() or not path.is_file():
        raise RuntimeError("no %s pool under %s; run analysis_sid_recall/build_g4_closure.py"
                           % (split, POOL_DIR))
    coverage = json.loads(coverage_path.read_text())
    expected = (coverage.get("pairs_sha") or {}).get(split)
    if not path.is_file() or expected is None:
        raise RuntimeError("no %s pool under %s; run analysis_sid_recall/build_g4_closure.py"
                           % (split, POOL_DIR))
    digest = _sha256(path)
    if digest != expected:
        raise RuntimeError("%s pool sha256 %s does not match the audit %s"
                           % (split, digest[:16], expected[:16]))
    payload = np.load(path, allow_pickle=False)
    return {"triples": np.asarray(payload["triples"], dtype=np.int64),
            "degrees": np.asarray(payload["degrees"], dtype=np.int64),
            "labels": np.asarray(payload["labels"], dtype=np.int64),
            "sha256": digest}


def load_graph() -> dict:
    """The canonical graph, hash-checked, for the edge arm's negatives."""
    metadata = json.loads((GRAPH_DIR / "metadata.json").read_text())
    record = (metadata.get("outputs") or {}).get("graph.npz")
    graph_path = GRAPH_DIR / "graph.npz"
    digest = _sha256(graph_path)
    if not record or digest != record.get("sha256"):
        raise RuntimeError("graph.npz does not match metadata.json; rebuild the graph")
    with np.load(graph_path, allow_pickle=False) as payload:
        rows = np.asarray(payload["rows"], dtype=np.int64)
        cols = np.asarray(payload["cols"], dtype=np.int64)
        n = int(payload["nodes"])
    degree = np.bincount(np.concatenate([rows, cols]), minlength=n)
    return {"rows": rows, "cols": cols, "degree": degree, "n_nodes": n,
            "edges": {(int(a), int(b)) for a, b in zip(rows, cols)}
                     | {(int(b), int(a)) for a, b in zip(rows, cols)}}


def _degree_bucket(value: int) -> int:
    for limit in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000):
        if value <= limit:
            return limit
    return 10_000


class EdgeSampler:
    """Behaviour edges and degree-matched non-edges for arm A."""

    def __init__(self, graph: dict):
        self.graph = graph
        buckets = {}
        for node, value in enumerate(graph["degree"].tolist()):
            if value > 0:
                buckets.setdefault(_degree_bucket(value), []).append(node)
        self.buckets = {key: np.asarray(values, dtype=np.int64) for key, values in buckets.items()}
        # Membership probes over a bucket are what the rejection loop used to do; the
        # degree buckets of this graph are dense cliques, so 99% of random draws were
        # rejected and a step could end up with zero pairs.  One pass over the 630k
        # directed edges turns every probe into a set lookup.
        self.neighbours = [set() for _ in range(graph["n_nodes"])]
        for source, target in graph["edges"]:
            self.neighbours[int(source)].add(int(target))

    def sample(self, generator: torch.Generator, count: int = PAIRS_PER_STEP):
        rng = np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,), generator=generator).item()))
        rows, cols = self.graph["rows"], self.graph["cols"]
        index = rng.integers(0, len(rows), size=count)
        positives = np.stack([rows[index], cols[index]], axis=1)
        negatives = np.empty_like(positives)
        # Each row is visited once: an anchor whose degree pool is fully adjacent to it
        # is swapped for the last still-pending row.  Retrying the same anchor until the
        # attempt budget ran out is what produced steps with zero pairs.
        accepted, pending = 0, int(count)
        attempts = 0
        while accepted < count and pending > 0 and attempts < count * 20:
            attempts += 1
            anchor = int(positives[accepted, 0])
            pool = self.buckets.get(_degree_bucket(int(self.graph["degree"][anchor])))
            candidate = None if pool is None else self._non_neighbour(anchor, pool, rng)
            if candidate is None:
                pending -= 1
                positives[accepted] = positives[pending]
                continue
            negatives[accepted] = (anchor, candidate)
            accepted += 1
        # Both sides line up: the trainer pairs row i with row i.
        return positives[:accepted], negatives[:accepted]


    def _non_neighbour(self, anchor: int, pool: np.ndarray, rng: np.random.Generator):
        """A member of ``pool`` that is neither the anchor nor one of its neighbours."""
        neighbours = self.neighbours[anchor]
        start = int(rng.integers(0, len(pool)))
        for offset in range(min(len(pool), 32)):
            candidate = int(pool[(start + offset) % len(pool)])
            if candidate != anchor and candidate not in neighbours:
                return candidate
        candidates = [int(value) for value in pool if value != anchor and int(value) not in neighbours]
        return candidates[int(rng.integers(0, len(candidates)))] if candidates else None


class PairSampler:
    """Frozen (a, b, c) triples with their closure labels, balanced per step."""

    def __init__(self, pool: dict):
        triples = pool["triples"]
        labels = pool["labels"]
        self.positive = triples[labels == 1]
        self.negative = triples[labels == 0]

    def sample(self, generator: torch.Generator, count: int = PAIRS_PER_STEP):
        rng = np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,), generator=generator).item()))
        take = min(count, len(self.positive), len(self.negative))
        pos = self.positive[rng.integers(0, len(self.positive), size=take)]
        neg = self.negative[rng.integers(0, len(self.negative), size=take)]
        return pos, neg


class TripletPredictor(nn.Module):
    """Permutation-invariant classifier; `interaction=False` is arm C."""

    def __init__(self, dim: int = 32, hidden: int = 64, interaction: bool = True):
        super().__init__()
        self.pair = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU()) if interaction else None
        self.node = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden * (2 if interaction else 1), hidden),
                                  nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, first, second, third):
        nodes = self.node(torch.stack([first, second, third], dim=1)).mean(dim=1)
        if self.pair is not None:
            nodes = torch.cat([nodes, self.pair((first * second + first * third + second * third) / 3)],
                              dim=1)
        return self.head(nodes).squeeze(1)


def cosine(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first = first / first.norm(dim=1, keepdim=True).clamp_min(EPS)
    second = second / second.norm(dim=1, keepdim=True).clamp_min(EPS)
    return (first * second).sum(dim=1)


def pairwise_infonce(positive_similarity: torch.Tensor, negative_similarity: torch.Tensor,
                     temperature: float) -> torch.Tensor:
    logits = torch.stack([positive_similarity / temperature, negative_similarity / temperature], dim=1)
    return F.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long, device=logits.device))


def triplet_bce(logits_positive: torch.Tensor, logits_negative: torch.Tensor) -> torch.Tensor:
    return (F.binary_cross_entropy_with_logits(logits_positive, torch.ones_like(logits_positive))
            + F.binary_cross_entropy_with_logits(logits_negative, torch.zeros_like(logits_negative))) / 2


def calibrate_weight(norm_rq: float, norm_aux: float, share: float = SHARE) -> float:
    """The weight that makes the auxiliary gradient 10% of the RQ-VAE gradient."""
    return float(share) * float(norm_rq) / max(float(norm_aux), EPS)


def calibrate_temperature(positive, negative, floor: float = 1e-3) -> float:
    """A data-derived temperature: the median gap, else the spread of the scores.

    A fixed floor turns a zero gap into logits of +-1000 (one run showed an
    auxiliary loss of 21 and a weight of 3e-9), so the spread carries the scale
    until the gap emerges -- the same defect that was fixed in E2.
    """
    positive = torch.as_tensor(positive).detach().float()
    negative = torch.as_tensor(negative).detach().float()
    gap = float((positive - negative).abs().median())
    spread = float(torch.cat([positive, negative]).std())
    return max(gap, spread, floor)
