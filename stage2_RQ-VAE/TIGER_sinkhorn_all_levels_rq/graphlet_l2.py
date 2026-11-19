"""E2: graphlet supervision on L2, inside the L1 group.

    L_graphlet2 = -log softmax( S2(a,p)/tau , S2(a,n)/tau )
    S2(x,y)     = sum_k p2_x(k) p2_y(k)          (k over the 256 level-2 codewords)

The pairs are drawn **inside one L1 group**: the positive is a behaviour neighbour
of the anchor that already shares its L1 code, and the negative is a degree-matched
non-neighbour that shares the same L1 code.  That makes the negatives hard -- the
question is no longer "is this item related at all" (L1 answered it) but "inside
this coarse group, which items does the local structure tie together".

Level-2 probabilities are the level-2 quantiser distances, centred as
``VQLayer.center_distance`` centres them, through ``softmax(-d / SK_EPSILON)``, so
the term sees the same geometry the assignment does.

Groups come from the model's current hard L1 codes, rebuilt every
``REFRESH_EPOCHS`` epochs (the codes move early and stabilise later); the pool of
candidate pairs is rebuilt with them.  Nothing is hand-tuned: ``tau`` is the
measured initial gap and ``lambda`` the calibrated share of the first step's total.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

SCHEMA_PREFIX = "genrec.behavior_graph"
GRAPHLET2_SHARE = 0.1
PAIRS_PER_STEP = 512
POOL_TARGET = 20000
REFRESH_EPOCHS = 50
#: Steps whose reconstruction/term medians calibrate the weight.  A single first
#: step can be an initialisation outlier (it made lambda 60x too small once), so the
#: weight is taken from the median of the first steps instead.
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
        n_nodes = int(payload["nodes"])
    degree = np.bincount(np.concatenate([rows, cols]), minlength=n_nodes)
    return {"rows": rows, "cols": cols, "degree": degree, "n_nodes": n_nodes}


def _degree_bucket(value: int) -> int:
    for limit in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000):
        if value <= limit:
            return limit
    return 10_000


class GroupPairSampler:
    """Neighbour positives and structural hard negatives inside one L1 group."""

    def __init__(self, graph: dict):
        self.degree = graph["degree"]
        self.n_nodes = graph["n_nodes"]
        neighbours = defaultdict(set)
        for left, right in zip(graph["rows"].tolist(), graph["cols"].tolist()):
            neighbours[left].add(right)
            neighbours[right].add(left)
        self.adjacency = {node: np.asarray(sorted(values), dtype=np.int64)
                          for node, values in neighbours.items()}
        self.pool = None
        self.codes = None

    def rebuild(self, hard_l1: np.ndarray, target: int = POOL_TARGET):
        """Group items by their current L1 code and freeze a pool of triples."""
        self.codes = hard_l1
        groups = defaultdict(list)
        for node, code in enumerate(hard_l1):
            if code >= 0 and node in self.adjacency:
                groups[int(code)].append(node)
        pool = []
        for code, items in groups.items():
            members = set(items)
            ordered = sorted(members)
            for anchor in ordered:
                inside = [int(n) for n in self.adjacency[anchor] if n in members]
                if not inside:
                    continue
                outside = [node for node in ordered
                           if node != anchor and node not in self.adjacency[anchor]]
                if not outside:
                    continue
                positive = inside[len(pool) % len(inside)]
                negative = outside[len(pool) % len(outside)]
                pool.append((anchor, positive, negative))
                if len(pool) >= target:
                    self.pool = np.asarray(pool, dtype=np.int64)
                    return
        self.pool = np.asarray(pool, dtype=np.int64) if pool else np.zeros((0, 3), np.int64)

    def sample(self, generator: torch.Generator, count: int = PAIRS_PER_STEP):
        if self.pool is None or len(self.pool) == 0:
            raise RuntimeError("the L2 group pool is empty; no L1 group has both a neighbour "
                               "and a non-neighbour")
        rng = np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,), generator=generator).item()))
        index = rng.integers(0, len(self.pool), size=min(count, len(self.pool)))
        return self.pool[index]


def soft_assignment(distances: torch.Tensor, epsilon: float) -> torch.Tensor:
    maximum = distances.max()
    minimum = distances.min()
    middle = (maximum + minimum) / 2
    amplitude = (maximum - middle) + 1e-5
    return torch.softmax(-((distances - middle) / amplitude) / epsilon, dim=1)


def shared_probability(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """S2: the chance both items take the same level-2 codeword."""
    return (first * second).sum(dim=1)


def graphlet2_infonce(positive_s2: torch.Tensor, negative_s2: torch.Tensor,
                      temperature: float) -> torch.Tensor:
    logits = torch.cat([(positive_s2 / temperature).unsqueeze(1),
                        (negative_s2 / temperature).unsqueeze(1)], dim=1)
    return torch.nn.functional.cross_entropy(
        logits, torch.zeros(len(logits), dtype=torch.long, device=logits.device))


def calibrate_temperature(positive_mean: float, negative_mean: float) -> float:
    return max(float(positive_mean) - float(negative_mean), 1e-3)


def calibrate_share(recon: float, term: float, share: float = GRAPHLET2_SHARE) -> float:
    share = float(share)
    return share * float(recon) / max((1.0 - share) * float(term), EPS)


def l2_agreement(codebook: torch.Tensor, latent: torch.Tensor, hard_codes: np.ndarray,
                 epsilon: float) -> float:
    distances = ((latent[:, None, :] - codebook[None, :, :]) ** 2).sum(dim=2)
    return float((soft_assignment(distances, epsilon).argmax(dim=1).detach().cpu().numpy()
                  == hard_codes).mean())
