"""Level-1 behaviour structure: adjacent items share their encoder neighbourhood.

    L = L_RQVAE + lambda * L_graph^(1)
    L_graph^(1) = mean ||z_i - z_j||^2 / mean ||z||^2      over behaviour edges

The edges come from the project's single undirected graph
(``results/research/behavior_graph``, rule 89b) and the file's sha256 is checked
against ``metadata.json`` before a single edge is read -- a rebuilt or edited
graph is refused instead of silently changing the target.

``lambda`` is not tuned.  ``GRAPH_SHARE`` fixes the share of the total loss the
term takes at the first step, and ``calibrate_share`` turns that share into the
weight from the two measured values at that step:

    lambda = share * L_recon(step 0) / ((1 - share) * L_graph(step 0))

so the graph term is exactly 10% of the total loss at initialisation, whatever
the scale of the reconstruction loss happens to be, and every later step uses the
same weight.  The weight, the term and the reconstruction are written to the
metrics stream, so the run can be audited without recomputation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

SCHEMA_PREFIX = "genrec.behavior_graph"
#: Share of the total loss the graph term takes at the first step (switch: 0.0 = off).
GRAPH_SHARE = 0.1
#: Edges drawn per optimisation step; one edge per vector slot of the batch.
EDGE_SAMPLES = 4096
#: The numeric floor of the normaliser and of the calibrated weight.
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


def load_edges(folder: Path | str | None = None) -> np.ndarray:
    """The graph's undirected edges, after the metadata contract verifies."""
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
    return np.stack([rows, cols], axis=1)


def sample_edges(edges: np.ndarray, count: int, generator: torch.Generator) -> np.ndarray:
    """A bounded, reproducible draw from the edge set."""
    if len(edges) <= count:
        return edges
    index = torch.randint(0, len(edges), (count,), generator=generator).numpy()
    return edges[index]


def graph_level1_loss(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Dimensionless squared distance between neighbouring encoded items."""
    distance = (left - right).pow(2).sum(dim=1).mean()
    scale = 0.5 * (left.pow(2).sum(dim=1).mean() + right.pow(2).sum(dim=1).mean())
    return distance / (scale + EPS)


def calibrate_share(recon: float, graph: float, share: float = GRAPH_SHARE) -> float:
    """The weight that makes the graph term ``share`` of the *total* at this step.

    ``lambda * g = share * (r + lambda * g)`` gives ``lambda = share * r / ((1 - share) * g)``;
    using ``share * r / g`` instead would make the term a share of the
    reconstruction alone and overshoot the total by ``1 / (1 - share)``.
    """
    share = float(share)
    return share * float(recon) / max((1.0 - share) * float(graph), EPS)
