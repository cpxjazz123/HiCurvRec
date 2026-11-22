"""Chain alignment: level-2 soft sharing for directed ``A -> B -> C`` chains.

Mechanism surface (CLAUDE.md 66): this module owns a behaviour statistic and a new loss.
It reads the canonical **directed** graph of Issue #221 (train sequences only), enumerates
strictly induced chains, and pushes the level-2 soft assignment so that adjacent pairs share
more than the two-hop pair:

    L = max(0, m + S2(A,C) - (S2(A,B) + S2(B,C)) / 2),      S2 = <p2, p2>

``S2`` deliberately does **not** multiply the level-1 agreement: with a 1e-3 level-1 share the
product attenuates the gradient by the same factor and the term becomes inert (measured).

The term never adds to the reported RQ-VAE loss; the trainer adds its gradient with a budget
measured against the reconstruction gradient's norm, so the optimisation recipe does not move.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

MARGIN = 0.5
BUDGET = 0.10
CAP = 60000
MIN_CHAINS = 1


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "stage2_RQ-VAE").is_dir() and (parent / "results").is_dir():
            return parent
    raise RuntimeError("repository root not found above " + __file__)


def _chain_instances(root: Path) -> np.ndarray:
    payload = np.load(root / "results" / "research" / "behavior_graph_directed" / "graph.npz")
    rows = payload["rows"].astype(np.int64)
    cols = payload["cols"].astype(np.int64)
    outgoing: dict = {}
    for a, b in zip(rows.tolist(), cols.tolist()):
        outgoing.setdefault(a, set()).add(b)
    rng = np.random.default_rng(42)
    found = []
    for index in rng.permutation(len(rows)):
        a, b = int(rows[index]), int(cols[index])
        for c in list(outgoing.get(b, ()))[:64]:
            c = int(c)
            if c == a or a in outgoing.get(c, ()) or c in outgoing.get(a, ()) \
                    or b in outgoing.get(c, ()):
                continue
            found.append((a, b, c))
            if len(found) >= CAP:
                return np.array(found, dtype=np.int64)
    return np.array(found, dtype=np.int64)


class ChainAlignment:
    """Behaviour term for the trainer; it computes a loss, never a schedule."""

    def __init__(self, margin: float = MARGIN, budget: float = BUDGET):
        self.margin = float(margin)
        self.budget = float(budget)
        self.instances = _chain_instances(_repo_root())
        self.by_first: dict = {}
        for a, b, c in self.instances.tolist():
            self.by_first.setdefault(int(a), []).append((int(a), int(b), int(c)))

    def _present(self, ids: torch.Tensor) -> np.ndarray:
        members = set(int(value) for value in ids.tolist())
        triples = [triple for item in members for triple in self.by_first.get(item, ())
                   if triple[1] in members and triple[2] in members]
        if not triples:
            return np.zeros((0, 3), dtype=np.int64)
        return np.array(triples, dtype=np.int64)

    def _level2(self, model, embeddings: torch.Tensor) -> torch.Tensor:
        """Second-level soft assignment probabilities, temperature from this batch."""
        residual = model.encoder(embeddings)
        probabilities = None
        for level, layer in enumerate(model.rq.vq_layers):
            table = layer.get_code_embs()
            distances = torch.cdist(residual, table)
            temperature = distances.min(dim=1).values.median().clamp(min=1e-6)
            probabilities = torch.softmax(-distances / temperature, dim=-1)
            if level >= 1:
                break
            residual = residual - probabilities @ table
        return probabilities

    def loss(self, model, ids: torch.Tensor, embeddings: torch.Tensor):
        """Return (loss, chains) for this batch, or (None, 0) when too few chains fit."""
        triples = self._present(ids)
        if len(triples) < MIN_CHAINS:
            return None, int(len(triples))
        position = {int(value): row for row, value in enumerate(ids.tolist())}
        rows = np.array([[position[a], position[b], position[c]] for a, b, c in triples.tolist()],
                        dtype=np.int64)
        probabilities = self._level2(model, embeddings)
        a = torch.from_numpy(rows[:, 0]).to(probabilities.device)
        b = torch.from_numpy(rows[:, 1]).to(probabilities.device)
        c = torch.from_numpy(rows[:, 2]).to(probabilities.device)
        share = lambda left, right: (probabilities[left] * probabilities[right]).sum(-1)
        hinge = torch.clamp(self.margin + share(a, c) - (share(a, b) + share(b, c)) / 2.0,
                            min=0.0)
        return hinge.mean(), int(len(triples))
