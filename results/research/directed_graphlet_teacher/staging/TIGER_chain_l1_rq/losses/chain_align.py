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
ALIGN = 0.001
CAP = 60000
MIN_CHAINS = 1

#: Structural offset for the next forward pass, indexed like the batch's rows.
PENDING_OFFSET = None


def install_offset(layers, align: float = ALIGN) -> None:
    """Add the structural offset inside the arm's own level-1 Sinkhorn cost.

    ``VQLayer._indices`` runs a balanced Sinkhorn on ``_distances`` during training, so adding
    the offset here puts the behaviour term inside the balanced assignment (CLAUDE.md 19h)
    rather than beside it.  The offset is scaled by the row's own distance spread and is
    mean-centred, so it cannot raise every item's preference for one codeword.
    """
    first = layers[0]
    original = first._distances

    def patched(latent):
        base = original(latent)
        if PENDING_OFFSET is None or PENDING_OFFSET.shape != base.shape:
            return base
        offset = PENDING_OFFSET.to(base.device)
        offset = (offset - offset.mean(dim=1, keepdim=True)) / offset.std(
            dim=1, keepdim=True).clamp(min=1e-6)
        spread = base.std(dim=1, keepdim=True).clamp(min=1e-6)
        return base + align * offset * spread

    first._distances = patched


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
        payload = np.load(_repo_root() / "results" / "research" / "behavior_graph_directed"
                          / "graph.npz")
        self.items = int(payload["nodes"])
        self.table = np.full((self.items, 3), -1, dtype=np.int64)
        self.partners: dict = {}
        for a, b, c in self.instances.tolist():
            self.partners.setdefault(int(a), set()).update((int(b), int(c)))
            self.partners.setdefault(int(b), set()).update((int(a), int(c)))
            self.partners.setdefault(int(c), set()).update((int(a), int(b)))
        self.partners = {key: sorted(value) for key, value in self.partners.items()}
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

    def remember(self, ids: torch.Tensor, tokens: torch.Tensor) -> None:
        """Store this batch's discrete codes so neighbours can be looked up by item id."""
        self.table[ids.detach().cpu().numpy()] = tokens.detach().cpu().numpy()

    def prepare(self, ids: torch.Tensor, codes: int = 256) -> None:
        """Build the next forward's structural offset from the neighbours' current codes."""
        global PENDING_OFFSET
        rows = ids.detach().cpu().numpy()
        prior = np.ones((len(rows), codes), dtype=np.float64)
        for row, item in enumerate(rows.tolist()):
            seen = 0
            for partner in self.partners.get(int(item), ()):
                code = int(self.table[partner, 0])
                if code >= 0:
                    prior[row, code] += 1.0
                    seen += 1
            if seen:
                prior[row] /= prior[row].sum()
        prior /= prior.sum(axis=1, keepdims=True)
        PENDING_OFFSET = torch.from_numpy(-np.log(prior).astype(np.float32))

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
