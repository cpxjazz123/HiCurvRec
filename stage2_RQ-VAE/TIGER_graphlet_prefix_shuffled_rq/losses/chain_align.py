"""Per-structure alignment: one margin per graphlet, applied at level 2.

Each structure states its own ordering of the pairwise L2 sharing, and the rules are applied
together with per-structure weights:

* chain  ``A->B->C``     : S2(A,B), S2(B,C) above S2(A,C);
* fork   ``A->{B,C}``    : S2(B,C) above S2(A,B), S2(A,C) -- the direction that the 31-arm
  survey shows *every* full-training arm violates (fork delta is negative everywhere), so this
  rule is the one that has to reverse a stable bias;
* merge  ``{B,C}->A``    : S2(B,C) above S2(B,A), S2(C,A) -- the same direction as the chain;
* cycle  ``A->B->C->A``  : the three pairs should be equally close (spread penalty); identity is
  kept by the L3 collision gate rather than by a pairwise ordering.

``S2 = <p2, p2>`` alone: multiplying by the level-1 agreement (1e-3 in practice) attenuates the
gradient by that factor and was measured to make the term inert.  No level-1 offset is used:
measured to cost capacity (22.8% of items change codeword even at weight 0.001).

The trainer injects this gradient after the RQ-VAE backward, capped at a fraction of the
reconstruction gradient, so the reported loss and the optimisation recipe do not move.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

MARGIN = 0.5
BUDGET = 0.10
CAP = 60000
MIN_INSTANCES = 1
WEIGHTS = {"chain": 1.0, "fork": 1.0, "merge": 0.5, "cycle": 0.5}


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "stage2_RQ-VAE").is_dir() and (parent / "results").is_dir():
            return parent
    raise RuntimeError("repository root not found above " + __file__)


def _banks(root: Path) -> dict:
    payload = np.load(root / "results" / "research" / "behavior_graph_directed" / "graph.npz")
    rows = payload["rows"].astype(np.int64)
    cols = payload["cols"].astype(np.int64)
    outgoing: dict = {}
    incoming: dict = {}
    for a, b in zip(rows.tolist(), cols.tolist()):
        outgoing.setdefault(a, set()).add(b)
        incoming.setdefault(b, set()).add(a)
    rng = np.random.default_rng(42)

    def has(table, a, b) -> bool:
        return b in table.get(a, ())

    chains, forks, merges = [], [], []
    for index in rng.permutation(len(rows)):
        a, b = int(rows[index]), int(cols[index])
        for c in list(outgoing.get(b, ()))[:64]:
            c = int(c)
            if c == a or has(outgoing, c, a) or has(outgoing, a, c) or has(outgoing, c, b):
                continue
            chains.append((a, b, c))
            if len(chains) >= CAP:
                break
        if len(chains) >= CAP:
            break
    for centre in rng.permutation(len(outgoing)):
        centre = int(centre)
        children = sorted(outgoing.get(centre, ()))
        if len(children) < 2:
            continue
        for first in range(len(children)):
            for second in range(first + 1, len(children)):
                b, c = children[first], children[second]
                if has(outgoing, b, c) or has(outgoing, c, b):
                    continue
                if has(outgoing, b, centre) or has(outgoing, c, centre):
                    continue
                forks.append((centre, b, c))
                if len(forks) >= CAP:
                    break
            if len(forks) >= CAP:
                break
        if len(forks) >= CAP:
            break
    for centre in rng.permutation(len(incoming)):
        centre = int(centre)
        parents = sorted(incoming.get(centre, ()))
        if len(parents) < 2:
            continue
        for first in range(len(parents)):
            for second in range(first + 1, len(parents)):
                b, c = parents[first], parents[second]
                if has(outgoing, b, c) or has(outgoing, c, b):
                    continue
                if has(outgoing, centre, b) or has(outgoing, centre, c):
                    continue
                merges.append((b, c, centre))
                if len(merges) >= CAP:
                    break
            if len(merges) >= CAP:
                break
        if len(merges) >= CAP:
            break
    cycles = []
    for index in rng.permutation(len(rows)):
        a, b = int(rows[index]), int(cols[index])
        for c in list(outgoing.get(b, ()))[:64]:
            c = int(c)
            if c == a or not has(outgoing, c, a):
                continue
            # strict 3-cycle: the reverse edges of the triangle must all be absent
            if has(outgoing, b, a) or has(outgoing, a, c) or has(outgoing, c, b):
                continue
            cycles.append((a, b, c))
            if len(cycles) >= CAP:
                break
        if len(cycles) >= CAP:
            break
    return {"chain": np.array(chains, dtype=np.int64),
            "fork": np.array(forks, dtype=np.int64),
            "merge": np.array(merges, dtype=np.int64),
            "cycle": np.array(cycles, dtype=np.int64)}


class ChainAlignment:
    """Behaviour term for the trainer; it computes a loss, never a schedule."""

    def __init__(self, margin: float = MARGIN, budget: float = BUDGET):
        self.margin = float(margin)
        self.budget = float(budget)
        self.banks = _banks(_repo_root())
        self.by_first: dict = {}
        for kind, bank in self.banks.items():
            index: dict = {}
            for triple in bank.tolist():
                index.setdefault(int(triple[0]), []).append(tuple(int(v) for v in triple))
            self.by_first[kind] = index

    def counts(self) -> dict:
        return {kind: int(len(bank)) for kind, bank in self.banks.items()}

    def _present(self, kind: str, members) -> np.ndarray:
        index = self.by_first[kind]
        triples = [triple for item in members for triple in index.get(item, ())
                   if triple[1] in members and triple[2] in members]
        if not triples:
            return np.zeros((0, 3), dtype=np.int64)
        return np.array(triples, dtype=np.int64)

    def _level2(self, model, embeddings: torch.Tensor) -> torch.Tensor:
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
        """Weighted sum of the per-structure margins over the instances present here."""
        members = set(int(value) for value in ids.tolist())
        found = {kind: self._present(kind, members) for kind in self.banks}
        total = sum(len(bank) for bank in found.values())
        if total < MIN_INSTANCES:
            return None, 0
        position = {int(value): row for row, value in enumerate(ids.tolist())}
        probabilities = self._level2(model, embeddings)
        device = probabilities.device
        share = lambda left, right: (probabilities[left] * probabilities[right]).sum(-1)
        parts, hinges = [], {}
        for kind, bank in found.items():
            if len(bank) == 0:
                continue
            rows = np.array([[position[a], position[b], position[c]]
                             for a, b, c in bank.tolist()], dtype=np.int64)
            a = torch.from_numpy(rows[:, 0]).to(device)
            b = torch.from_numpy(rows[:, 1]).to(device)
            c = torch.from_numpy(rows[:, 2]).to(device)
            s_ab, s_bc, s_ac = share(a, b), share(b, c), share(a, c)
            if kind == "cycle":
                # a 3-cycle wants the three pairs equally close, not one pair above another:
                # penalise the spread instead of stating an ordering
                stacked = torch.stack([s_ab, s_bc, s_ac])
                hinge = (stacked.max(dim=0).values - stacked.min(dim=0).values)
            elif kind == "chain":
                hinge = torch.clamp(self.margin + s_ac - (s_ab + s_bc) / 2.0, min=0.0)
            else:
                # fork triple (centre, b, c): siblings are b-c; merge triple (b, c, centre):
                # siblings are a-b.  Both want the sibling pair above the parent pairs.
                sibling = s_bc if kind == "fork" else s_ab
                parents = ((s_ab + s_ac) / 2.0) if kind == "fork" else ((s_ac + s_bc) / 2.0)
                hinge = torch.clamp(self.margin + parents - sibling, min=0.0)
            hinges[kind] = float(hinge.mean().detach())
            parts.append(WEIGHTS.get(kind, 1.0) * hinge.mean())
        if not parts:
            return None, 0
        return torch.stack(parts).sum(), total
