"""The batch-restricted AF3 supervision the trainer calls once per step (issue #155).

The frozen edge table speaks in graph nodes and the loader speaks in embedding rows, so the
runtime resolves rows to item ids exactly (model.item_index) and keeps only the edges and closed
triangles whose endpoints are both inside the batch. Anything else would supervise against
neighbours the batch never saw.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from . import item_index as ii
from . import target_distance as td


class StructureSupervision:
    """Holds the frozen table and turns a batch of rows into the two loss terms."""

    def __init__(self, edge_items: np.ndarray, edge_u: np.ndarray,
                 triples_items: np.ndarray, d_ref: float, row_index: dict,
                 node_items: dict):
        self.edge_items = edge_items
        self.edge_u = edge_u
        self.triples_items = triples_items
        self.d_ref = float(d_ref)
        self.row_index = row_index
        self.node_items = node_items

    @classmethod
    def from_repo(cls, repo_root: Path, d_ref: float) -> "StructureSupervision":
        repo_root = Path(repo_root)
        table = json.loads((repo_root / "results" / "research" / "issue155"
                            / "af3_edges.json").read_text())
        raw = np.load(repo_root / "results" / "research" / "issue150" / "structure_prior.npz",
                      allow_pickle=True)
        node_items = {index: int(item) for index, item in enumerate(raw["item_ids"])}
        rows, u = [], []
        for (tail, head), value in zip(table["edges"], table["u"]):
            left, right = node_items.get(int(tail)), node_items.get(int(head))
            if left is None or right is None:
                continue
            rows.append((left, right))
            u.append(float(value))
        edges = np.asarray(rows, dtype=np.int64)
        edge_u = np.asarray(u, dtype=np.float64)
        high = set(tuple(map(int, edge)) for edge in
                   edges[np.isclose(edge_u, 1.0, atol=1e-9)])
        low = set(tuple(map(int, edge)) for edge in
                  edges[np.isclose(edge_u, 0.0, atol=1e-9)])
        triples = []
        for triangle in raw["triangles"].tolist():
            items = [node_items.get(int(node)) for node in triangle]
            if any(item is None for item in items):
                continue
            members = [(items[0], items[1]), (items[1], items[2]), (items[0], items[2])]
            if all(member in high for member in members):
                triples.append(items)
        embeddings = np.load(repo_root / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy")
        return cls(edges, edge_u, np.asarray(triples, dtype=np.int64), d_ref,
                   ii.build(embeddings), node_items)

    def batch_edges(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The edges (and closed triangles) whose endpoints both lie in this batch."""
        items = ii.lookup(rows, self.row_index).numpy()
        position = {int(item): index for index, item in enumerate(items)}
        inside = np.zeros(int(max(self.edge_items.max(), items.max())) + 1, dtype=bool)
        inside[items] = True
        mask = inside[self.edge_items[:, 0]] & inside[self.edge_items[:, 1]]
        pairs = torch.tensor([(position[int(a)], position[int(b)])
                              for a, b in self.edge_items[mask]], dtype=torch.long)
        targets = torch.tensor(self.edge_u[mask], dtype=torch.float64)
        if self.triples_items.size:
            tri_mask = np.array([all(int(item) in position for item in triangle)
                                 for triangle in self.triples_items], dtype=bool)
            triples = torch.tensor([[position[int(item)] for item in triangle]
                                    for triangle in self.triples_items[tri_mask]],
                                   dtype=torch.long)
            if triples.numel() == 0:
                triples = torch.zeros(0, 3, dtype=torch.long)
        else:
            triples = torch.zeros(0, 3, dtype=torch.long)
        return pairs, targets, triples

    def terms(self, model, rows: torch.Tensor, curvature: float = 1.0):
        """``L_struct + L_tri`` for this batch, or ``None`` when the batch carries no edge."""
        pairs, targets, triples = self.batch_edges(rows)
        if pairs.numel() == 0:
            return None
        losses = td.structure_losses(model.rq.last_prefixes, pairs,
                                     td.target_distance(targets, self.d_ref), triples,
                                     torch.tensor(self.d_ref), curvature)
        return losses["struct"] + losses["tri"]
