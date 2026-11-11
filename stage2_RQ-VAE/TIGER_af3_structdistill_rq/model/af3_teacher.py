"""The frozen AF3 structural teacher of issue #181.

Built from TRAIN interactions only: an undirected weighted graph whose edge weights are ``1 + u_ij`` with ``u_ij`` the AF3 value
clipped into [0, 1] at the train graph's q05/q95, symmetrised. Nothing is deleted -- a bridge edge with a low or negative AF3
still carries weight, because the issue forbids reading a negative AF3 as a negative sample.

The teacher is ``T = (P + P^2) / 2`` with ``P = D^-1/2 W D^-1/2``: ``P`` keeps direct connections, ``P^2`` reaches two hops and
shared neighbourhoods. It is kept sparse; :func:`dense` exists for tests and small graphs only.
"""
from __future__ import annotations

import numpy as np
import torch


def af3_values(edges, degrees) -> dict:
    """``AF3(i, j) = 4 - deg(i) - deg(j) + 3 m_ij`` for every observed edge (``m_ij`` = 1 for a simple graph)."""
    return {(int(i), int(j)): 4.0 - float(degrees[i]) - float(degrees[j]) + 3.0 for i, j in edges}


def quantiles(af3: dict, q05: float = 0.05, q95: float = 0.95) -> tuple[float, float]:
    """The train graph's clipping quantiles, frozen into the manifest."""
    values = np.array(list(af3.values()), dtype=np.float64)
    return float(np.quantile(values, q05)), float(np.quantile(values, q95))


def weights(coo, n_items: int, q05: float, q95: float) -> torch.Tensor:
    """``W_ij = 1 + clip((AF3-q05)/(q95-q05), 0, 1)`` on observed edges, symmetric, sparse COO of shape ``[N, N]``."""
    edges = [(int(i), int(j)) for i, j in coo]
    degrees: dict[int, int] = {}
    for i, j in edges:
        degrees[i] = degrees.get(i, 0) + 1
        degrees[j] = degrees.get(j, 0) + 1
    af3 = af3_values(edges, degrees)
    span = max(q95 - q05, 1e-12)
    rows, cols, vals = [], [], []
    for (i, j), value in af3.items():
        u = min(max((value - q05) / span, 0.0), 1.0)
        w = 1.0 + u
        rows.extend((i, j)); cols.extend((j, i)); vals.extend((w, w))
    index = torch.tensor([rows, cols], dtype=torch.long)
    return torch.sparse_coo_tensor(index, torch.tensor(vals, dtype=torch.float64), (n_items, n_items)).coalesce()


def normalized(w: torch.Tensor) -> torch.Tensor:
    """``P = D^-1/2 W D^-1/2``; a zero-degree node keeps an all-zero row (the issue's explicit degenerate rule)."""
    dense = w.to_dense()
    degree = dense.sum(1)
    inv = torch.where(degree > 0, 1.0 / torch.sqrt(degree.clamp_min(1e-30)), torch.zeros_like(degree))
    return dense * inv.unsqueeze(1) * inv.unsqueeze(0)


def teacher(w: torch.Tensor) -> torch.Tensor:
    """``T = (P + P^2) / 2`` -- dense by construction, for tests and small graphs."""
    p = normalized(w)
    return 0.5 * (p + p @ p)


def build(coo, n_items: int) -> tuple[torch.Tensor, dict]:
    """The teacher and the manifest a checkpoint binds."""
    edges = [(int(i), int(j)) for i, j in coo]
    degrees: dict[int, int] = {}
    for i, j in edges:
        degrees[i] = degrees.get(i, 0) + 1
        degrees[j] = degrees.get(j, 0) + 1
    q05, q95 = quantiles(af3_values(edges, degrees))
    w = weights(coo, n_items, q05, q95)
    manifest = {"n_items": int(n_items), "edges": len(edges), "q05": q05, "q95": q95,
                "rule": "W=1+clip((AF3-q05)/(q95-q05),0,1) on observed edges, symmetrised, zero-degree rows stay zero"}
    return teacher(w), manifest


def build_sparse(coo, n_items: int) -> tuple[torch.Tensor, dict]:
    """The same teacher as :func:`build`, kept sparse so 24k items never materialise an N x N dense tensor.

    ``T = (P + P^2)/2`` with ``P`` sparse; ``P^2`` goes through ``torch.sparse.mm``. Identical to the dense result elementwise,
    which the tests check on a toy graph.
    """
    edges = [(int(i), int(j)) for i, j in coo]
    degrees: dict[int, int] = {}
    for i, j in edges:
        degrees[i] = degrees.get(i, 0) + 1
        degrees[j] = degrees.get(j, 0) + 1
    q05, q95 = quantiles(af3_values(edges, degrees))
    w = weights(coo, n_items, q05, q95)
    dense = w.to_dense()
    degree = dense.sum(1)
    inv = torch.where(degree > 0, 1.0 / torch.sqrt(degree.clamp_min(1e-30)), torch.zeros_like(degree))
    index = w.indices()
    values = w.values() * inv[index[0]] * inv[index[1]]
    p = torch.sparse_coo_tensor(index, values, (n_items, n_items)).coalesce()
    p2 = torch.sparse.mm(p, p)
    teacher_sparse = torch.sparse_coo_tensor(
        torch.cat((p.indices(), p2.indices()), dim=1),
        torch.cat((0.5 * p.values(), 0.5 * p2.values())),
        (n_items, n_items),
    ).coalesce()
    manifest = {"n_items": int(n_items), "edges": len(edges), "q05": q05, "q95": q95,
                "rule": "W=1+clip((AF3-q05)/(q95-q05),0,1) on observed edges, symmetrised; T=(P+P^2)/2 kept sparse"}
    return teacher_sparse, manifest


class SparseRows:
    """Row-wise lookup into a coalesced sparse matrix without materialising it dense.

    The coalesced indices are sorted lexicographically by ``(row, col)``, so a row is a contiguous slice found by binary search;
    gathering a row costs O(log nnz + cols log cols) instead of an N x N allocation.
    """

    def __init__(self, sparse: torch.Tensor):
        coalesced = sparse.coalesce()
        order = torch.argsort(coalesced.indices()[0] * sparse.shape[1] + coalesced.indices()[1])
        self.indices = coalesced.indices()[:, order]
        self.values = coalesced.values()[order]
        self.rows = self.indices[0]
        self.cols = self.indices[1]
        self.n_rows, self.n_cols = sparse.shape
        # first position of every row in the sorted order
        self.row_start = torch.searchsorted(self.rows.contiguous(), torch.arange(self.n_rows + 1))

    def gather(self, anchors: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((len(anchors), candidates.shape[1]), dtype=self.values.dtype)
        for a, anchor in enumerate(anchors.tolist()):
            lo, hi = int(self.row_start[anchor]), int(self.row_start[anchor + 1])
            if hi <= lo:
                continue
            row_cols = self.cols[lo:hi]
            row_vals = self.values[lo:hi]
            wanted = candidates[a]
            pos = torch.searchsorted(row_cols.contiguous(), wanted.contiguous())
            pos_clamped = pos.clamp(max=row_cols.numel() - 1)
            hit = (pos < row_cols.numel()) & (row_cols[pos_clamped] == wanted)
            out[a][hit] = row_vals[pos_clamped][hit]
        return out


def teacher_scores(teacher_sparse: torch.Tensor, anchors: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    """``[A, C]`` teacher relations for per-anchor candidate id lists, read out of the sparse teacher.

    Uses :class:`SparseRows`, so nothing is densified; the caller may cache the lookup across steps.
    """
    return SparseRows(teacher_sparse).gather(anchors, candidates)
