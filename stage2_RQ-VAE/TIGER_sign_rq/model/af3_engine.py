"""The exact structural-distortion engine: a full recompute and a local update.

The reference path in :mod:`af3_structure` walks the pairs in Python, which is
the right shape for an oracle but costs 12 seconds on the 947k-pair support. The
per-prefix arm of issue #139 needs twelve thousand evaluations of it, so this
module carries the same arithmetic in two forms:

* :meth:`StructuralState.full` recomputes everything with sparse products and
  returns the distortion, and
* :meth:`StructuralState.update` recomputes *exactly* the pairs whose readout or
  whose triangle term can have moved when a set of items changes its codes.

The second one is only admissible because the first one exists: a test asserts
that after any number of local updates the cached state equals a full
recomputation to machine precision, which is the alignment the issue demands.

What a change of item set ``U`` touches, exactly:

* the readout ``delta_ij`` moves for every supported pair with an endpoint in
  ``U``;
* the soft degree of a node moves if one of its supported partners is in ``U``;
* the triangle term ``sum_k p_ik p_jk`` moves for a pair ``(i, j)`` if any
  common supported partner ``k`` is in ``U`` -- that is, for pairs within two
  hops of ``U``.

So the affected pairs are the supported pairs incident to ``U`` or to a
supported neighbour of ``U``, and the new triangle terms are read from the rows
of ``P @ P`` that those pairs need. Everything else is untouched by
construction, not by approximation.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from .af3_structure import TAU_FLOOR, huber, readout_delta, readout_probability


class StructuralState:
    """One frozen support, one complete set of hard SIDs, and the distortion."""

    def __init__(self, pairs: np.ndarray, is_edge: np.ndarray, kappa: np.ndarray,
                 n_nodes: int, *, tau: float, intercept: float, scale: float,
                 device: torch.device | None = None):
        self.pairs = np.asarray(pairs, dtype=np.int64)
        self.is_edge = np.asarray(is_edge, dtype=bool)
        self.kappa = np.asarray(kappa, dtype=np.float64)
        self.n_nodes = int(n_nodes)
        self.tau = float(max(tau, TAU_FLOOR))
        self.intercept = float(intercept)
        self.scale = float(scale)
        self.device = device or torch.device("cpu")
        count = len(self.pairs)
        if self.kappa.shape[0] != count:
            raise ValueError("every supported pair needs a target, edges and non-edges alike")
        self.positions: dict[tuple[int, int], int] = {
            (int(i), int(j)): position for position, (i, j) in enumerate(self.pairs)
        }
        incident: list[list[int]] = [[] for _ in range(self.n_nodes)]
        for position, (i, j) in enumerate(self.pairs):
            incident[int(i)].append(position)
            incident[int(j)].append(position)
        self.incident = [np.asarray(sorted(entries), dtype=np.int64) for entries in incident]
        self.edge_mask = self.is_edge
        self.other_mask = ~self.is_edge
        self.delta = np.zeros(count, dtype=np.float64)
        self.probability = np.zeros(count, dtype=np.float64)
        self.triangle = np.zeros(count, dtype=np.float64)
        self.predicted = np.zeros(count, dtype=np.float64)

    # -- the sparse view ---------------------------------------------------
    def _matrix(self) -> sp.csr_matrix:
        """The probability matrix, rebuilt only when its pattern is not cached.

        The support never changes, so the sparsity pattern never changes: the
        matrix is built once and afterwards only its ``data`` vector is
        rewritten. Rebuilding it on every local update cost four seconds for a
        thirty-two item change, which is the whole budget of the per-prefix loop.
        """
        if getattr(self, "_pattern", None) is None:
            rows = np.concatenate([self.pairs[:, 0], self.pairs[:, 1]])
            cols = np.concatenate([self.pairs[:, 1], self.pairs[:, 0]])
            values = np.concatenate([self.probability, self.probability])
            matrix = sp.csr_matrix((values, (rows, cols)), shape=(self.n_nodes, self.n_nodes))
            self._pattern = (matrix.indptr.copy(), matrix.indices.copy())
            self._matrix_cache = matrix
            # Where each supported pair's two directed entries live in ``data``.
            self._data_positions = self._locate_entries(matrix)
        matrix = self._matrix_cache
        values = np.concatenate([self.probability, self.probability])
        matrix.data[self._data_positions] = values
        return matrix

    def _locate_entries(self, matrix: sp.csr_matrix) -> np.ndarray:
        """The ``data`` slot of every directed (pair, direction) entry."""
        indptr, indices = matrix.indptr, matrix.indices
        directed_rows = np.concatenate([self.pairs[:, 0], self.pairs[:, 1]])
        directed_cols = np.concatenate([self.pairs[:, 1], self.pairs[:, 0]])
        positions = np.empty(len(directed_rows), dtype=np.int64)
        for index in range(len(directed_rows)):
            row = int(directed_rows[index])
            column = int(directed_cols[index])
            slots = np.nonzero(indices[indptr[row]:indptr[row + 1]] == column)[0]
            positions[index] = indptr[row] + int(slots[0])
        return positions

    def _readout(self, phi: torch.Tensor, positions: np.ndarray) -> np.ndarray:
        """``1 - cos`` and ``sigmoid(b - delta / tau)`` at the given positions."""
        subset = self.pairs[positions]
        delta = readout_delta(phi, subset)
        probability = readout_probability(delta, self.tau, self.intercept)
        return delta.detach().cpu().numpy(), probability.detach().cpu().numpy()

    def full(self, phi: torch.Tensor) -> dict:
        """Recompute every cached quantity from scratch."""
        every = np.arange(len(self.pairs), dtype=np.int64)
        self.delta, self.probability = self._readout(phi, every)
        matrix = self._matrix()
        rows = self.pairs[:, 0]
        cols = self.pairs[:, 1]
        values = self._triangle_values(matrix, every)
        degree = np.asarray(matrix.sum(axis=1)).reshape(-1)
        self.triangle = values
        term = (2.0 - (degree[rows] - self.probability)
                - (degree[cols] - self.probability) + 3.0 * values)
        self.predicted = self.probability * term
        return self.summary()

    def _triangle_values(self, matrix: sp.csr_matrix,
                         positions: np.ndarray) -> np.ndarray:
        """Read only supported entries of P squared, in bounded row blocks.

        A sparse P does not imply a sparse P squared. Never materialise the
        entire N by N product even in the full-recomputation oracle.
        """
        rows = self.pairs[positions, 0]
        columns = self.pairs[positions, 1]
        needed = np.unique(rows)
        values = np.empty(len(positions), dtype=np.float64)
        for start in range(0, len(needed), 256):
            block_rows = needed[start:start + 256]
            selected = (rows >= block_rows[0]) & (rows <= block_rows[-1])
            product = (matrix[block_rows] @ matrix).tocsr()
            local_rows = np.searchsorted(block_rows, rows[selected])
            values[selected] = np.asarray(product[local_rows, columns[selected]]).reshape(-1)
        return values

    def update(self, phi: torch.Tensor, items: np.ndarray) -> dict:
        """Recompute exactly the pairs a change of these items can have moved."""
        items = np.asarray(items, dtype=np.int64)
        direct = (np.unique(np.concatenate([self.incident[int(item)] for item in items]))
                  if items.size else np.array([], dtype=np.int64))
        if not direct.size:
            return self.summary()
        # Only these probabilities change. Both endpoints of each direct edge
        # can change soft degree, including a neighbour outside U itself.
        neighbours = np.unique(self.pairs[direct])
        inner = np.zeros(self.n_nodes, dtype=bool)
        inner[neighbours] = True
        degree_positions = np.unique(np.concatenate(
            [self.incident[int(node)] for node in neighbours]))
        # Triangle updates have a narrower dependency than degree updates.
        # Direct pairs may change through an endpoint; otherwise both endpoints
        # must neighbour a changed common partner. Do NOT apply this restriction
        # to the degree contribution (the path 0--1--2 is a counterexample).
        triangle_positions = np.unique(np.concatenate([
            direct, degree_positions[inner[self.pairs[degree_positions, 0]]
                                     & inner[self.pairs[degree_positions, 1]]]]))
        self.delta[direct], self.probability[direct] = self._readout(phi, direct)
        matrix = self._matrix()
        self.triangle[triangle_positions] = self._triangle_values(matrix, triangle_positions)
        degree = np.asarray(matrix.sum(axis=1)).reshape(-1)
        positions = degree_positions
        subset_rows, subset_cols = self.pairs[positions].T
        term = (2.0 - (degree[subset_rows] - self.probability[positions])
                - (degree[subset_cols] - self.probability[positions])
                + 3.0 * self.triangle[positions])
        self.predicted[positions] = self.probability[positions] * term
        return self.summary()

    # -- the objective -----------------------------------------------------
    def summary(self) -> dict:
        edge = self.is_edge
        other = self.other_mask
        edge_half = float(huber((self.predicted[edge] - self.kappa[edge]) / self.scale).mean())
        other_half = float(huber(self.predicted[other] / self.scale).mean())
        return {"distortion": 0.5 * edge_half + 0.5 * other_half,
                "edge_half": edge_half, "unobserved_half": other_half,
                "scale": self.scale, "tau": self.tau, "intercept": self.intercept}

    def state(self) -> dict:
        """The per-pair arrays, so a test can compare an update with a full run."""
        return {"delta": self.delta.copy(), "probability": self.probability.copy(),
                "triangle": self.triangle.copy(), "predicted": self.predicted.copy()}

