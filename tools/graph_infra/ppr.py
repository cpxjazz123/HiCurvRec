"""Exact personalised PageRank by sparse factorisation (issue #246 C21).

``x = α e_s + (1−α)(Pᵀ x + d(x) e_s)`` with ``P`` row-normalised and ``d`` the dangling mass,
so a dangling row returns its mass to the source instead of silently losing probability.
Solving ``(I − (1−α)Pᵀ) x0 = e_s`` once per source gives the answer in closed form:

    β  = α (dᵀ x0) / (1 − (1−α) dᵀ x0)
    x  = (α + (1−α) β) x0

which is what ``solve`` returns; the residual of the original equation is reported.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _transition(forward: sp.csr_matrix):
    row_sums = np.asarray(forward.sum(axis=1)).ravel().astype(np.float64)
    inverse = np.where(row_sums > 0, 1.0 / np.maximum(row_sums, 1e-300), 0.0)
    matrix = sp.diags(inverse) @ forward.astype(np.float64)
    dangling = (row_sums == 0).astype(np.float64)
    return matrix.tocsr(), dangling


def solve(forward: sp.csr_matrix, sources, alpha: float = 0.15, chunk: int = 256) -> dict:
    """Exact PPR rows for every source; residual and mass are reported, nothing is truncated."""
    n = forward.shape[0]
    matrix, dangling = _transition(forward)
    left = (sp.identity(n, format="csr") - (1.0 - alpha) * matrix.T).tocsc()
    factor = spla.splu(left)
    sources = np.asarray(sources, dtype=np.int64)
    rows = np.zeros((sources.size, n), dtype=np.float64)
    dangling_mass = np.zeros(sources.size, dtype=np.float64)
    residuals = np.zeros(sources.size, dtype=np.float64)
    for start in range(0, sources.size, chunk):
        stop = min(start + chunk, sources.size)
        block = sources[start:stop]
        rhs = np.zeros((n, stop - start), dtype=np.float64)
        rhs[block, np.arange(stop - start)] = 1.0
        solved = factor.solve(rhs)
        mass = dangling @ solved
        beta = alpha * mass / np.maximum(1.0 - (1.0 - alpha) * mass, 1e-300)
        rows[start:stop] = (alpha + (1.0 - alpha) * beta)[:, None] * solved.T
        dangling_mass[start:stop] = mass
    # residual of the original fixed point x = a e_s + (1-a)(P^T x + (d^T x) e_s)
    probe = min(8, sources.size)
    residuals = np.zeros(probe, dtype=np.float64)
    for index in range(probe):
        source = int(sources[index])
        vector = rows[index]
        lhs = vector - (1.0 - alpha) * (matrix.T @ vector) - (1.0 - alpha) * float(dangling @ vector) * np.eye(1, n, source).ravel()
        rhs = alpha * np.eye(1, n, source).ravel()
        residuals[index] = float(np.abs(lhs - rhs).max())
    return {"sources": sources, "rows": rows, "dangling_mass": dangling_mass,
            "residual_max": float(np.max(np.abs(residuals[:probe]))) if probe else 0.0,
            "alpha": float(alpha), "unit": "probability mass (not a distance)"}


class Solver:
    """One factorisation, reused for every bank: ``solve_pairs`` for any pair array."""

    def __init__(self, forward: sp.csr_matrix, alpha: float = 0.15, method: str = "auto",
                 tolerance: float = 1e-6, max_iterations: int = 150, batch: int = 128):
        """``method='lu'`` is exact but factorises (pathological on this graph);
        ``method='power'`` is the matrix-free Neumann series with a stated tolerance.
        ``'auto'`` uses LU for small graphs and the series for large ones.
        """
        self.forward = forward
        self.alpha = float(alpha)
        self.n = forward.shape[0]
        self.matrix, self.dangling = _transition(forward)
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        self.batch = int(batch)
        if method == "auto":
            method = "lu" if self.n <= 5000 else "power"
        self.method = method
        self.factor = None
        if method == "lu":
            left = (sp.identity(self.n, format="csr") - (1.0 - alpha) * self.matrix.T).tocsc()
            self.factor = spla.splu(left)

    def solve_pairs(self, pairs, chunk: int = 256) -> dict:
        if self.method == "power":
            return self._solve_power(pairs)
        pairs = np.asarray(pairs, dtype=np.int64)
        by_source = {}
        for index, (source, target) in enumerate(pairs.tolist()):
            by_source.setdefault(int(source), []).append((index, int(target)))
        sources = np.array(sorted(by_source), dtype=np.int64)
        values = np.zeros(pairs.shape[0], dtype=np.float64)
        residuals = []
        for start in range(0, sources.size, chunk):
            block = sources[start:start + chunk]
            rhs = np.zeros((self.n, block.size), dtype=np.float64)
            rhs[block, np.arange(block.size)] = 1.0
            solved = self.factor.solve(rhs)
            mass = self.dangling @ solved
            beta = self.alpha * mass / np.maximum(1.0 - (1.0 - self.alpha) * mass, 1e-300)
            scaled = (self.alpha + (1.0 - self.alpha) * beta)[:, None] * solved.T
            for row, source in enumerate(block.tolist()):
                for index, target in by_source[source]:
                    values[index] = scaled[row, target]
            if len(residuals) < 8:
                for row, source in enumerate(block.tolist()[:max(0, 8 - len(residuals))]):
                    vector = scaled[row]
                    lhs = (vector - (1.0 - self.alpha) * (self.matrix.T @ vector)
                           - (1.0 - self.alpha) * float(self.dangling @ vector)
                           * np.eye(1, self.n, source).ravel())
                    residuals.append(float(np.abs(lhs - self.alpha * np.eye(1, self.n, source).ravel()).max()))
        return {"values": values, "sources": int(sources.size),
                "residual_max": float(np.max(residuals)) if residuals else 0.0,
                "alpha": self.alpha, "unit": "probability mass (not a distance)",
                "coverage": float(sources.size / max(1, len(set(pairs[:, 0].tolist()))))}


    def _solve_power(self, pairs) -> dict:
        """Neumann series, matrix free: no fill-in, a stated tolerance and a reported residual."""
        pairs = np.asarray(pairs, dtype=np.int64)
        by_source = {}
        for index, (source, target) in enumerate(pairs.tolist()):
            by_source.setdefault(int(source), []).append((index, int(target)))
        sources = np.array(sorted(by_source), dtype=np.int64)
        values = np.zeros(pairs.shape[0], dtype=np.float64)
        residuals = []
        iterations_used = 0
        for start in range(0, sources.size, self.batch):
            block = sources[start:start + self.batch]
            vector = np.zeros((self.n, block.size), dtype=np.float64)
            vector[block, np.arange(block.size)] = self.alpha
            residual = float("inf")
            for iteration in range(1, self.max_iterations + 1):
                previous = vector
                updated = (1.0 - self.alpha) * (self.matrix.T @ vector)
                mass = self.dangling @ vector
                updated[block, np.arange(block.size)] += self.alpha + (1.0 - self.alpha) * mass
                vector = updated
                iterations_used = max(iterations_used, iteration)
                # the full max-norm residual costs as much as the matvec, so it is measured
                # every few rounds; the source entries give a cheap progress proxy in between
                if iteration % 5 == 0 or iteration == self.max_iterations:
                    residual = float(np.abs(updated - previous).max())
                    if residual < self.tolerance:
                        break
            for row, source in enumerate(block.tolist()):
                for index, target in by_source[source]:
                    values[index] = vector[target, row]
            residuals.append(residual)
        return {"values": values, "sources": int(sources.size),
                "residual_max": float(max(residuals)) if residuals else 0.0,
                "alpha": self.alpha, "unit": "probability mass (not a distance)",
                "coverage": float(sources.size / max(1, len(set(pairs[:, 0].tolist())))),
                "method": f"neumann series (matrix free), tolerance {self.tolerance}",
                "iterations_max": iterations_used}


def solve_selected(forward: sp.csr_matrix, pairs, alpha: float = 0.15, chunk: int = 256) -> dict:
    """PPR for exactly the requested pairs, one value per row, without materialising the rows."""
    n = forward.shape[0]
    pairs = np.asarray(pairs, dtype=np.int64)
    matrix, dangling = _transition(forward)
    left = (sp.identity(n, format="csr") - (1.0 - alpha) * matrix.T).tocsc()
    factor = spla.splu(left)
    by_source = {}
    for index, (source, target) in enumerate(pairs.tolist()):
        by_source.setdefault(int(source), []).append((index, int(target)))
    sources = np.array(sorted(by_source), dtype=np.int64)
    values = np.zeros(pairs.shape[0], dtype=np.float64)
    residuals = []
    for start in range(0, sources.size, chunk):
        block = sources[start:start + chunk]
        rhs = np.zeros((n, block.size), dtype=np.float64)
        rhs[block, np.arange(block.size)] = 1.0
        solved = factor.solve(rhs)
        mass = dangling @ solved
        beta = alpha * mass / np.maximum(1.0 - (1.0 - alpha) * mass, 1e-300)
        scaled = (alpha + (1.0 - alpha) * beta)[:, None] * solved.T
        for row, source in enumerate(block.tolist()):
            for index, target in by_source[source]:
                values[index] = scaled[row, target]
        if len(residuals) < 8:
            for row, source in enumerate(block.tolist()[:max(0, 8 - len(residuals))]):
                vector = scaled[row]
                lhs = (vector - (1.0 - alpha) * (matrix.T @ vector)
                       - (1.0 - alpha) * float(dangling @ vector) * np.eye(1, n, source).ravel())
                residuals.append(float(np.abs(lhs - alpha * np.eye(1, n, source).ravel()).max()))
    return {"values": values, "sources": int(sources.size),
            "residual_max": float(np.max(residuals)) if residuals else 0.0,
            "alpha": float(alpha), "unit": "probability mass (not a distance)",
            "coverage": float(sources.size / max(1, len(set(pairs[:, 0].tolist()))))}
