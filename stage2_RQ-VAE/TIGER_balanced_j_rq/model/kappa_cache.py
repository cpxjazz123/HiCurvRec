"""Frozen per-triple curvature cache produced by the issue #269 calibrated fit.

The file is an input, not a result: it is hashed at load, never written by the
trainer, and its rows are bound to the frozen target table row ids by the same
order the sampler uses.  A missing, changed or out-of-domain cache refuses the
model instead of silently falling back to a Euclidean or uniform curvature.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "genrec.issue269.kappa_cache/1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class KappaTable:
    def __init__(self, root):
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("missing frozen curvature cache: " + str(manifest_path))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != SCHEMA:
            raise ValueError("unexpected kappa cache schema")
        self.manifest = manifest
        self.rows = int(manifest["rows"])
        self.scale = float(manifest["scale"])
        if not (self.scale > 0.0) or self.scale != self.scale or self.scale == float("inf"):
            raise ValueError("invalid frozen coordinate scale")
        self.grid = np.asarray(manifest["grid"], dtype=np.float64)
        if self.grid.ndim != 1 or self.grid.size < 2:
            raise ValueError("invalid frozen curvature grid")
        values = self.root / "kappa.npy"
        actual = sha256(values)
        if actual != manifest["outputs"]["kappa_sha256"]:
            raise ValueError("frozen curvature cache changed after the fit")
        self._values = np.load(values, mmap_mode="r", allow_pickle=False)
        if self._values.dtype != np.float32 or self._values.shape != (self.rows,):
            raise ValueError("invalid frozen curvature array")
        step = float(self.grid[1] - self.grid[0])
        if step <= 0 or not np.allclose(np.diff(self.grid), step, rtol=0, atol=1e-9):
            raise ValueError("the frozen curvature grid must be uniform")
        worst = 0.0
        for start in range(0, self.rows, 1_000_000):
            block = np.asarray(self._values[start:start + 1_000_000], dtype=np.float64)
            index = np.rint((block - float(self.grid[0])) / step)
            reconstructed = float(self.grid[0]) + index * step
            error = np.abs(reconstructed - block)
            worst = max(worst, float(np.nanmax(error)) if error.size else 0.0)
            if not np.isfinite(error).all() or float(error.max()) > 1e-5:
                raise ValueError("frozen curvature value outside the recorded grid")
        self.sha256 = actual

    def for_rows(self, row_ids) -> np.ndarray:
        ids = np.asarray(row_ids, dtype=np.int64).reshape(-1)
        if ids.size and (ids.min() < 0 or ids.max() >= self.rows):
            raise ValueError("target row id outside the frozen curvature cache")
        if ids.size != len(set(ids.tolist())):
            raise ValueError("duplicated target row id in a structural sample")
        return np.asarray(self._values[ids], dtype=np.float32)

    def identity(self) -> dict:
        return {
            "schema": self.manifest["schema"],
            "path": str(self.root),
            "rows": self.rows,
            "sha256": self.sha256,
            "grid_size": int(self.grid.size),
            "k_min": float(self.manifest["k_min"]),
            "k_max": float(self.manifest["k_max"]),
            "scale": float(self.manifest["scale"]),
        }
