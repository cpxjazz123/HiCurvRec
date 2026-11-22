"""Does whitening restore between-item variance that a 256-codeword level can use?

Proxy for "would a whitened structure embedding fix TIGER_g237_struct_rq": fit PCA-32 +
whitening on the struct embedding, run k-means with 256 centres on the whitened vectors and
on the raw PCA-32 vectors, then compare the level-1 entropy and the largest cluster share --
the two numbers that collapsed to 0.232 bit / 98.3% in the trained arm.
"""
from __future__ import annotations

import json

import numpy as np

STRUCT = "results/research/issue237/enhanced_768.npy"
SEED = 42
DIM = 32
CENTRES = 256
ITERS = 30


def kmeans(X: np.ndarray, centres: int, seed: int, iters: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), centres, replace=False)
    C = X[idx].copy()
    labels = np.zeros(len(X), dtype=np.int64)
    for _ in range(iters):
        d = (X * X).sum(1)[:, None] - 2 * X @ C.T + (C * C).sum(1)[None, :]
        labels = d.argmin(1)
        for k in range(centres):
            mask = labels == k
            if mask.any():
                C[k] = X[mask].mean(0)
    return labels


def stats(labels: np.ndarray, tag: str) -> dict:
    counts = np.bincount(labels, minlength=labels.max() + 1).astype(np.float64)
    p = counts[counts > 0] / counts.sum()
    entropy = float(-(p * np.log2(p)).sum())
    out = {"arm": tag, "used": int((counts > 0).sum()), "entropy_bits": round(entropy, 4),
           "max_share": round(float(p.max()), 4)}
    print(json.dumps(out))
    return out


def main() -> int:
    X = np.load(STRUCT).astype(np.float32)
    X = X - X.mean(0, keepdims=True)
    # PCA-32 via randomised range finder on a subsample for speed, then project all rows.
    rng = np.random.default_rng(SEED)
    sample = X[rng.choice(len(X), min(4000, len(X)), replace=False)]
    cov = (sample.T @ sample) / len(sample)
    w, V = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1][:DIM]
    w, V = w[order], V[:, order]
    Z = X @ V
    white = Z / np.sqrt(np.clip(w, 1e-8, None))[None, :]
    print(json.dumps({"pca": "32", "explained": float(w.sum() / np.trace(cov)),
                      "pc1_share": float(w[0] / w.sum())}))
    stats(kmeans(Z, CENTRES, SEED, ITERS), "pca32 raw")
    stats(kmeans(white, CENTRES, SEED, ITERS), "pca32 whitened")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
