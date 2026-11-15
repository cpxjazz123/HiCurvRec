"""Reproduce the frozen TIGER SID from the encoder output and the three codebooks.

Nothing downstream of a curvature sweep is trustworthy unless this passes: the c = 0 readout must
give back the published identifier.  Three conventions are tried -- plain Euclidean residual coding,
the same on L2-normalised vectors, and the same starting from the quantized vector file.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
VECTORS = REPO / "results" / "research" / "issue215" / "vectors"
CKPT = REPO / "analysis_sid_recall" / "r2" / "24" / "rqvae_best.pth"
OUT = REPO / "results" / "research" / "issue215" / "validate_quantizer.json"


def codebooks() -> list[np.ndarray]:
    checkpoint = torch.load(CKPT, map_location="cpu")
    state = checkpoint["state_dict"]
    return [state[f"rq.vq_layers.{level}.embed.weight"].numpy().astype(np.float64)
            for level in range(3)]


def quantise(vectors: np.ndarray, books: list[np.ndarray], start: str) -> np.ndarray:
    residual = vectors.copy()
    tokens = np.zeros((vectors.shape[0], 3), dtype=np.int64)
    for level, book in enumerate(books):
        distances = ((residual[:, None, :] - book[None, :, :]) ** 2).sum(axis=2)
        choice = distances.argmin(axis=1)
        tokens[:, level] = choice
        residual = residual - book[choice]
    return tokens


def main() -> int:
    books = codebooks()
    print("codebooks:", [book.shape for book in books], flush=True)
    z = np.load(VECTORS / "TIGER_z.npy").astype(np.float64)
    q = np.load(VECTORS / "TIGER_q.npy").astype(np.float64)
    published = np.load(VECTORS / "TIGER_tokens.npy").astype(np.int64)

    report = {}
    for name, vectors in (("z", z), ("q", q)):
        for start in ("raw", "normalised"):
            data = vectors if start == "raw" else vectors / np.linalg.norm(vectors, axis=1,
                                                                         keepdims=True)
            tokens = quantise(data, books, start)
            match = float(np.mean(np.all(tokens == published, axis=1)))
            per_level = [float(np.mean(tokens[:, level] == published[:, level])) for level in range(3)]
            report[f"{name}/{start}"] = {"all_three": match, "per_level": per_level}
            print(f"{name}/{start}: all three {match:.4f} | per level {per_level}", flush=True)
    OUT.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
