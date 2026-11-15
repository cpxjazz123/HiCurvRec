"""Why does curvature not move the level-1 assignment? Measure the lever instead of asserting it.

The sweep changed almost nothing across c = 1..256 (0.03% of items at c = 256).  A curvature change
can only reorder candidates if the conformal factor (1 - c||p||^2)^-2 differs between them, so this
script measures:

    * the norm distribution of the frozen level-1 codebook after the tanh map into the ball
    * how much the conformal factor varies across the 256 codes at each curvature
    * the level-1 decision margin: second-best minus best distance, relative to the best
    * an explicit ball-scale sweep, because the lever only exists near the boundary

    python3 results/research/issue215/curvature_lever_diagnostic.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
VECTORS = HERE / "vectors"
CKPT = REPO / "analysis_sid_recall" / "r2" / "24" / "rqvae_best.pth"
OUT = HERE / "curvature_lever_diagnostic.json"
CURVATURES = (1.0, 4.0, 16.0, 64.0, 256.0)
SCALES = (0.05, 0.3, 0.6, 0.9)


def to_ball(vectors: np.ndarray, scale: float) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return scale * np.tanh(norms / 2.0) * vectors / np.maximum(norms, 1e-12)


def poincare(points: np.ndarray, codes: np.ndarray, curvature: float) -> np.ndarray:
    squared_points = (points ** 2).sum(axis=1)[:, None]
    squared_codes = (codes ** 2).sum(axis=1)[None, :]
    euclid = np.maximum(squared_points + squared_codes - 2 * points @ codes.T, 0.0)
    denominator = (1 - curvature * squared_points) * (1 - curvature * squared_codes)
    argument = 1 + 2 * curvature * euclid / np.maximum(denominator, 1e-12)
    return np.arccosh(np.maximum(argument, 1.0)) / np.sqrt(curvature)


def main() -> int:
    state = torch.load(CKPT, map_location="cpu")["state_dict"]
    book = state["rq.vq_layers.0.embed.weight"].numpy().astype(np.float64)
    z = np.load(VECTORS / "TIGER_z.npy").astype(np.float64)
    published = np.load(VECTORS / "TIGER_tokens.npy").astype(np.int64)

    baseline = ((z[:, None, :] - book[None, :, :]) ** 2).sum(axis=2)
    order = np.argsort(baseline, axis=1)
    best = baseline[np.arange(len(z)), order[:, 0]]
    second = baseline[np.arange(len(z)), order[:, 1]]
    margin = (second - best) / np.maximum(best, 1e-12)

    report = {"codebook_norms": {"min": float(np.linalg.norm(book, axis=1).min()),
                                 "max": float(np.linalg.norm(book, axis=1).max()),
                                 "mean": float(np.linalg.norm(book, axis=1).mean()),
                                 "cv": float(np.linalg.norm(book, axis=1).std()
                                             / np.linalg.norm(book, axis=1).mean())},
              "vector_norms": {"mean": float(np.linalg.norm(z, axis=1).mean()),
                               "p95": float(np.percentile(np.linalg.norm(z, axis=1), 95))},
              "margin": {"median": float(np.median(margin)),
                         "p10": float(np.percentile(margin, 10)),
                         "share_below_5pct": float(np.mean(margin < 0.05))},
              "sweeps": {}}
    norms_block = report["codebook_norms"]
    print(f"L1 codebook norms: min {norms_block['min']:.3f} max {norms_block['max']:.3f} "
          f"mean {norms_block['mean']:.3f} cv {norms_block['cv']:.3f}")
    vector_block = report["vector_norms"]
    print(f"z norms: mean {vector_block['mean']:.3f} p95 {vector_block['p95']:.3f}")
    margin_block = report["margin"]
    print(f"level-1 margin (second-best vs best): median {margin_block['median']:.3f}, "
          f"p10 {margin_block['p10']:.3f}, share below 5% {margin_block['share_below_5pct']:.3f}")

    for scale in SCALES:
        points = to_ball(z, scale)
        codes = to_ball(book, scale)
        for curvature in CURVATURES:
            if curvature * float((codes ** 2).sum(axis=1).max()) >= 1.0:
                continue
            distances = poincare(points, codes, curvature)
            choice = distances.argmin(axis=1)
            changed = float(np.mean(choice != published[:, 0]))
            # conformal factor spread across the candidate codes, at this curvature
            factor = 1.0 / (1.0 - curvature * (codes ** 2).sum(axis=1))
            spread = float(factor.std() / factor.mean())
            report["sweeps"][f"scale={scale}/c={curvature:g}"] = {
                "changed_share": changed, "factor_cv": spread,
                "max_curvature_times_norm2": float(curvature * (codes ** 2).sum(axis=1).max())}
            print(f"scale {scale:<5} c {curvature:<6g} level-1 changed {changed:.4f}  "
                  f"conformal-factor cv {spread:.4f}", flush=True)
    OUT.write_text(json.dumps(report, indent=1))
    print(f"written {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
