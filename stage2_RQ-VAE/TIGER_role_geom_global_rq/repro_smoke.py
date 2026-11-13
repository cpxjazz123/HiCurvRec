"""Reproduce the smoke failure on the real path and show where the numbers go wrong.

    python3 stage2_RQ-VAE/TIGER_role_geom_rq/repro_smoke.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

TREE = Path(__file__).resolve().parent
sys.path.insert(0, str(TREE))
sys.path.insert(0, str(TREE.parents[1]))

import torch

import train_rqvae as trainer
from model import mixed_rq

trainer.EPOCHS = 1

original = mixed_rq.MixedVQLevel.distances


def traced(self, residual_e, residual_h, alpha, weights=None, scales=(1.0, 1.0, 1.0)):
    cost = original(self, residual_e, residual_h, alpha, weights=weights, scales=scales)
    finite = torch.isfinite(cost)
    print(f"[trace] shapes: residual_e {tuple(residual_e.shape)} codes {tuple(self.get_code_embs().shape)} "
          f"alpha {tuple(alpha.shape)} weights {None if weights is None else tuple(weights.shape)}", flush=True)
    print(f"[trace] cost shape {tuple(cost.shape)} | residual_h shape {tuple(residual_h.shape)} "
          f"| residual_e finite {bool(torch.isfinite(residual_e).all())} | alpha values {alpha.detach().flatten()[:3].tolist()}", flush=True)
    if finite.numel():
        print(f"[trace] level distances: finite {bool(finite.all())} | nan {int((~finite).sum())} "
              f"| min {float(cost[finite].min()):.6g} max {float(cost[finite].max()):.6g}", flush=True)
    else:
        print("[trace] EMPTY COST", flush=True)
    return cost


mixed_rq.MixedVQLevel.distances = traced

if __name__ == "__main__":
    trainer.main()
