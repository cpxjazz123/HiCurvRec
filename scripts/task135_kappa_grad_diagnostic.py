#!/usr/bin/env python3
"""Task #135 — κ=0 硬分支导致 θ_m 梯度通路断裂诊断.

Runs two zero-cost checks (per user feedback 2026-07-24):
  Step 1: After first backward(), print θ_m.grad (any non-zero → no bug).
  Step 2: Init θ_m to ±0.5, train 50 epoch, see if it moves from init.

Both arms trained on a tiny batch of Musical_Instruments item embedding
(same data as Task #84/89 — 9922 items × 768 dim T5 embeddings).

Usage:
    python3 scripts/task135_kappa_grad_diagnostic.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO / "HG-Rec"))

from model.hrqvae_free_curv import FreeCurvHRQVAE  # noqa: E402

EMB_PATH = REPO / "HG-Rec/dataset/Instruments/item_emb.parquet"
LOG_PATH = REPO / "verdicts/task135_step1_step2_diagnostic.json"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_item_embeddings() -> torch.Tensor:
    """Load T5 768d item embeddings for Musical_Instruments (9922 × 768)."""
    df = pd.read_parquet(EMB_PATH)
    # Determine embedding column; parquet may be 'emb' / numpy array / list
    if "emb" in df.columns:
        emb = np.stack(df["emb"].values)
    elif "embedding" in df.columns:
        emb = np.stack(df["embedding"].values)
    else:
        # last column
        col = df.columns[-1]
        emb = np.stack(df[col].values)
    print(f"  Loaded embeddings: shape={emb.shape}, dtype={emb.dtype}")
    return torch.tensor(emb, dtype=torch.float32)


def train_one_run(theta_init: list[float], n_epochs: int, lr: float, label: str,
                  M: int = 1, num_emb_list=None) -> dict:
    """Train FreeCurvHRQ-VAE briefly. Return diagnostics."""
    if num_emb_list is None:
        num_emb_list = [64, 128, 256]

    torch.manual_seed(42)
    np.random.seed(42)

    emb = load_item_embeddings()
    N, in_dim = emb.shape
    print(f"  [Run {label}] N={N}, in_dim={in_dim}, θ_init={theta_init}, "
          f"n_epochs={n_epochs}, M={M}")

    # Build model
    model = FreeCurvHRQVAE(
        in_dim=in_dim, num_emb_list=num_emb_list, e_dim=32, M=M,
        kappa_max=2.0, layers=[512, 256, 128], dropout_prob=0.0,
        bn=False, loss_type="mse", quant_loss_weight=1.0, beta=0.25,
        kmeans_init=True, kmeans_iters=10,
        sk_eps=[0.003, 0.003, 0.003], sk_iters=3,
    )

    # Override θ_m init
    with torch.no_grad():
        for vq in model.hrq.vq_layers:
            vq.theta_m.data = torch.tensor(theta_init, dtype=torch.float32)
            vq.initted = False  # force kmeans-init on first forward

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Build dataloader (full-batch to be deterministic)
    ds = TensorDataset(emb)
    dl = DataLoader(ds, batch_size=256, shuffle=True)

    # ─── Step 1 — grad check on first backward ───
    print(f"\n  [Step 1 — grad check]")
    # One forward pass on first batch to trigger codebook init
    first_batch = next(iter(dl))[0]
    z = model.encoder(first_batch)
    out, rq_loss, _ = model.hrq(z, use_sk=False)
    out = model.decoder(out)
    loss = F.mse_loss(out, first_batch, reduction="mean") + model.quant_loss_weight * rq_loss
    opt.zero_grad()
    loss.backward()

    grad_norm_per_layer = []
    theta_grad_per_layer = []
    for i, vq in enumerate(model.hrq.vq_layers):
        if vq.theta_m.grad is None:
            grad_norm_per_layer.append(None)
            theta_grad_per_layer.append([None] * vq.M)
            print(f"    Layer {i}: θ_m.grad = None ❌")
        else:
            g = vq.theta_m.grad
            gn = g.norm().item()
            grad_norm_per_layer.append(gn)
            theta_grad_per_layer.append(g.detach().cpu().tolist())
            print(f"    Layer {i}: θ_m.grad norm = {gn:.6e}, "
                  f"per-component = {[f'{x:+.3e}' for x in g.detach().cpu().tolist()]}")

    step1_diagnosis = "OK_GRAD_NONZERO" if all(
        g is not None and g > 1e-10 for g in grad_norm_per_layer
    ) else "BUG_GRAD_ZERO_OR_NONE"
    print(f"    → Step 1 verdict: {step1_diagnosis}")

    # ─── Step 2 — train 50 epoch, track θ_m trajectory ───
    print(f"\n  [Step 2 — {n_epochs} epoch trajectory]")
    theta_history = []  # per-epoch list-of-list
    for ep in range(n_epochs):
        for batch in dl:
            x = batch[0]
            out, rq_loss, _ = model(x, use_sk=False)
            loss = F.mse_loss(out, x, reduction="mean") + model.quant_loss_weight * rq_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep == 0 or (ep + 1) % 5 == 0 or ep == n_epochs - 1:
            snapshot = [vq.theta_m.detach().cpu().tolist() for vq in model.hrq.vq_layers]
            kappa_snapshot = [vq.kappa_m().detach().cpu().tolist() for vq in model.hrq.vq_layers]
            print(f"    Epoch {ep+1:3d}/{n_epochs}: "
                  f"θ_m[0]={snapshot[0]}, κ[0]={kappa_snapshot[0]}")
            theta_history.append({"epoch": ep + 1, "theta": snapshot, "kappa": kappa_snapshot})

    # Final snapshot
    final_theta = [vq.theta_m.detach().cpu().tolist() for vq in model.hrq.vq_layers]
    final_kappa = [vq.kappa_m().detach().cpu().tolist() for vq in model.hrq.vq_layers]
    max_move = max(
        max(abs(final_theta[i][m] - theta_init[m]) for m in range(M))
        for i in range(len(final_theta))
    )
    step2_diagnosis = (
        "OK_THETA_MOVED" if max_move > 1e-4
        else "BUG_THETA_FROZEN_AT_INIT"
    )
    print(f"    Final θ_m: {final_theta}")
    print(f"    Final κ_m: {final_kappa}")
    print(f"    Max |θ_final − θ_init| = {max_move:.6e}")
    print(f"    → Step 2 verdict: {step2_diagnosis}")

    return {
        "label": label,
        "theta_init": theta_init,
        "lr": lr,
        "n_epochs": n_epochs,
        "M": M,
        "step1_grad_norm_per_layer": grad_norm_per_layer,
        "step1_theta_grad_per_layer": theta_grad_per_layer,
        "step1_diagnosis": step1_diagnosis,
        "step2_theta_history": theta_history,
        "step2_final_theta": final_theta,
        "step2_final_kappa": final_kappa,
        "step2_max_theta_move": max_move,
        "step2_diagnosis": step2_diagnosis,
    }


def main() -> int:
    print("=" * 70)
    print("Task #135 — κ=0 硬分支 θ_m 梯度通路诊断")
    print("=" * 70)

    results = {}

    # Run A: θ_init = [0.0] (baseline — should reproduce Task #89 finding)
    print("\n[Run A — θ_init=[0.0]]")
    results["A_init_0"] = train_one_run(
        theta_init=[0.0], n_epochs=50, lr=1e-3, label="A_init_0", M=1
    )

    # Run B: θ_init = [+0.5] (off-zero, positive — should move if no bug)
    print("\n[Run B — θ_init=[+0.5]]")
    results["B_init_p0.5"] = train_one_run(
        theta_init=[+0.5], n_epochs=50, lr=1e-3, label="B_init_p0.5", M=1
    )

    # Run C: θ_init = [-0.5] (off-zero, negative — should move if no bug)
    print("\n[Run C — θ_init=[-0.5]]")
    results["C_init_n0.5"] = train_one_run(
        theta_init=[-0.5], n_epochs=50, lr=1e-3, label="C_init_n0.5", M=1
    )

    # Run D: M=2 θ_init=[+0.5, -0.5] (multi-component sanity)
    print("\n[Run D — θ_init=[+0.5, -0.5], M=2]")
    results["D_M2_init_p0.5_n0.5"] = train_one_run(
        theta_init=[+0.5, -0.5], n_epochs=50, lr=1e-3,
        label="D_M2_init_p0.5_n0.5", M=2
    )

    # ─── Final verdict ───
    print("\n" + "=" * 70)
    print("诊断总结")
    print("=" * 70)
    all_step1_frozen = all(r["step1_diagnosis"] != "OK_GRAD_NONZERO" for r in results.values())
    all_step2_frozen = all(r["step2_diagnosis"] != "OK_THETA_MOVED" for r in results.values())
    print(f"  Step 1 (backward grad nonzero): "
          f"{'❌ ALL FROZEN (bug confirmed)' if all_step1_frozen else '✅ some nonzero'}")
    print(f"  Step 2 (θ_m moves within 50 epoch): "
          f"{'❌ ALL FROZEN (bug confirmed)' if all_step2_frozen else '✅ some moved'}")
    if all_step1_frozen and all_step2_frozen:
        print("\n  >>> BUG CONFIRMED: hard-branch at κ=0切断梯度信号")
        print("  >>> Task #89 '18/18 κ_m=0' 结论 = 梯度断, 非数据本质")
        overall = "BUG_CONFIRMED"
    elif all_step1_frozen:
        print("\n  >>> 矛盾信号: grad=0 但 θ 仍动 → 需要查 pathway")
        overall = "MIXED_SIGNAL"
    else:
        print("\n  >>> 梯度非零, Task #89 结论基本安全, 仅阈值偏低")
        overall = "NO_BUG"
    print(f"  Overall: {overall}")

    # Save
    with open(LOG_PATH, "w") as f:
        json.dump({"results": results, "overall_diagnosis": overall}, f, indent=2,
                  ensure_ascii=False)
    print(f"\n  Saved: {LOG_PATH}")
    return 0 if overall != "BUG_CONFIRMED" else 2  # exit 2 if bug confirmed


if __name__ == "__main__":
    sys.exit(main())
