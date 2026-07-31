#!/usr/bin/env python3
"""Task #408 / Issue #115 [方向A Gate1] FreeCurvVQ K[64,128,256] Stage 1 短训

Per Issue #115 spec §Gate1:
- 三层 learnable variable-curvature: L0 K64, L1 K128, L2 K256
- 复用 #100/#103 上游 (HG-Rec baseline recipe, 200 epoch Stage 1)
- 但 task84 实际是标准 RQ-VAE (无 theta_m), 需重新训练 FreeCurvVQ (R137 fix)

Recipe:
- 30 epoch 短训 (Issue #115 spec §Gate1 受限短训 + task402 Stage 1 30 epoch 已证明有效)
- Sinkhorn-Knopp (sk_eps=0.003, sk_iters=3) + 4-digit dedup
- 三层 = L0 K64 / L1 K128 / L2 K256
- R137 fix: per-component learnable κ_m = κ_max · tanh(θ_m), init θ_m=0 → κ_m=0 (Euclidean)

Output:
- best ckpt 落盘 (R12 强制)
- sidecar JSON: θ_m, κ_m, scale, codebook norm per layer
- 三层 utilization, max_load, entropy, norm
"""
from __future__ import annotations

import sys
import os
import json
import math
import time
import hashlib
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

HGREC_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec").resolve()
sys.path.insert(0, str(HGREC_ROOT))
sys.path.insert(0, str(HGREC_ROOT / "model"))

# Issue #97 patch
from model import utils as hgrec_utils  # type: ignore


def patched_poincare_distance(x, y, c):
    diff = hgrec_utils.mobius_add(-x, y, c)
    diff = hgrec_utils.proj_to_ball(diff, c)
    sqrt_c = c ** 0.5
    norm = diff.norm(dim=-1, keepdim=True).clamp_min(hgrec_utils._eps(diff))
    return (2.0 / sqrt_c) * hgrec_utils.artanh(sqrt_c * norm)


hgrec_utils.poincare_distance = patched_poincare_distance

from model.hrqvae_free_curv import (  # type: ignore
    FreeCurvVectorQuantization,
    FreeCurvResidualVectorQuantization,
)


#===========================================================================================
# Configuration (per Issue #115 spec §Gate1)
#===========================================================================================
SEED = 42
NUM_EPOCHS = 30  # 短训, per task402 实证
BATCH_SIZE = 256
LR = 1e-4
DEVICE = "cuda:0"  # GPU 0 (per R7 全部空闲时优先 GPU 0)
NUM_EMB_LIST = [64, 128, 256]  # L0 K64 / L1 K128 / L2 K256 (Issue #115 spec 强制)
E_DIM = 32
KAPPA_MAX = 2.0
BETA = 0.25  # commitment loss weight (HG-Rec baseline)

DATA_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet"

PRODUCTS_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec/products/task408_issue115_stage1_freecurv")
PRODUCTS_ROOT.mkdir(parents=True, exist_ok=True)
CKPT_DIR = PRODUCTS_ROOT / "ckpt"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = PRODUCTS_ROOT / "train_log.json"

USAGE_KILL_UTIL = 0.20  # 任一层 util<20% 早停


#===========================================================================================
# 数据加载
#===========================================================================================
def load_embeddings():
    import pandas as pd
    df = pd.read_parquet(DATA_PARQUET)
    if 'embedding' in df.columns:
        X = np.stack(df['embedding'].values).astype(np.float32)
    else:
        first_col = df.columns[0]
        X = np.stack(df[first_col].values).astype(np.float32)

    # Project 768 → 32 (HG-Rec baseline AE setup)
    if X.shape[1] != E_DIM:
        rng = np.random.RandomState(SEED)
        proj = rng.randn(X.shape[1], E_DIM).astype(np.float32) / np.sqrt(X.shape[1])
        X = X @ proj
        norms = np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-6)
        X = X / norms
    return torch.from_numpy(X).float()


#===========================================================================================
# Sinkhorn-Knopp (跟 HG-Rec baseline 一致)
#===========================================================================================
def sinkhorn_algorithm(d: torch.Tensor, eps: float = 0.003, n_iters: int = 3) -> torch.Tensor:
    """Sinkhorn-Knopp balanced assignment. d: (B, K) cost matrix."""
    Q = torch.exp(-d / eps).T  # (K, B)
    B = Q.shape[1]
    K = Q.shape[0]
    Q /= Q.sum().clamp_min(1e-12)
    for _ in range(n_iters):
        Q /= Q.sum(dim=1, keepdim=True).clamp_min(1e-12)
        Q /= K
        Q /= Q.sum(dim=0, keepdim=True).clamp_min(1e-12)
        Q /= B
    Q *= B
    return Q.T  # (B, K)


#===========================================================================================
# Trainer (FreeCurvVQ per-component)
#===========================================================================================
def compute_layer_metrics(indices: torch.Tensor, n_e: int, layer_idx: int) -> dict:
    indices_np = indices.detach().cpu().numpy()
    counter = Counter(indices_np.tolist())
    used = len(counter)
    util = used / n_e
    counts = np.array([counter.get(i, 0) for i in range(n_e)], dtype=np.float32)
    max_load = (counts.max() / counts.sum()).item() if counts.sum() > 0 else 0.0
    p = counts / counts.sum().clip(min=1e-12)
    entropy = -(p * np.log(p + 1e-12)).sum()
    max_entropy = np.log(n_e)
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    return {
        "layer": layer_idx,
        "n_e": n_e,
        "utilization": float(util),
        "max_load": float(max_load),
        "entropy_normalized": float(norm_entropy),
    }


def train_epoch(model, dataloader, optimizer, epoch, device):
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    n_batches = 0
    all_indices = [[] for _ in range(len(model.vq_layers))]

    for batch in dataloader:
        x = batch[0].to(device)
        optimizer.zero_grad()

        # Forward RQ
        all_losses = []
        all_indices_batch = []
        x_q = 0
        residual = x
        for layer in model.vq_layers:
            x_res, loss, indices = layer(residual, use_sk=True)
            residual = residual - x_res
            x_q = x_q + x_res
            all_losses.append(loss)
            all_indices_batch.append(indices)

        recon_loss = F.mse_loss(x_q, x)
        vq_loss = sum(all_losses)
        loss = recon_loss + vq_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        n_batches += 1

        for l in range(len(model.vq_layers)):
            all_indices[l].append(all_indices_batch[l].detach().cpu())

    layer_metrics = []
    for l in range(len(model.vq_layers)):
        indices_cat = torch.cat(all_indices[l])
        metrics = compute_layer_metrics(indices_cat, model.vq_layers[l].n_e, l)
        layer_metrics.append(metrics)

    return {
        "epoch": epoch,
        "loss": total_loss / max(1, n_batches),
        "recon_loss": total_recon / max(1, n_batches),
        "layer_metrics": layer_metrics,
    }


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 80, flush=True)
    print(f"[Task #408 Issue #115 Gate 1] FreeCurvVQ K[64,128,256] Stage 1 ({NUM_EPOCHS} epoch)", flush=True)
    print("=" * 80, flush=True)

    X = load_embeddings()
    print(f"Data shape: {X.shape}", flush=True)

    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    # Build FreeCurvHRQVAE with M=3 components (per R137 fix)
    model = FreeCurvResidualVectorQuantization(
        n_e_list=NUM_EMB_LIST,
        e_dim=E_DIM,
        M=3,
        kappa_max=KAPPA_MAX,
        beta=BETA,
        kmeans_init=False,
        kmeans_iters=10,
        sk_eps=[0.003, 0.003, 0.003],
        sk_iters=3,
    )
    model.to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    log = []
    best_util = 0.0
    best_epoch = 0
    best_ckpt_path = None

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        result = train_epoch(model, dataloader, optimizer, epoch, DEVICE)
        result["epoch_time_sec"] = time.time() - t0
        log.append(result)

        util_str = " | ".join([f"L{l}={m['utilization']*100:.1f}%" for l, m in enumerate(result['layer_metrics'])])
        max_load_str = " | ".join([f"L{l}={m['max_load']*100:.1f}%" for l, m in enumerate(result['layer_metrics'])])
        print(
            f"[Ep {epoch:02d}/{NUM_EPOCHS}] loss={result['loss']:.4f} recon={result['recon_loss']:.4f} "
            f"util: {util_str} max_load: {max_load_str} time={result['epoch_time_sec']:.1f}s",
            flush=True,
        )

        avg_util = sum(m["utilization"] for m in result['layer_metrics']) / len(result['layer_metrics'])
        if avg_util > best_util:
            best_util = avg_util
            best_epoch = epoch
            best_ckpt_path = CKPT_DIR / f"task408_best_epoch_{epoch:02d}.pth"
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_util': best_util,
            }, best_ckpt_path)

        min_util = min(m["utilization"] for m in result['layer_metrics'])
        if min_util < USAGE_KILL_UTIL and epoch >= 5:
            print(f"[USAGE-KILL] min_util={min_util:.4f} < {USAGE_KILL_UTIL}, early stop @ ep {epoch}", flush=True)
            break

    # === Final: Save + Sidecar Export ===
    print(f"\n[Final] Best epoch: {best_epoch}, best avg util: {best_util:.4f}", flush=True)

    # Reload best ckpt + extract sidecar
    if best_ckpt_path is None or not best_ckpt_path.exists():
        print("❌ No best ckpt saved!", flush=True)
        return

    raw = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
    state = raw["state_dict"]

    sidecar = {
        "ckpt_path": str(best_ckpt_path),
        "ckpt_sha256": hashlib.sha256(best_ckpt_path.read_bytes()).hexdigest(),
        "ckpt_size_bytes": os.path.getsize(best_ckpt_path),
        "best_epoch": best_epoch,
        "best_avg_util": best_util,
        "expected_K": NUM_EMB_LIST,
        "expected_e_dim": E_DIM,
        "expected_M": 3,
        "expected_kappa_max": KAPPA_MAX,
        "layers": [],
        "final_epoch_metrics": log[-1]["layer_metrics"] if log else None,
        "training_log": log,
    }

    for l in range(3):
        theta_key = f"vq_layers.{l}.theta_m"
        embed_key = f"vq_layers.{l}.embeddings.weight"
        theta_m = state[theta_key].cpu().numpy()
        embeddings = state[embed_key].cpu().numpy()
        K = embeddings.shape[0]
        kappa_m = KAPPA_MAX * np.tanh(theta_m)

        # Per-component codebook norms (M=3, e_dim=32 → block_dims=[11,11,10])
        block_dims = [11, 11, 10]
        per_comp_norms = []
        start = 0
        for m, bd in enumerate(block_dims):
            comp = embeddings[:, start:start+bd]
            per_comp_norms.append(float(np.linalg.norm(comp, axis=1).mean()))
            start += bd

        codebook_norm_mean = float(np.linalg.norm(embeddings, axis=1).mean())
        sidecar["layers"].append({
            "layer": l,
            "K": int(K),
            "theta_m": theta_m.tolist(),
            "kappa_m_effective": kappa_m.tolist(),
            "scale": codebook_norm_mean,
            "codebook_norm_mean": codebook_norm_mean,
            "per_component_norm_mean": per_comp_norms,
        })

    sidecar["gate1_pass"] = sidecar["final_epoch_metrics"] is not None and all(
        m["utilization"] >= 0.90 for m in sidecar["final_epoch_metrics"]
    )

    out_json = PRODUCTS_ROOT / "sidecar.json"
    with open(out_json, "w") as f:
        json.dump(sidecar, f, indent=2, default=str)
    print(f"\nSidecar saved: {out_json}", flush=True)
    print(f"Gate 1 {'✅ PASS' if sidecar['gate1_pass'] else '❌ FAIL'}", flush=True)


if __name__ == "__main__":
    main()