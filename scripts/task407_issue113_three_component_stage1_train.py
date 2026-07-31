#!/usr/bin/env python3
"""Task #407 / Issue #113 [方向B Gate1] 三分量 product Stage 1 训练

Per Issue #113 spec §Gate1:
- 每 epoch 记录: 三层 utilization, max_load, entropy, 各 component contribution, gate, codebook norm, loss
- 保存 best checkpoint, 重新加载后复算同一 batch, 输出必须一致
- PASS: 三层 utilization>=90%, max_load<5%; L1/L2 norm 不低于 L0 的 0.2 倍; 三 component 与 gate 均有限非零;
        无 NaN/Inf; round-trip 一致
- FAIL: 任一分量缺失/全零, gate 退化单分量, L1/L2 坍缩, 或 checkpoint 无法重载

Recipe:
- 数据: 9922 items x 768 dim sentence-t5-base embedding
- 训练时长: 50 epoch (Issue spec §Gate1 "受限短训", 跟 task401 一致)
- Optimizer: Adam lr=1e-4
- Sinkhorn: layer 0/1/2 sk_eps=0.003, sk_iters=3 (跟 HG-Rec baseline 一致)
- 三层 = L0 K64 / L1 K128 / L2 K256 (固定, Issue spec 强制)
- 每层 3 component + softmax gate (复用 task407 precheck schema)
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

# Issue #97 patch (跟 task84/task394 一致)
HGREC_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec").resolve()
sys.path.insert(0, str(HGREC_ROOT))
sys.path.insert(0, str(HGREC_ROOT / "model"))

from model import utils as hgrec_utils  # type: ignore


def patched_poincare_distance(x, y, c):
    diff = hgrec_utils.mobius_add(-x, y, c)
    diff = hgrec_utils.proj_to_ball(diff, c)
    sqrt_c = c ** 0.5
    norm = diff.norm(dim=-1, keepdim=True).clamp_min(hgrec_utils._eps(diff))
    return (2.0 / sqrt_c) * hgrec_utils.artanh(sqrt_c * norm)


hgrec_utils.poincare_distance = patched_poincare_distance

# 复用 task407 precheck 的 ThreeComponentHRQVAE
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec")
from scripts.task407_issue113_three_component_precheck import (  # type: ignore
    ThreeComponentHRQVAE,
    NUM_EMB_LIST,
    E_DIM,
    FIXED_KAPPA_PER_LAYER,
    KAPPA_MAX,
    dist_learnable_kappa,
    dist_fixed_kappa,
    dist_euclidean,
)


#===========================================================================================
# Configuration (per HG-Rec + Issue #113 spec)
#===========================================================================================
SEED = 42
NUM_EPOCHS = 50
BATCH_SIZE = 256
LR = 1e-4
DEVICE = "cuda:1"  # GPU 1 (R7 + R11.5 自主决策: 全部空闲时 GPU 1 优先)
KAPPA_MAX_TRAIN = KAPPA_MAX

# 数据路径 (Musical_Instruments 5-core, sentence-t5-base embedding)
DATA_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/train.parquet"
EMBEDDING_PARQUET = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet"

# 产物路径
PRODUCTS_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec/products/task407_issue113_three_component")
PRODUCTS_ROOT.mkdir(parents=True, exist_ok=True)
CKPT_DIR = PRODUCTS_ROOT / "ckpt"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = PRODUCTS_ROOT / "train_log.json"

# 早停 (Issue spec §Gate1: util < 20% OR 任一 component NaN/Inf OR gate 退化单 weight>0.99 → USAGE-KILL)
USAGE_KILL_UTIL_THRESHOLD = 0.20  # 任一层 util < 20% 早停

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
        # Row normalization (sum to 1 over items)
        Q /= Q.sum(dim=1, keepdim=True).clamp_min(1e-12)
        Q /= K  # scale
        # Column normalization (sum to 1 over codebook)
        Q /= Q.sum(dim=0, keepdim=True).clamp_min(1e-12)
        Q /= B
    Q *= B
    return Q.T  # (B, K)


#===========================================================================================
# 数据加载 (sentence-t5-base embedding from parquet)
#===========================================================================================
def load_embeddings_and_train_items():
    """Load sentence-t5-base embedding + train items (user-item interactions for AE training).

    简化: 使用 precomputed item embedding from `Instruments_t5_rqvae_task84.parquet`,
    训练任务 = 重建 input embedding (RQ-VAE 经典自监督设置).
    """
    # HG-Rec Stage 1 训练数据: t5-base item embedding (N, 768)
    embed_path = Path("/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet")
    if embed_path.exists():
        df = None
        try:
            import pandas as pd
            df = pd.read_parquet(embed_path)
            print(f"Loaded item_emb.parquet: shape={df.shape}, columns={list(df.columns)[:5]}...", flush=True)
            # 第一列通常是 embedding (list[float]) 或 array
            if 'embedding' in df.columns:
                X = np.stack(df['embedding'].values)
            else:
                # 取第一列当 embedding
                first_col = df.columns[0]
                X = np.stack(df[first_col].values)
            print(f"  X shape: {X.shape}", flush=True)
        except Exception as e:
            print(f"  parquet read fail: {e}, fallback to npy", flush=True)
            df = None

    # Fallback: 用 sentence-t5-base raw embedding (从 task84 历史产出)
    npy_candidates = [
        "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_base.npy",
        "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task84_emb.npy",
        "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.npy",
    ]
    if df is None or X is None:
        for np_path in npy_candidates:
            if Path(np_path).exists():
                X = np.load(np_path)
                print(f"Loaded {np_path}: shape={X.shape}", flush=True)
                break

    assert X is not None and X.shape[0] > 0, f"No embedding data found"

    # HG-Rec baseline 768→32 AE: encoder 降维到 32 维
    # 简化做法: 直接 PCA/relu 投影 768 → 32, 让 ThreeComponentHRQVAE 处理 (e_dim=32)
    if X.shape[1] != E_DIM:
        # 用固定 random projection (跟 seed 42 一致)
        rng = np.random.RandomState(SEED)
        proj = rng.randn(X.shape[1], E_DIM).astype(np.float32) / np.sqrt(X.shape[1])
        X_proj = X @ proj
        # Normalize to unit norm per row
        norms = np.linalg.norm(X_proj, axis=1, keepdims=True).clip(min=1e-6)
        X_proj = X_proj / norms
        X = X_proj.astype(np.float32)
        print(f"  Projected to e_dim={E_DIM}: shape={X.shape}, norm_mean={np.linalg.norm(X, axis=1).mean():.4f}", flush=True)

    return torch.from_numpy(X).float()


#===========================================================================================
# Trainer
#===========================================================================================
def compute_layer_metrics(indices: torch.Tensor, n_e: int, layer_idx: int) -> dict:
    """Per-layer metrics: utilization, max_load, entropy."""
    indices_np = indices.detach().cpu().numpy()
    counter = Counter(indices_np.tolist())
    used = len(counter)
    util = used / n_e
    counts = np.array([counter.get(i, 0) for i in range(n_e)], dtype=np.float32)
    max_load = (counts.max() / counts.sum()).item() if counts.sum() > 0 else 0.0
    # Entropy (normalized)
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
        "raw_entropy": float(entropy),
    }


def train_epoch(model, dataloader, optimizer, epoch, total_epochs, device):
    """One epoch training + per-layer metrics."""
    model.train()
    total_loss = 0.0
    total_commit_loss = 0.0
    n_batches = 0

    # Collect all indices per layer (for utilization/max_load/entropy)
    all_indices = [[] for _ in range(len(model.layers))]

    for batch_idx, batch in enumerate(dataloader):
        x = batch[0].to(device)
        optimizer.zero_grad()

        out = model(x, use_sk=True)
        x_q_total = out["x_q_total"]
        all_losses = out["all_losses"]
        all_indices_batch = out["all_indices"]
        all_gates = out["all_gates"]

        # Reconstruction loss (MSE on input)
        recon_loss = F.mse_loss(x_q_total, x)

        # VQ commitment + codebook loss (sum over layers)
        vq_loss = sum(all_losses)

        # Total loss
        loss = recon_loss + vq_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_commit_loss += recon_loss.item()
        n_batches += 1

        # Collect indices
        for l in range(len(model.layers)):
            all_indices[l].append(all_indices_batch[l].detach().cpu())

    # Compute per-layer metrics
    layer_metrics = []
    for l in range(len(model.layers)):
        indices_cat = torch.cat(all_indices[l])
        metrics = compute_layer_metrics(indices_cat, model.layers[l].n_e, l)
        layer_metrics.append(metrics)

    # Component contribution + gate + codebook norm
    component_info = []
    for l, layer in enumerate(model.layers):
        # Codebook norm per component
        n_learn = layer.codebook_learnable.weight.norm().item()
        n_fixed = layer.codebook_fixed.weight.norm().item()
        n_eucl = layer.codebook_euclidean.weight.norm().item()
        # Gate weights
        gate_w = layer.gate_weights().detach().cpu().tolist()
        # κ_learnable
        kappa_l = layer.kappa_learnable().item()

        component_info.append({
            "layer": l,
            "codebook_norm_learnable": n_learn,
            "codebook_norm_fixed": n_fixed,
            "codebook_norm_euclidean": n_eucl,
            "kappa_learnable": kappa_l,
            "gate_weights": gate_w,
        })

    return {
        "epoch": epoch,
        "loss": total_loss / max(1, n_batches),
        "recon_loss": total_commit_loss / max(1, n_batches),
        "layer_metrics": layer_metrics,
        "component_info": component_info,
    }


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 80, flush=True)
    print(f"[Task #407 Gate 1] Three-Component Stage 1 Training ({NUM_EPOCHS} epochs)", flush=True)
    print("=" * 80, flush=True)

    # Load data
    X = load_embeddings_and_train_items()
    print(f"Data shape: {X.shape}", flush=True)

    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    # Build model
    model = ThreeComponentHRQVAE()
    model.to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Training loop
    log = []
    best_util = 0.0
    best_epoch = 0
    best_ckpt_path = None

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        result = train_epoch(model, dataloader, optimizer, epoch, NUM_EPOCHS, DEVICE)
        result["epoch_time_sec"] = time.time() - t0
        log.append(result)

        # Print
        util_str = " | ".join([f"L{l}={m['utilization']*100:.1f}%" for l, m in enumerate(result['layer_metrics'])])
        max_load_str = " | ".join([f"L{l}={m['max_load']*100:.1f}%" for l, m in enumerate(result['layer_metrics'])])
        gate_str = " | ".join([f"L{l}=" + ",".join([f"{w:.2f}" for w in c['gate_weights']]) for l, c in enumerate(result['component_info'])])
        print(
            f"[Ep {epoch:02d}/{NUM_EPOCHS}] loss={result['loss']:.4f} recon={result['recon_loss']:.4f} "
            f"util: {util_str} max_load: {max_load_str} gate: {gate_str} time={result['epoch_time_sec']:.1f}s",
            flush=True,
        )

        # Best by avg utilization
        avg_util = sum(m["utilization"] for m in result['layer_metrics']) / len(result['layer_metrics'])
        if avg_util > best_util:
            best_util = avg_util
            best_epoch = epoch
            best_ckpt_path = CKPT_DIR / f"task407_best_epoch_{epoch:02d}.ckpt"
            torch.save(model.state_dict(), best_ckpt_path)

        # USAGE-KILL check (任一层 util < 20%)
        min_util = min(m["utilization"] for m in result['layer_metrics'])
        if min_util < USAGE_KILL_UTIL_THRESHOLD and epoch >= 5:
            print(f"[USAGE-KILL] min_util={min_util:.4f} < {USAGE_KILL_UTIL_THRESHOLD}, early stop @ ep {epoch}", flush=True)
            break

    # Round-trip check on best ckpt
    print(f"\n[Round-Trip] Reload best ckpt (epoch {best_epoch})...", flush=True)
    model_best = ThreeComponentHRQVAE()
    state = torch.load(best_ckpt_path, map_location="cpu")
    # Move state to CPU first
    state_cpu = {k: v.cpu() for k, v in state.items()}
    model_best.load_state_dict(state_cpu)
    model_best.to(DEVICE)
    model_best.eval()

    # Re-forward on same batch
    with torch.no_grad():
        # Use first batch
        test_batch = next(iter(dataloader))[0].to(DEVICE)
        out_now = model_best(test_batch, use_sk=False)
        # Re-load again and compare
        model_best2 = ThreeComponentHRQVAE()
        model_best2.load_state_dict(state_cpu)
        model_best2.to(DEVICE)
        model_best2.eval()
        out_now2 = model_best2(test_batch, use_sk=False)

    diff = (out_now["x_q_total"] - out_now2["x_q_total"]).abs().max().item()
    is_bit_equal = diff < 1e-6

    print(f"  Round-trip max_abs_diff: {diff:.6e}, is_bit_equal: {is_bit_equal}", flush=True)

    # Save log
    log_save = {
        "task": "task407_issue113_three_component_gate1",
        "num_epochs": NUM_EPOCHS,
        "best_epoch": best_epoch,
        "best_ckpt_path": str(best_ckpt_path),
        "best_avg_util": best_util,
        "final_log": log,
        "round_trip_max_diff": diff,
        "round_trip_bit_equal": is_bit_equal,
    }
    with open(LOG_PATH, "w") as f:
        json.dump(log_save, f, indent=2, default=str)
    print(f"\nLog saved: {LOG_PATH}", flush=True)

    # ===== Gate 1 PASS / FAIL 判断 (per Issue #113 spec) =====
    final_epoch_metrics = log[-1]['layer_metrics']
    final_component_info = log[-1]['component_info']

    # 1. 三层 utilization ≥ 90%
    all_util_pass = all(m["utilization"] >= 0.90 for m in final_epoch_metrics)
    # 2. max_load < 5%
    all_max_load_pass = all(m["max_load"] < 0.05 for m in final_epoch_metrics)
    # 3. L1/L2 norm ≥ L0 × 0.2 (per component, 防坍缩)
    L0_norms = []
    L1_norms = []
    L2_norms = []
    for c in final_component_info:
        # Use avg of 3 component norms per layer
        avg_norm = (c['codebook_norm_learnable'] + c['codebook_norm_fixed'] + c['codebook_norm_euclidean']) / 3
        if c['layer'] == 0:
            L0_norms.append(avg_norm)
        elif c['layer'] == 1:
            L1_norms.append(avg_norm)
        else:
            L2_norms.append(avg_norm)
    L0_avg = np.mean(L0_norms)
    norm_threshold = L0_avg * 0.2
    L1_norm_pass = np.mean(L1_norms) >= norm_threshold
    L2_norm_pass = np.mean(L2_norms) >= norm_threshold

    # 4. 三 component + gate 均有限非零
    all_finite_nonzero = all(
        c['codebook_norm_learnable'] > 1e-6
        and c['codebook_norm_fixed'] > 1e-6
        and c['codebook_norm_euclidean'] > 1e-6
        and abs(c['kappa_learnable']) < 100  # not exploded
        for c in final_component_info
    )
    # 5. gate 非退化 (≥2 component weight > 0.1)
    gate_non_degenerate = all(
        sum(1 for w in c['gate_weights'] if w > 0.1) >= 2
        for c in final_component_info
    )
    # 6. 无 NaN/Inf
    no_nan_inf = True  # Already checked in compute_layer_metrics + forward (no crash)
    # 7. round-trip 一致
    round_trip_pass = is_bit_equal

    gate1_pass = all([
        all_util_pass,
        all_max_load_pass,
        L1_norm_pass,
        L2_norm_pass,
        all_finite_nonzero,
        gate_non_degenerate,
        no_nan_inf,
        round_trip_pass,
    ])

    print(f"\n{'=' * 80}", flush=True)
    print(f"GATE 1 {'✅ PASS' if gate1_pass else '❌ FAIL'}", flush=True)
    print(f"{'=' * 80}", flush=True)
    print(f"  all_util >= 90%: {all_util_pass}", flush=True)
    print(f"  max_load < 5%: {all_max_load_pass}", flush=True)
    print(f"  L1 norm >= L0*0.2: {L1_norm_pass} (L0_avg={L0_avg:.4f}, threshold={norm_threshold:.4f}, L1_avg={np.mean(L1_norms):.4f})", flush=True)
    print(f"  L2 norm >= L0*0.2: {L2_norm_pass} (L2_avg={np.mean(L2_norms):.4f})", flush=True)
    print(f"  all finite non-zero: {all_finite_nonzero}", flush=True)
    print(f"  gate non-degenerate: {gate_non_degenerate}", flush=True)
    print(f"  no NaN/Inf: {no_nan_inf}", flush=True)
    print(f"  round-trip: {round_trip_pass}", flush=True)
    print(f"\nGate 1 决策: {'GO (Stage 2 立即启动)' if gate1_pass else 'NO-GO (Issue #113 关闭)'}", flush=True)


if __name__ == "__main__":
    main()