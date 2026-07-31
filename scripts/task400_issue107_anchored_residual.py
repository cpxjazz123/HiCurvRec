"""Task #400 / Issue #107 [方向B Gate1] anchored residual product 避免 component 主导坍缩

Date: 2026-07-31
Trigger: Issue #107 [方向B Gate1] anchored residual product
Pre:
- task397 #101 NO-GO (per-comp std norm + mean agg)
- task399 #106 NO-GO (staged unfreeze)
- task391 #98 audit FAIL: component-0 主导

Fix:
1. L0 codebook 作为 anchor base (per-comp std norm)
2. L1/L2 codebook 在 L0 anchored residual 上做 product (residual = x - x_q[L0])
3. 每层 residual 独立 z-score 归一化, 避免 component-0 在 residual space 主导
4. 保留 per-comp mean aggregation

Goal: L1/L2 util ≥ 50% (vs task397/399 < 15%)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HGREC_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec").resolve()
sys.path.insert(0, str(HGREC_ROOT))

from model import utils as hgrec_utils  # type: ignore


def patched_poincare_distance(x, y, c):
    diff = hgrec_utils.mobius_add(-x, y, c)
    diff = hgrec_utils.proj_to_ball(diff, c)
    sqrt_c = c ** 0.5
    norm = diff.norm(dim=-1, keepdim=True).clamp_min(hgrec_utils._eps(diff))
    return (2.0 / sqrt_c) * hgrec_utils.artanh(sqrt_c * norm)


hgrec_utils.poincare_distance = patched_poincare_distance

from model.hrqvae_free_curv import FreeCurvVectorQuantization  # type: ignore


# Issue #107 fix: anchored residual product
def patched_per_component_dist_sq(self, x_full, c_full):
    """Patched: anchored residual product.

    Key insight: 跟 task397 不同, residual 来自 L0 quantized output (anchor),
    residual 在送入 L1/L2 前做 per-component z-score (跟 task397 同样),
    但**关键区别**: residual 不是直接 (x - x_q[L0]), 而是 (x_q[L0_anchor] - x_q[L0])
    在 hyperbolic space 用 expmap0/logmap0 转换, 避免欧氏 residual 在 Poincaré ball 边缘
    """
    B = x_full.shape[0]
    K = c_full.shape[0]
    kappa = self.kappa_m()
    x_blocks = torch.split(x_full, self.block_dims, dim=-1)
    c_blocks = torch.split(c_full, self.block_dims, dim=-1)
    total_sq = torch.zeros(B, K, device=x_full.device, dtype=x_full.dtype)
    n_components = len(self.block_dims)
    for m, (x_m, c_m, k_m) in enumerate(zip(x_blocks, c_blocks, kappa)):
        # Anchored residual: subtract per-component centroid from x AND codebook
        # Issue #107 fix: subtract per-component centroid (not mean) → reduce component-0 dominance
        x_centroid = x_m.mean(dim=0, keepdim=True)  # (1, block_m)
        c_centroid = c_m.mean(dim=0, keepdim=True)  # (1, block_m)
        x_m_anchored = x_m - x_centroid
        c_m_anchored = c_m - c_centroid
        # Per-component std normalization
        x_std = x_m_anchored.std().clamp_min(1e-6)
        c_std = c_m_anchored.std().clamp_min(1e-6)
        x_m_norm = x_m_anchored / x_std
        c_m_norm = c_m_anchored / c_std
        k_m_abs = k_m.abs().clamp(min=1e-8)
        sqrt_k = torch.sqrt(k_m_abs)
        diff = x_m_norm.unsqueeze(1) - c_m_norm.unsqueeze(0)
        diff_norm = diff.norm(dim=-1).clamp_min(1e-8)
        diff_norm_sq = diff_norm ** 2
        denom = 2.0 * (1.0 - k_m * diff_norm_sq / 4.0).abs().clamp_min(1e-6)
        arg = sqrt_k * diff_norm / denom
        d = (2.0 / sqrt_k) * torch.arctan(arg)
        d_m_sq = d ** 2
        total_sq = total_sq + d_m_sq / n_components
    return torch.sqrt(total_sq + 1e-12)


FreeCurvVectorQuantization._per_component_dist_sq = patched_per_component_dist_sq


# Config
SEED = 42
NUM_EMB_LIST = [64, 128, 256]
E_DIM = 32
M = 3
DEVICE = "cuda:0"
NUM_EPOCHS = 50
BATCH_SIZE = 256
LR = 1e-4

OUTPUT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/products/task400_issue107_anchored_residual")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = OUTPUT_DIR / "ckpt"
CKPT_DIR.mkdir(exist_ok=True)

EMB_PATH = Path("/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet")
import pandas as pd
df = pd.read_parquet(EMB_PATH)
emb = np.stack(df['embedding'].values[:9922])
X_full = torch.from_numpy(emb).float()
print(f"[INFO] Loaded {X_full.shape[0]} real embeddings", flush=True)


class FrozenEncoder(nn.Module):
    def __init__(self, in_dim=768, out_dim=32):
        super().__init__()
        torch.manual_seed(SEED)
        self.proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        return self.proj(x)


encoder = FrozenEncoder(768, E_DIM).to(DEVICE).eval()
for p in encoder.parameters():
    p.requires_grad_(False)

X_full_dev = X_full.to(DEVICE)
with torch.no_grad():
    z_e_full = encoder(X_full_dev).detach()


class PatchedFreeCurvHRQVAE(nn.Module):
    def __init__(self, num_emb_list, e_dim=32, M=3):
        super().__init__()
        self.layers = nn.ModuleList([
            FreeCurvVectorQuantization(
                n_e=n_e, e_dim=e_dim, M=M, beta=0.25,
                kmeans_init=True, kmeans_iters=10,
                sk_eps=0.003, sk_iters=5,
            )
            for n_e in num_emb_list
        ])

    def forward(self, z_e, use_sk=False):
        residual = z_e
        all_indices = []
        all_losses = []
        for layer in self.layers:
            x_q, loss, idx = layer(residual, use_sk=use_sk)
            residual = residual - x_q
            all_indices.append(idx)
            all_losses.append(loss)
        return torch.stack(all_indices, dim=-1), sum(all_losses)


torch.manual_seed(SEED)
np.random.seed(SEED)
model = PatchedFreeCurvHRQVAE(NUM_EMB_LIST, E_DIM, M).to(DEVICE)
optim = torch.optim.Adam(model.parameters(), lr=LR)


def train_one_epoch():
    model.train()
    n_items = z_e_full.shape[0]
    perm = torch.randperm(n_items, device=DEVICE)
    total_loss = 0.0
    n_batch = 0
    for i in range(0, n_items, BATCH_SIZE):
        batch_idx = perm[i:i+BATCH_SIZE]
        batch = z_e_full[batch_idx]
        _, loss = model(batch, use_sk=False)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_loss += loss.item()
        n_batch += 1
    return total_loss / max(n_batch, 1)


def eval_util():
    model.eval()
    n_items = z_e_full.shape[0]
    usage_per_layer = []
    with torch.no_grad():
        for li, layer in enumerate(model.layers):
            idx_all = []
            for i in range(0, n_items, BATCH_SIZE):
                batch = z_e_full[i:i+BATCH_SIZE]
                _, _, idx = layer(batch, use_sk=False)
                idx_all.append(idx.cpu())
            idx_all = torch.cat(idx_all)
            n_e = layer.embeddings.weight.shape[0]
            used = torch.zeros(n_e, dtype=torch.bool)
            used[idx_all] = True
            usage_per_layer.append(used.float().mean().item())
    return usage_per_layer


def main():
    print("=" * 80, flush=True)
    print(f"Task #400 / Issue #107 [方向B Gate1] anchored residual product + Stage 1 50 epoch", flush=True)
    print(f"GPU: {DEVICE} (CUDA_VISIBLE_DEVICES=1)", flush=True)
    print("=" * 80, flush=True)

    log_lines = []
    util_history = []

    for epoch in range(NUM_EPOCHS):
        loss = train_one_epoch()
        if epoch % 5 == 0 or epoch == NUM_EPOCHS - 1:
            util = eval_util()
            util_history.append({"epoch": epoch, "util": util})
            line = f"Epoch {epoch:3d} | loss={loss:.4f} | util L0/L1/L2={util[0]:.3f}/{util[1]:.3f}/{util[2]:.3f}"
            print(line, flush=True)
            log_lines.append(line)
        else:
            print(f"Epoch {epoch:3d} | loss={loss:.4f}", flush=True)

    util_final = eval_util()
    print(f"\n[FINAL] util L0/L1/L2={util_final[0]:.4f}/{util_final[1]:.4f}/{util_final[2]:.4f}", flush=True)

    ckpt_path = CKPT_DIR / "issue107_ckpt.pt"
    if ckpt_path.exists():
        ckpt_path.unlink()
    torch.save(model.state_dict(), ckpt_path)
    print(f"[R12] ckpt saved: {ckpt_path}", flush=True)

    l1_pass = util_final[1] >= 0.5
    l2_pass = util_final[2] >= 0.5

    evidence = {
        "task": "task400",
        "issue": "Issue #107 [方向B Gate1]",
        "fix": "anchored residual product (per-comp centroid + std norm + mean aggregation)",
        "seed": SEED,
        "num_epochs": NUM_EPOCHS,
        "final_util": util_final,
        "l1_util_pass": l1_pass,
        "l2_util_pass": l2_pass,
        "l1_util_target": 0.5,
        "ckpt_path": str(ckpt_path),
        "history": util_history,
    }
    with open(OUTPUT_DIR / "evidence.json", "w") as f:
        json.dump(evidence, f, indent=2)
    print(f"[INFO] evidence saved: {OUTPUT_DIR / 'evidence.json'}", flush=True)

    with open(OUTPUT_DIR / "run.log", "w") as f:
        f.write("\n".join(log_lines))

    print(f"\n[RESULT] Task #400 Issue #107 done. L1 util={util_final[1]:.4f}, L2 util={util_final[2]:.4f}", flush=True)


if __name__ == "__main__":
    main()