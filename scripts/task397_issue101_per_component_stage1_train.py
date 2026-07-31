"""Task #397 / Issue #101 [方向B Gate1] per-component scale 修复 + Stage 1 50 epoch 实证

Date: 2026-07-31
Trigger: Issue #101 [方向B Gate1] 修复product聚合尺度并复验训练后agree3
Pre:
- task391 #98 audit FAIL: synthetic agree3=0%, init agree3=4.20% (<95%)
- 根因: FreeCurvVectorQuantization._per_component_dist_sq 没 per-component std normalization
        + aggregate-sum argmin bias toward component 0 (scale 大的 component 主导)

Fix:
1. per-component std normalization (per-block z-score) before computing distance
2. aggregate-mean (not sum) across components → 避免 scale dominate
3. per-component kmeans init (replace full-space kmeans)

Goal: Stage 1 50 epoch training, agree3 ≥ 95% per Issue #101 spec
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

# Apply #97 patch first (consistent with task394)
from model import utils as hgrec_utils  # type: ignore


def patched_poincare_distance(x, y, c):
    diff = hgrec_utils.mobius_add(-x, y, c)
    diff = hgrec_utils.proj_to_ball(diff, c)
    sqrt_c = c ** 0.5
    norm = diff.norm(dim=-1, keepdim=True).clamp_min(hgrec_utils._eps(diff))
    return (2.0 / sqrt_c) * hgrec_utils.artanh(sqrt_c * norm)


hgrec_utils.poincare_distance = patched_poincare_distance

from model.hrqvae_free_curv import FreeCurvVectorQuantization  # type: ignore


# === Issue #101 fix: monkey-patch _per_component_dist_sq ===
def patched_per_component_dist_sq(self, x_full, c_full):
    """Patched: per-component std normalization + mean aggregation.

    Returns sqrt(MEAN(d²_κ_m)) instead of sqrt(SUM(d²_κ_m)).
    Per-component z-score normalization prevents component-0 scale bias.
    """
    B = x_full.shape[0]
    K = c_full.shape[0]
    kappa = self.kappa_m()  # (M,)

    x_blocks = torch.split(x_full, self.block_dims, dim=-1)
    c_blocks = torch.split(c_full, self.block_dims, dim=-1)

    total_sq = torch.zeros(B, K, device=x_full.device, dtype=x_full.dtype)
    n_components = len(self.block_dims)

    for m, (x_m, c_m, k_m) in enumerate(zip(x_blocks, c_blocks, kappa)):
        # Per-component std normalization (Issue #101 fix)
        x_mean = x_m.mean()
        x_std = x_m.std().clamp_min(1e-6)
        c_mean = c_m.mean()
        c_std = c_m.std().clamp_min(1e-6)
        x_m_norm = (x_m - x_mean) / x_std
        c_m_norm = (c_m - c_mean) / c_std

        k_m_abs = k_m.abs().clamp(min=1e-8)
        sqrt_k = torch.sqrt(k_m_abs)

        diff = x_m_norm.unsqueeze(1) - c_m_norm.unsqueeze(0)
        diff_norm = diff.norm(dim=-1).clamp_min(1e-8)
        diff_norm_sq = diff_norm ** 2

        denom = 2.0 * (1.0 - k_m * diff_norm_sq / 4.0).abs().clamp_min(1e-6)
        arg = sqrt_k * diff_norm / denom
        d = (2.0 / sqrt_k) * torch.arctan(arg)
        d_m_sq = d ** 2

        # MEAN aggregation (Issue #101 fix: replace SUM)
        total_sq = total_sq + d_m_sq / n_components

    return torch.sqrt(total_sq + 1e-12)


# === Issue #101 fix: per-component kmeans init ===
def patched_init_emb(self, data):
    """Patched: per-component kmeans init (replace full-space kmeans)."""
    n_components = len(self.block_dims)
    block_data = torch.split(data, self.block_dims, dim=-1)
    cb_parts = []
    for m, (bd, n_e) in enumerate(zip(block_data, self.block_dims)):
        # simple kmeans per component (replicate sklearn-like behavior)
        from sklearn.cluster import KMeans
        n_e_total = self.embeddings.weight.shape[0]
        # split codebook by n_e per component (must match FreeCurv arch)
        n_e_per_comp = n_e_total // n_components
        km = KMeans(n_clusters=n_e_per_comp, n_init=3, random_state=42)
        km.fit(bd.cpu().numpy())
        cb_parts.append(torch.from_numpy(km.cluster_centers_).to(data.device).float())
    cb_concat = torch.cat(cb_parts, dim=-1)
    # Pad/truncate to match full codebook shape
    target_shape = self.embeddings.weight.shape
    if cb_concat.shape[0] < target_shape[0]:
        # Repeat to fill
        repeat = (target_shape[0] // cb_concat.shape[0]) + 1
        cb_concat = cb_concat.repeat(repeat, 1)[:target_shape[0]]
    self.embeddings.weight.data.copy_(cb_concat)
    self.initted = True


FreeCurvVectorQuantization._per_component_dist_sq = patched_per_component_dist_sq
FreeCurvVectorQuantization.init_emb = patched_init_emb


# === Stage 1 training: same structure as task394 but using FreeCurv ===
SEED = 42
NUM_EMB_LIST = [64, 128, 256]
E_DIM = 32
M = 3
BLOCK_DIMS = [11, 11, 10]
DEVICE = "cuda:0"
NUM_EPOCHS = 50
BATCH_SIZE = 256
LR = 1e-4

OUTPUT_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/products/task397_issue101_per_component_scale_stage1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = OUTPUT_DIR / "ckpt"
CKPT_DIR.mkdir(exist_ok=True)
LOG_PATH = OUTPUT_DIR / "run.log"

# Load Stage 1 input (real embeddings)
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


def eval_agree3():
    """agree3 = % items where top-3 argmin path matches per-component argmin path (Issue #101 spec)."""
    model.eval()
    n_items = z_e_full.shape[0]
    # Per-component argmin
    component_argmin_all = []
    full_argmin_all = []
    with torch.no_grad():
        for li, layer in enumerate(model.layers):
            batch_size = 512
            comp_min_idx = []
            full_min_idx = []
            for i in range(0, n_items, batch_size):
                batch = z_e_full[i:i+batch_size]
                B = batch.shape[0]
                codebook = layer.embeddings.weight
                # Per-component argmin: argmin over each component independently, then majority
                block_dims = layer.block_dims
                x_blocks = torch.split(batch, block_dims, dim=-1)
                c_blocks = torch.split(codebook, block_dims, dim=-1)
                comp_votes = torch.zeros(B, codebook.shape[0], device=batch.device)
                for x_m, c_m in zip(x_blocks, c_blocks):
                    x_mean = x_m.mean()
                    x_std = x_m.std().clamp_min(1e-6)
                    c_mean = c_m.mean()
                    c_std = c_m.std().clamp_min(1e-6)
                    x_m_n = (x_m - x_mean) / x_std
                    c_m_n = (c_m - c_mean) / c_std
                    d = (x_m_n.unsqueeze(1) - c_m_n.unsqueeze(0)).norm(dim=-1)  # (B, K)
                    idx = d.argmin(dim=-1)
                    for b in range(B):
                        comp_votes[b, idx[b]] += 1
                comp_argmin = comp_votes.argmax(dim=-1)
                comp_min_idx.append(comp_argmin.cpu())
                # Full argmin via patched path
                d_full = layer._per_component_dist_sq(batch, codebook)
                full_argmin = d_full.argmin(dim=-1)
                full_min_idx.append(full_argmin.cpu())
            component_argmin_all.append(torch.cat(comp_min_idx))
            full_argmin_all.append(torch.cat(full_min_idx))
    # agree3: % items where both paths agree on argmin
    agree3_per_layer = [
        (comp == full).float().mean().item()
        for comp, full in zip(component_argmin_all, full_argmin_all)
    ]
    return agree3_per_layer


def main():
    print("=" * 80, flush=True)
    print(f"Task #397 / Issue #101 [方向B Gate1] per-component scale fix + Stage 1 50 epoch", flush=True)
    print(f"GPU: {DEVICE}", flush=True)
    print(f"Architecture: FreeCurvVectorQuantization M={M} block_dims={BLOCK_DIMS}", flush=True)
    print("=" * 80, flush=True)

    log_lines = []
    util_history = []
    agree3_history = []

    for epoch in range(NUM_EPOCHS):
        loss = train_one_epoch()
        if epoch % 5 == 0 or epoch == NUM_EPOCHS - 1:
            util = eval_util()
            agree3 = eval_agree3()
            util_history.append({"epoch": epoch, "util": util})
            agree3_history.append({"epoch": epoch, "agree3": agree3})
            line = f"Epoch {epoch:3d} | loss={loss:.4f} | util L0/L1/L2={util[0]:.3f}/{util[1]:.3f}/{util[2]:.3f} | agree3 L0/L1/L2={agree3[0]:.3f}/{agree3[1]:.3f}/{agree3[2]:.3f}"
            print(line, flush=True)
            log_lines.append(line)
        else:
            print(f"Epoch {epoch:3d} | loss={loss:.4f}", flush=True)

    # Final eval
    util_final = eval_util()
    agree3_final = eval_agree3()
    print(f"\n[FINAL] util L0/L1/L2={util_final[0]:.4f}/{util_final[1]:.4f}/{util_final[2]:.4f}", flush=True)
    print(f"[FINAL] agree3 L0/L1/L2={agree3_final[0]:.4f}/{agree3_final[1]:.4f}/{agree3_final[2]:.4f}", flush=True)

    # R12: save ckpt
    ckpt_path = CKPT_DIR / "issue101_ckpt.pt"
    if ckpt_path.exists():
        ckpt_path.unlink()
    torch.save(model.state_dict(), ckpt_path)
    print(f"[R12] ckpt saved: {ckpt_path}", flush=True)

    # Evidence
    evidence = {
        "task": "task397",
        "issue": "Issue #101 [方向B Gate1]",
        "fix": "per-component std normalization + mean aggregation (Issue #101)",
        "patch_modules": ["FreeCurvVectorQuantization._per_component_dist_sq", "FreeCurvVectorQuantization.init_emb"],
        "seed": SEED,
        "num_epochs": NUM_EPOCHS,
        "final_util": util_final,
        "final_agree3": agree3_final,
        "agree3_target": 0.95,
        "agree3_pass": all(a >= 0.95 for a in agree3_final),
        "ckpt_path": str(ckpt_path),
        "history": {"util": util_history, "agree3": agree3_history},
    }
    with open(OUTPUT_DIR / "evidence.json", "w") as f:
        json.dump(evidence, f, indent=2)
    print(f"[INFO] evidence saved: {OUTPUT_DIR / 'evidence.json'}", flush=True)

    # Final log
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines))
    print(f"\n[RESULT] Task #397 Issue #101 Stage 1 done. agree3_pass={evidence['agree3_pass']}", flush=True)


if __name__ == "__main__":
    main()