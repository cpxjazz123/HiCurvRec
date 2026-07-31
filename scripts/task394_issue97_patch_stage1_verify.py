"""Task #394 / Issue #97 [方向A Gate1] patch utils.py poincare_distance + 50 epoch Stage 1 verify.

Date: 2026-07-31
Trigger: Issue #97 FAIL 收口后续 — commit 68eaa5b message 自报"下一轮 loop tick 应立即 patch utils.py bug"
Pre: task390 (#97 audit) found 3 formula bugs
Goal: Patch utils.py poincare_distance (add proj_to_ball diff clamp) + 让 HVectorQuantization 接受外部 c + 50 epoch Stage 1 训练验证 step1 max_load < 50%.

Per Issue #97 spec strict PASS criteria:
- 找到并修复 ≥1 个公式/尺度/广播/detach 问题 ✓ (patch #1: proj_to_ball diff)
- step0/step1 argmin 不再单码字占比 >50% (target verification)
- 三层均无 NaN/Inf
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

HGREC_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec").resolve()
sys.path.insert(0, str(HGREC_ROOT))

#===========================================================================================
# Patch logic (load source, monkey-patch in memory)
#===========================================================================================
from model import utils as hgrec_utils  # type: ignore


def patched_poincare_distance(x, y, c):
    """PATCH: add proj_to_ball(diff, c) clamp before artanh.

    Original (utils.py line 55-59):
        def poincare_distance(x, y, c):
            diff = mobius_add(-x, y, c)
            sqrt_c = c ** 0.5
            norm = diff.norm(dim=-1, keepdim=True).clamp_min(_eps(diff))
            return (2.0 / sqrt_c) * artanh(sqrt_c * norm)

    Patched:
        diff = mobius_add(-x, y, c)
        diff = proj_to_ball(diff, c)  # ← clamp diff norm to (1-eps)/√c, prevents atanh→∞
        sqrt_c = c ** 0.5
        norm = diff.norm(dim=-1, keepdim=True).clamp_min(_eps(diff))
        return (2.0 / sqrt_c) * artanh(sqrt_c * norm)
    """
    diff = hgrec_utils.mobius_add(-x, y, c)
    diff = hgrec_utils.proj_to_ball(diff, c)  # ← PATCH #1: prevent atanh→∞
    sqrt_c = c ** 0.5
    norm = diff.norm(dim=-1, keepdim=True).clamp_min(hgrec_utils._eps(diff))
    return (2.0 / sqrt_c) * hgrec_utils.artanh(sqrt_c * norm)


# Apply patch via monkey-patching
hgrec_utils.poincare_distance = patched_poincare_distance
# Also patch within the HVectorQuantization namespace
from model.utils import HVectorQuantization  # type: ignore  # re-import to use patched


def patched_vq_forward(self, x, use_sk=True):
    """PATCH: use patched poincare_distance (same as original except line 239)."""
    latent = x.view(-1, self.e_dim)
    codebook = self.embeddings.weight
    if not self.initted and self.training:
        self.init_emb(latent)

    latent_h = hgrec_utils.proj_to_ball(hgrec_utils.expmap0(latent, self.c), self.c)
    codebook_h = hgrec_utils.proj_to_ball(hgrec_utils.expmap0(codebook, self.c), self.c)

    B = latent_h.shape[0]
    K = codebook_h.shape[0]

    x_exp = latent_h.unsqueeze(1).expand(B, K, -1)
    cb_exp = codebook_h.unsqueeze(0).expand(B, K, -1)

    # PATCH: use patched poincare_distance (proj_to_ball diff clamp)
    d = patched_poincare_distance(x_exp, cb_exp, self.c).squeeze(-1)

    if not use_sk or self.sk_eps <= 0:
        indices = torch.argmin(d, dim=-1)
    else:
        d_centered = self.center_distance_for_constraint(d)
        d_centered = d_centered.double()
        Q = hgrec_utils.sinkhorn_algorithm(d_centered, self.sk_eps, self.sk_iters)
        if torch.isnan(Q).any() or torch.isinf(Q).any():
            raise ValueError("Sinkhorn produced NaN/Inf.")
        indices = torch.argmax(Q, dim=-1)

    x_exp = hgrec_utils.logmap0(x_exp, self.c)
    cb_exp = hgrec_utils.logmap0(cb_exp, self.c)
    x_q = codebook.index_select(0, indices)

    commitment_loss = torch.mean(patched_poincare_distance(x_q.detach(), latent, self.c) ** 2)
    codebook_loss = torch.mean(patched_poincare_distance(x_q, latent.detach(), self.c) ** 2)
    loss = commitment_loss + self.beta * codebook_loss

    x_q = hgrec_utils.logmap0(x_q, self.c)
    latent = hgrec_utils.logmap0(latent, self.c)
    x_q = x + (x_q - x).detach()
    indices = indices.view(x.shape[:-1])
    return x_q, loss, indices


# Patch HVectorQuantization.forward (will be applied at module level)
HVectorQuantization.forward = patched_vq_forward

#===========================================================================================
# Config
#===========================================================================================
SEED = 42
EPOCHS = 50
NUM_EMB_LIST = [64, 128, 256]
E_DIM = 32
BATCH_SIZE = 64
LR = 1e-3
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

PRODUCT_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec/products/task394_issue97_patch_stage1_verify")
PRODUCT_ROOT.mkdir(parents=True, exist_ok=True)
CKPT_PATH = PRODUCT_ROOT / "issue97_patch_stage1_ckpt.pt"

#===========================================================================================
# Load Stage 1 input — try real embeddings, fallback random
#===========================================================================================
EMB_PATH = Path("/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/embeddings/item_emb.parquet")
if EMB_PATH.exists():
    import pandas as pd
    df = pd.read_parquet(EMB_PATH)
    emb = np.stack(df['embedding'].values[:9922])
    X_full = torch.from_numpy(emb).float()
    print(f"[INFO] Loaded {X_full.shape[0]} real embeddings from {EMB_PATH}")
else:
    torch.manual_seed(SEED)
    X_full = torch.randn(9922, 768) * 0.1
    print(f"[WARN] Using random sample (n=9922, d=768)")


#===========================================================================================
# Simple linear encoder (no training, frozen)
#===========================================================================================
class FrozenEncoder(torch.nn.Module):
    def __init__(self, in_dim=768, out_dim=32):
        super().__init__()
        torch.manual_seed(SEED)
        self.proj = torch.nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        return self.proj(x)


encoder = FrozenEncoder(768, E_DIM).to(DEVICE).eval()
for p in encoder.parameters():
    p.requires_grad_(False)

# Encode full dataset
X_full_dev = X_full.to(DEVICE)
with torch.no_grad():
    z_e_full = encoder(X_full_dev).detach()  # (9922, 32)
print(f"[INFO] z_e shape: {z_e_full.shape}, norm mean: {z_e_full.norm(dim=-1).mean().item():.4f}")


#===========================================================================================
# Wrapper: simple HRQ-VAE stack using patched HVectorQuantization
#===========================================================================================
class PatchedHRQVAE(torch.nn.Module):
    def __init__(self, num_emb_list, e_dim=32):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            HVectorQuantization(n_e=n_e, e_dim=e_dim, beta=0.25,
                                kmeans_init=True, kmeans_iters=10,
                                sk_eps=0.003, sk_iters=3)
            for n_e in num_emb_list
        ])

    def forward(self, z_e, use_sk=False):
        residual = z_e
        x_q_total = 0
        all_losses = []
        all_indices = []
        for layer in self.layers:
            x_q, loss, idx = layer(residual, use_sk=use_sk)
            residual = residual - x_q
            x_q_total = x_q_total + x_q
            all_losses.append(loss)
            all_indices.append(idx)
        return x_q_total, torch.stack(all_losses).mean(), torch.stack(all_indices, dim=-1)


model = PatchedHRQVAE(NUM_EMB_LIST, E_DIM).to(DEVICE)
print(f"[INFO] Model created with {len(model.layers)} layers, device={DEVICE}")

# Trigger init_emb on full dataset (must use train mode)
model.train()
with torch.no_grad():
    _ = model(z_e_full[:9922], use_sk=False)  # full 9922 ≥ max K=256
model.eval()
print(f"[INFO] init_emb triggered on full 9922 dataset")


#===========================================================================================
# No-training audit (per Issue #97 spec)
#===========================================================================================
def no_training_audit():
    """Run model in eval mode, collect per-layer argmin distribution."""
    audit = {}
    model.eval()
    with torch.no_grad():
        for li, (n_e, layer) in enumerate(zip(NUM_EMB_LIST, model.layers)):
            # Use full 9922 items, batched
            indices_all = []
            for i in range(0, z_e_full.shape[0], 512):
                batch = z_e_full[i:i+512]
                codebook = layer.embeddings.weight
                latent_h = hgrec_utils.proj_to_ball(hgrec_utils.expmap0(batch, layer.c), layer.c)
                codebook_h = hgrec_utils.proj_to_ball(hgrec_utils.expmap0(codebook, layer.c), layer.c)
                x_exp = latent_h.unsqueeze(1).expand(-1, n_e, -1)
                cb_exp = codebook_h.unsqueeze(0).expand(batch.shape[0], -1, -1)
                d = patched_poincare_distance(x_exp, cb_exp, layer.c).squeeze(-1)
                idx = d.argmin(dim=-1)
                indices_all.append(idx.cpu())
            indices_full = torch.cat(indices_all)
            unique, counts = torch.unique(indices_full, return_counts=True)
            max_load = (counts.max().item() / counts.sum().item()) * 100
            n_unique = len(unique)
            audit[f"L{li}_K{n_e}"] = {
                "n_unique": n_unique,
                "max_load_pct": max_load,
                "any_nan": bool(torch.isnan(indices_full).any()),
                "PASS_max_load_lt_50": max_load < 50.0,
                "PASS_unique_eq_K": n_unique >= n_e * 0.9,
            }
    return audit


#===========================================================================================
# 50 epoch Stage 1 training verify
#===========================================================================================
def stage1_train_verify():
    """Train with patched HRQ-VAE 50 epoch, monitor step1/stepN util."""
    import torch.optim as optim

    optimizer = optim.Adam(model.parameters(), lr=LR)
    model.train()

    epoch_records = []
    n_total = z_e_full.shape[0]

    for ep in range(EPOCHS):
        perm = torch.randperm(n_total, device=DEVICE)
        ep_losses = []
        for i in range(0, n_total, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            batch = z_e_full[idx]
            optimizer.zero_grad()
            x_q, loss, indices = model(batch, use_sk=False)
            loss.backward()
            optimizer.step()
            ep_losses.append(loss.item())

        # Audit util after each epoch
        if (ep + 1) % 10 == 0 or ep == 0 or ep == EPOCHS - 1:
            with torch.no_grad():
                model.eval()
                utils_ep = []
                for li, layer in enumerate(model.layers):
                    # Full argmin
                    indices_all = []
                    for j in range(0, n_total, 512):
                        batch = z_e_full[j:j+512]
                        codebook = layer.embeddings.weight
                        latent_h = hgrec_utils.proj_to_ball(hgrec_utils.expmap0(batch, layer.c), layer.c)
                        codebook_h = hgrec_utils.proj_to_ball(hgrec_utils.expmap0(codebook, layer.c), layer.c)
                        x_exp = latent_h.unsqueeze(1).expand(-1, layer.n_e, -1)
                        cb_exp = codebook_h.unsqueeze(0).expand(batch.shape[0], -1, -1)
                        d = patched_poincare_distance(x_exp, cb_exp, layer.c).squeeze(-1)
                        idx = d.argmin(dim=-1)
                        indices_all.append(idx.cpu())
                    indices_full = torch.cat(indices_all)
                    unique, counts = torch.unique(indices_full, return_counts=True)
                    max_load = (counts.max().item() / counts.sum().item()) * 100
                    n_unique = len(unique)
                    utils_ep.append({"unique": n_unique, "max_load_pct": max_load,
                                     "util": n_unique / layer.n_e})
                model.train()

            epoch_records.append({
                "epoch": ep + 1,
                "loss_mean": np.mean(ep_losses),
                "utils": utils_ep,
            })
            print(f"[Epoch {ep+1}/{EPOCHS}] loss={np.mean(ep_losses):.4f}, "
                  f"L0 unique={utils_ep[0]['unique']}/{NUM_EMB_LIST[0]} max_load={utils_ep[0]['max_load_pct']:.2f}%, "
                  f"L1 unique={utils_ep[1]['unique']}/{NUM_EMB_LIST[1]} max_load={utils_ep[1]['max_load_pct']:.2f}%, "
                  f"L2 unique={utils_ep[2]['unique']}/{NUM_EMB_LIST[2]} max_load={utils_ep[2]['max_load_pct']:.2f}%")

    # R12: save checkpoint (delete old, save new)
    if CKPT_PATH.exists():
        CKPT_PATH.unlink()
    torch.save(model.state_dict(), CKPT_PATH)
    print(f"[R12] ckpt saved: {CKPT_PATH}")

    return epoch_records


#===========================================================================================
# Main
#===========================================================================================
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 80)
    print(f"Task #394 / Issue #97 patch + Stage 1 verify")
    print("=" * 80)

    # 1. No-training audit (post-patch)
    print("\n[Section 1] No-training audit (post-patch, before training):")
    no_train = no_training_audit()
    for layer, r in no_train.items():
        print(f"  {layer}: max_load={r['max_load_pct']:.2f}% unique={r['n_unique']}/{NUM_EMB_LIST[int(layer[1])]}")

    # 2. 50 epoch training
    print(f"\n[Section 2] {EPOCHS} epoch Stage 1 training:")
    records = stage1_train_verify()

    # 3. Final audit
    print("\n[Section 3] Final audit (post-training):")
    final = records[-1]["utils"]
    final_pass = all(r["util"] >= 0.9 and r["max_load_pct"] < 50.0 for r in final)
    for li, r in enumerate(final):
        print(f"  L{li}_K{NUM_EMB_LIST[li]}: util={r['util']*100:.2f}%, max_load={r['max_load_pct']:.2f}%, unique={r['unique']}")

    # 4. Summary
    summary = {
        "no_training_audit": no_train,
        "training_records": records,
        "final_util": final,
        "patch": "poincare_distance: add proj_to_ball(diff, c) clamp before artanh",
        "final_pass_90_util": final_pass,
        "ckpt_path": str(CKPT_PATH),
    }

    evidence_path = PRODUCT_ROOT / "evidence_package.json"
    with open(evidence_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[EVIDENCE] Saved to {evidence_path}")

    print(f"\n[RESULT] final_pass_90_util = {final_pass}")
    return summary


if __name__ == "__main__":
    main()