"""Task #396a / Issue #99 - Generate 4-digit SID file for Stage 3 T5 training.

Date: 2026-07-31
Trigger: task395 #99 Stage 2 PASS, need SID file for Stage 3 T5 training
Pre:
- task394 Stage 1 ckpt: products/task394_issue97_patch_stage1_verify/issue97_patch_stage1_ckpt.pt
- task395 Stage 2 Sinkhorn PASS: SID 4-digit unique 9922/9922 (100%)

Goal: Generate 4-digit SID .npy file compatible with GenRecDataset (HG-Rec training framework)
      Codebook_size = [64, 128, 256, 1] for Stage 3 input.
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

# Apply same patch as task394/task395
from model import utils as hgrec_utils  # type: ignore
from model.utils import HVectorQuantization  # type: ignore


def patched_poincare_distance(x, y, c):
    diff = hgrec_utils.mobius_add(-x, y, c)
    diff = hgrec_utils.proj_to_ball(diff, c)
    sqrt_c = c ** 0.5
    norm = diff.norm(dim=-1, keepdim=True).clamp_min(hgrec_utils._eps(diff))
    return (2.0 / sqrt_c) * hgrec_utils.artanh(sqrt_c * norm)


hgrec_utils.poincare_distance = patched_poincare_distance


def patched_vq_forward(self, x, use_sk=True):
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


HVectorQuantization.forward = patched_vq_forward


# Config
SEED = 42
NUM_EMB_LIST = [64, 128, 256]
E_DIM = 32
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SK_EPS = [0.003, 0.003, 0.003]
SK_ITERS = 5

# Output path
OUTPUT_SID_PATH = Path("/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task396.npy")
CKPT_PATH = Path("/home/wlia0047/ar57/wenyu/GeneRec/products/task394_issue97_patch_stage1_verify/issue97_patch_stage1_ckpt.pt")

# Load Stage 1 input
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


class PatchedHRQVAE(nn.Module):
    def __init__(self, num_emb_list, e_dim=32, sk_eps_list=None):
        super().__init__()
        if sk_eps_list is None:
            sk_eps_list = [0.003] * len(num_emb_list)
        self.layers = nn.ModuleList([
            HVectorQuantization(n_e=n_e, e_dim=e_dim, beta=0.25,
                                kmeans_init=True, kmeans_iters=10,
                                sk_eps=sk_eps, sk_iters=SK_ITERS)
            for n_e, sk_eps in zip(num_emb_list, sk_eps_list)
        ])

    def forward(self, z_e, use_sk=False):
        residual = z_e
        all_indices = []
        for layer in self.layers:
            x_q, _, idx = layer(residual, use_sk=use_sk)
            residual = residual - x_q
            all_indices.append(idx)
        return torch.stack(all_indices, dim=-1)


model = PatchedHRQVAE(NUM_EMB_LIST, E_DIM, sk_eps_list=SK_EPS).to(DEVICE)
if CKPT_PATH.exists():
    state = torch.load(CKPT_PATH, map_location=DEVICE)
    try:
        model.load_state_dict(state)
        print(f"[INFO] Loaded task394 ckpt from {CKPT_PATH}")
    except Exception as e:
        print(f"[WARN] ckpt load failed ({e}); using fresh-init")
        model.train()
        with torch.no_grad():
            _ = model(z_e_full, use_sk=False)
        model.eval()
else:
    print(f"[WARN] ckpt not found; using fresh-init")
    model.train()
    with torch.no_grad():
        _ = model(z_e_full, use_sk=False)
    model.eval()


# Stage 2 Sinkhorn inference
def stage2_inference():
    model.eval()
    n_items = z_e_full.shape[0]
    all_indices = []

    with torch.no_grad():
        for li, layer in enumerate(model.layers):
            indices_batch = []
            n_e = layer.n_e
            for i in range(0, n_items, 256):
                batch = z_e_full[i:i+256]
                B = batch.shape[0]
                codebook = layer.embeddings.weight
                latent_h = hgrec_utils.proj_to_ball(hgrec_utils.expmap0(batch, layer.c), layer.c)
                codebook_h = hgrec_utils.proj_to_ball(hgrec_utils.expmap0(codebook, layer.c), layer.c)
                x_exp = latent_h.unsqueeze(1).expand(B, n_e, -1)
                cb_exp = codebook_h.unsqueeze(0).expand(B, -1, -1)
                d = patched_poincare_distance(x_exp, cb_exp, layer.c).squeeze(-1)
                d_centered = layer.center_distance_for_constraint(d)
                d_centered = d_centered.double()
                Q = hgrec_utils.sinkhorn_algorithm(d_centered, layer.sk_eps, layer.sk_iters)
                idx = Q.argmax(dim=-1)
                indices_batch.append(idx.cpu())
            all_indices.append(torch.cat(indices_batch))

    sid_3digit = torch.stack(all_indices, dim=-1).numpy()  # (9922, 3)
    return sid_3digit


# Main
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 80)
    print(f"Task #396a / Issue #99 Generate 4-digit SID file for Stage 3")
    print("=" * 80)

    sid_3digit = stage2_inference()
    print(f"[INFO] 3-digit SID shape: {sid_3digit.shape}, sample: {sid_3digit[:5].tolist()}")

    # 4th-digit dedup
    unique_3d, inverse, counts_3d = np.unique(sid_3digit, axis=0, return_inverse=True, return_counts=True)
    print(f"[INFO] unique 3-digit SID: {len(unique_3d)}/{len(sid_3digit)}")

    digit_4 = np.zeros(len(sid_3digit), dtype=np.int64)
    counter = np.zeros(len(unique_3d), dtype=np.int64)
    for i in range(len(sid_3digit)):
        grp = inverse[i]
        digit_4[i] = counter[grp]
        counter[grp] += 1

    sid_4digit = np.concatenate([sid_3digit, digit_4[:, None]], axis=-1)  # (9922, 4)
    print(f"[INFO] 4-digit SID shape: {sid_4digit.shape}, sample: {sid_4digit[:5].tolist()}")

    # Per HG-Rec GenRecDataset convention: each row is [c0, c1, c2, c3]
    # codebook_size = [64, 128, 256, 1] for Stage 3 T5 (num_hierarchies=4)
    # Stage 3 expects 4-digit SID with 4th digit being dedup position (max K=1 means most are 0)
    np.save(OUTPUT_SID_PATH, sid_4digit)
    print(f"[INFO] Saved 4-digit SID to {OUTPUT_SID_PATH}")

    # Verify
    loaded = np.load(OUTPUT_SID_PATH)
    print(f"[VERIFY] Loaded SID shape: {loaded.shape}, dtype: {loaded.dtype}")
    print(f"[VERIFY] Loaded SID[0]: {loaded[0].tolist()}")

    # Save metadata
    meta = {
        "n_items": int(len(sid_4digit)),
        "codebook_size": [64, 128, 256, 1],
        "unique_4digit": int(len(set(tuple(s) for s in sid_4digit))),
        "ckpt_source": str(CKPT_PATH),
        "patch": "poincare_distance: proj_to_ball(diff, c) clamp",
    }
    meta_path = OUTPUT_SID_PATH.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] Saved meta to {meta_path}")

    print(f"\n[RESULT] SID generation complete")
    print(f"  Output: {OUTPUT_SID_PATH}")
    print(f"  Shape: {sid_4digit.shape}")
    print(f"  Unique 4-digit: {meta['unique_4digit']}/{meta['n_items']} ({meta['unique_4digit']/meta['n_items']*100:.2f}%)")


if __name__ == "__main__":
    main()