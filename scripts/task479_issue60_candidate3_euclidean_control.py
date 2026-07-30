#!/usr/bin/env python3
"""Task #479 — Issue #60 candidate 3: Euclidean control (no mixed_curv_dist).

Question: 如果把 mixed_curv_dist 替换成 pure Euclidean, encoder 还坍缩吗?

Decision tree (per Issue #60):
  - 不坍缩 → 根因锁定 mixed_curv_dist 梯度信号缺失
  - 仍坍缩 → 根因是 encoder 架构/commitment 设计, 候选 1/2 优先

Recipe 跟 #477/#478 一致, 仅替换 _per_component_dist_sq 用 Euclidean distance.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO / "HG-Rec"))

from model.hrqvae_free_curv import FreeCurvHRQVAE, FreeCurvResidualVectorQuantization
from model.hrqvae_issue55_56_fixed import FreeCurvVectorQuantizationMixedCurvFixed
import model.hrqvae_issue55_56_fixed as fixed_mod


class EuclideanVQ(FreeCurvVectorQuantizationMixedCurvFixed):
    """Issue #60 candidate 3: pure Euclidean distance (no mixed_curv_dist).

    Override _per_component_dist_sq to use ‖x - y‖² directly (Euclidean squared).
    Other logic (Sinkhorn, soft VQ, straight-through) inherited.
    """

    def _per_component_dist_sq(self, x_full: torch.Tensor, c_full: torch.Tensor) -> torch.Tensor:
        """Pure Euclidean squared distance, summed across components."""
        B = x_full.shape[0]
        K = c_full.shape[0]

        x_blocks = torch.split(x_full, self.block_dims, dim=-1)
        c_blocks = torch.split(c_full, self.block_dims, dim=-1)

        total_d = torch.zeros(B, K, device=x_full.device, dtype=x_full.dtype)
        for m, (x_m, c_m) in enumerate(zip(x_blocks, c_blocks)):
            x_m_exp = x_m.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
            c_m_exp = c_m.unsqueeze(0).expand(B, K, -1).reshape(B * K, -1)
            # Pure Euclidean squared distance (sum across component dim)
            d_m = ((x_m_exp - c_m_exp) ** 2).sum(dim=-1, keepdim=True)
            d_m = d_m.view(B, K)
            total_d = total_d + d_m

        return total_d

    def _mixed_curv_commitment_loss(self, x_q: torch.Tensor, x: torch.Tensor):
        """Pure Euclidean commitment/codebook loss (per-component)."""
        commitment_sq = torch.zeros((), device=x.device, dtype=x.dtype)
        codebook_sq = torch.zeros((), device=x.device, dtype=x.dtype)

        x_blocks_lat = torch.split(x, self.block_dims, dim=-1)
        x_blocks_q = torch.split(x_q, self.block_dims, dim=-1)

        for xl_m, xq_m in zip(x_blocks_lat, x_blocks_q):
            d_c = ((xq_m - xl_m.detach()) ** 2).sum(dim=-1).mean()
            d_b = ((xq_m.detach() - xl_m) ** 2).sum(dim=-1).mean()
            commitment_sq = commitment_sq + d_c
            codebook_sq = codebook_sq + d_b

        return commitment_sq, codebook_sq


def load_embeddings():
    df = pd.read_parquet(REPO / "HG-Rec/dataset/Instruments/item_emb.parquet")
    emb = np.stack(df["embedding"].values)
    return torch.tensor(emb, dtype=torch.float32)


def replace_with_euclidean(model: FreeCurvHRQVAE):
    n_e_list = model.num_emb_list
    e_dim = model.e_dim
    M = model.M
    beta = model.beta
    sk_eps = model.hrq.vq_layers[0].sk_eps
    sk_iters = model.hrq.vq_layers[0].sk_iters
    kmeans_init = model.hrq.vq_layers[0].kmeans_init
    kmeans_iters = model.hrq.vq_layers[0].kmeans_iters
    new_layers = []
    for n_e in n_e_list:
        vq = EuclideanVQ(
            n_e=n_e, e_dim=e_dim, M=M,
            kappa_fixed=1.0,  # unused for Euclidean
            beta=beta,
            kmeans_init=kmeans_init, kmeans_iters=kmeans_iters,
            sk_eps=sk_eps, sk_iters=sk_iters,
        ).to(next(model.parameters()).device)
        new_layers.append(vq)
    model.hrq.vq_layers = torch.nn.ModuleList(new_layers)
    print(f"  Replaced vq_layers with {len(new_layers)} EuclideanVQ layers (pure Euclidean)")


def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    print(f"Task #479 — Issue #60 candidate 3: Euclidean control")
    print(f"  num_epochs = {args.num_epochs}, beta = {args.beta}")
    print(f"  device = {device}")
    print("=" * 70)

    emb = load_embeddings().to(device)
    print(f"  Loaded embeddings: {emb.shape}")

    in_dim = emb.shape[1]
    model = FreeCurvHRQVAE(
        in_dim=in_dim, layers=[512, 256, 128, 64],
        num_emb_list=args.num_emb_list, e_dim=args.e_dim,
        kappa_max=1.0, beta=args.beta,
        kmeans_init=True, kmeans_iters=100,
        sk_eps=[0.0] * len(args.num_emb_list), sk_iters=30,
        quant_loss_weight=args.beta,
    ).to(device)
    replace_with_euclidean(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    dataset = TensorDataset(emb)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    save_dir = REPO / f"products/task479/ckpt/Instruments"
    save_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_loss = float("inf")
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batch = 0
        for (x,) in loader:
            x = x.to(device)
            x_q, vq_loss, _ = model(x)
            recon_loss = F.mse_loss(x_q, x)
            total_loss = recon_loss + vq_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            n_batch += 1
        epoch_loss /= max(n_batch, 1)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
        history.append({"epoch": epoch, "loss": epoch_loss})
        if epoch % 10 == 0 or epoch <= 5:
            print(f"  epoch {epoch:3d}/{args.num_epochs}: loss={epoch_loss:.6f}")

    # Save ckpt + history
    ckpt_path = save_dir / "best_loss_model.pth"
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    torch.save({"epoch": args.num_epochs, "model_state_dict": model.state_dict(),
                "loss": best_loss, "history": history}, ckpt_path)

    # SID diversity check
    model.eval()
    all_indices = []
    with torch.no_grad():
        for i in range(0, len(emb), args.batch_size):
            x = emb[i:i + args.batch_size]
            x_q, _, indices = model(x, use_sk=False)
            all_indices.append(indices.cpu().numpy())
    sid = np.concatenate(all_indices, axis=0)
    n3 = len(np.unique(sid[:, :3], axis=0))
    n4 = len(np.unique(sid, axis=0))
    print(f"\n  ✅ Done. best_loss={best_loss:.6f}")
    print(f"  SID shape={sid.shape}, 3-digit unique={n3}/{len(sid)} ({100*n3/len(sid):.2f}%)")
    print(f"  ckpt: {ckpt_path}")
    print(f"  ⏱  Issue #60 candidate 3 verdict:")
    if 100*n3/len(sid) > 50:
        print(f"     ✅ NOT collapsed → 根因是 mixed_curv_dist 梯度信号缺失")
    else:
        print(f"     ❌ Still collapsed → 根因是 encoder 架构/commitment 设计, 候选 1/2 优先")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--num_emb_list", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--e_dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()