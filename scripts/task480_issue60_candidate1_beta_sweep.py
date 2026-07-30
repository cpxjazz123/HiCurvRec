#!/usr/bin/env python3
"""Task #480 — Issue #60 candidate 1: β weight sweep.

Question: β=0.25 是否过大? 用 0.05/0.01 测试 commitment loss 权重影响.

Per Issue #60 §执行顺序: 候选 3 已证明 EUCLIDEAN 也坍缩, 候选 1/2 优先.
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

from model.hrqvae_free_curv import FreeCurvHRQVAE
from model.hrqvae_issue55_56_fixed import FreeCurvVectorQuantizationMixedCurvFixed


def load_embeddings():
    df = pd.read_parquet(REPO / "HG-Rec/dataset/Instruments/item_emb.parquet")
    emb = np.stack(df["embedding"].values)
    return torch.tensor(emb, dtype=torch.float32)


def replace_vq(model: FreeCurvHRQVAE, kappa_fixed: float):
    n_e_list = model.num_emb_list
    e_dim = model.e_dim
    M = model.M
    beta = model.beta
    new_layers = []
    for n_e in n_e_list:
        vq = FreeCurvVectorQuantizationMixedCurvFixed(
            n_e=n_e, e_dim=e_dim, M=M,
            kappa_fixed=kappa_fixed, beta=beta,
            kmeans_init=True, kmeans_iters=100,
            sk_eps=0.0, sk_iters=30,
        ).to(next(model.parameters()).device)
        new_layers.append(vq)
    model.hrq.vq_layers = torch.nn.ModuleList(new_layers)


def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    print(f"Task #480 — Issue #60 candidate 1: β={args.beta} smoke test")
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
    replace_vq(model, kappa_fixed=args.kappa_fixed)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    dataset = TensorDataset(emb)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    save_dir = REPO / f"products/task480_beta{args.beta}/ckpt/Instruments"
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
        if epoch % 5 == 0 or epoch <= 3:
            print(f"  epoch {epoch:3d}/{args.num_epochs}: loss={epoch_loss:.6f}")

    ckpt_path = save_dir / "best_loss_model.pth"
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    torch.save({"epoch": args.num_epochs, "model_state_dict": model.state_dict(),
                "loss": best_loss, "history": history,
                "beta": args.beta}, ckpt_path)

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
    print(f"  SID 3-digit unique = {n3}/{len(sid)} ({100*n3/len(sid):.2f}%)")
    print(f"  ckpt: {ckpt_path}")
    print(f"  ⏱  β={args.beta} verdict:")
    if 100*n3/len(sid) > 50:
        print(f"     ✅ NOT collapsed → β 是关键变量, 候选 1 确认")
    else:
        print(f"     ❌ Still collapsed → β 不是根因, 候选 2 优先")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--kappa_fixed", type=float, default=0.74)
    parser.add_argument("--num_emb_list", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--e_dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:3")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()