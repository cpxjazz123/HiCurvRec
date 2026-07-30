#!/usr/bin/env python3
"""Task #477 — Issue #59 Stage 1 重训练 #56 fixed (GPU 0).

基于 Issue #58 审计 + #59 修复, 重跑 Issue #56 Stage 1 训练:
  - 使用 FreeCurvVectorQuantizationMixedCurvFixed (α_l 真正可微)
  - recipe 跟 #156 完全一致 (κ=0.74, α_init=0.5, lr=1e-3, 1000 epoch)
  - 验证: α_l 不再全程停在 0.5, Stage 1 loss > 0.01, SID 3-digit unique > 50%

R12: 每个 epoch eval 后保存 ckpt + 删除旧.
R4: 修改后 py_compile 验证.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO / "HG-Rec"))

from model.hrqvae_free_curv import FreeCurvHRQVAE, FreeCurvResidualVectorQuantization
# Issue #59 fix: 用 Fixed 版本
from model.hrqvae_issue55_56_fixed import FreeCurvVectorQuantizationMixedCurvFixed

EMB_PATH_DEFAULT = REPO / "HG-Rec/dataset/Instruments/item_emb.parquet"


def load_item_embeddings(path: Path) -> torch.Tensor:
    df = pd.read_parquet(path)
    if "emb" in df.columns:
        emb = np.stack(df["emb"].values)
    elif "embedding" in df.columns:
        emb = np.stack(df["embedding"].values)
    else:
        col = df.columns[-1]
        emb = np.stack(df[col].values)
    print(f"  Loaded embeddings: shape={emb.shape}, dtype={emb.dtype}")
    return torch.tensor(emb, dtype=torch.float32)


def replace_vq_with_mixed_curv_fixed(model: FreeCurvHRQVAE, kappa_fixed: float) -> None:
    """替换 hrq.vq_layers 为 FreeCurvVectorQuantizationMixedCurvFixed (Issue #59 修复)."""
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
        vq = FreeCurvVectorQuantizationMixedCurvFixed(
            n_e=n_e, e_dim=e_dim, M=M,
            kappa_fixed=kappa_fixed,  # Issue #59: 用 fixed kappa
            beta=beta,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            sk_eps=sk_eps,
            sk_iters=sk_iters,
        )
        # Issue #59 fix: 移到正确 device (parent 可能没传 device 到子模块)
        vq = vq.to(next(model.parameters()).device)
        new_layers.append(vq)
    model.hrq.vq_layers = torch.nn.ModuleList(new_layers)
    print(f"  Replaced vq_layers with {len(new_layers)} MixedCurvFixed layers (κ={kappa_fixed})")


def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    print(f"Task #477 — Issue #59 Stage 1 重训练 #56 Fixed")
    print(f"  kappa_fixed = {args.kappa_fixed}")
    print(f"  num_epochs  = {args.num_epochs}")
    print(f"  lr          = {args.lr}")
    print(f"  beta        = {args.beta}")
    print(f"  device      = {device}")
    print("=" * 70)

    # Load embeddings
    emb = load_item_embeddings(EMB_PATH_DEFAULT).to(device)
    print(f"  Embeddings: {emb.shape}, on {device}")

    # Build model
    layers = [512, 256, 128, 64]
    in_dim = emb.shape[1]
    model = FreeCurvHRQVAE(
        in_dim=in_dim,
        layers=layers,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        kappa_max=args.kappa_fixed,  # 用 kappa_max 作为占位, replace 后改 fixed
        beta=args.beta,
        kmeans_init=True,
        kmeans_iters=100,
        sk_eps=[0.0] * len(args.num_emb_list),
        sk_iters=30,
        quant_loss_weight=args.beta,
    ).to(device)
    replace_vq_with_mixed_curv_fixed(model, args.kappa_fixed)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # DataLoader
    dataset = TensorDataset(emb)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, drop_last=False)

    # Output dir
    save_dir = REPO / "products/task477/ckpt/Instruments"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Train
    best_loss = float("inf")
    history = []
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batch = 0
        for (x,) in loader:
            x = x.to(device)
            x_q, vq_loss, indices = model(x)
            recon_loss = F.mse_loss(x_q, x)
            total_loss = recon_loss + vq_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            n_batch += 1
        epoch_loss /= max(n_batch, 1)

        # Track α_l/scale_l values
        alpha_l_values = []
        for vq in model.hrq.vq_layers:
            alpha_l_values.append(vq.alpha_l.item())
        mean_alpha = float(np.mean(alpha_l_values))

        history.append({
            "epoch": epoch,
            "loss": epoch_loss,
            "mean_alpha_l": mean_alpha,
        })

        # R12: 每 epoch 保存 (覆盖前一个)
        ckpt_path = save_dir / "best_loss_model.pth"
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss,
            "history": history,
        }, ckpt_path)

        if epoch_loss < best_loss:
            best_loss = epoch_loss

        if epoch % 10 == 0 or epoch <= 5:
            print(f"  epoch {epoch:4d}/{args.num_epochs}: loss={epoch_loss:.6f}, "
                  f"best={best_loss:.6f}, mean_alpha_l={mean_alpha:.4f}")

    # Save final history
    history_path = save_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n  ✅ Training done. best_loss={best_loss:.6f}")
    print(f"     ckpt: {ckpt_path}")
    print(f"     history: {history_path}")
    print(f"     Initial alpha_l was 0.5 (sigmoid(0)=0.5)")
    print(f"     Final mean_alpha_l: {history[-1]['mean_alpha_l']:.4f}")
    print(f"     ⚠️  若 mean_alpha_l 始终 ≈ 0.5 → Bug #1 修复未生效 (NO-GO)")
    print(f"     ✅ 若 mean_alpha_l 偏离 0.5 → 修复有效 (GO Stage 2)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kappa_fixed", type=float, default=0.74)
    parser.add_argument("--num_emb_list", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--e_dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
