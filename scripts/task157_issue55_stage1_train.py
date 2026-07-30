#!/usr/bin/env python3
"""Task #157 — Issue #55 Stage 1 训练 (Riemannian AdamW + per-layer s_l, GPU 3).

跟 Task #156 类似, 但:
  - optimizer: RiemannianAdamW 替代 Adam (codebook params)
  - 加 per-layer s_l 参数 (scale_recalibration)
  - kappa 用 κ_max=1.0 默认 (跟 HG-Rec baseline 一致)

Recipe:
  - num_emb_list=[64, 128, 256]
  - kappa_max=1.0 (跟 #84 baseline 一致)
  - scale_init=1.0
  - lr=1e-3
  - beta=0.25
  - sk_eps=[0, 0, 0.000]
  - layers=[512, 256, 128, 64]
  - epochs=1000
  - batch_size=1024

R12 强制保存 ckpt.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO / "HG-Rec"))

from model.hrqvae_free_curv import FreeCurvHRQVAE, FreeCurvResidualVectorQuantization
from model.hrqvae_issue55_56 import (
    FreeCurvVectorQuantizationMixedCurvWithScale,
    RiemannianAdamW,
)

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


def replace_vq_with_mixed_curv_scale(model: FreeCurvHRQVAE, kappa_fixed: float) -> None:
    """替换 hrq.vq_layers 为 FreeCurvVectorQuantizationMixedCurvWithScale (Issue #55+#56)."""
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
        vq = FreeCurvVectorQuantizationMixedCurvWithScale(
            n_e=n_e, e_dim=e_dim, M=M,
            kappa_max=kappa_fixed,
            beta=beta,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            sk_eps=sk_eps,
            sk_iters=sk_iters,
            kappa_fixed=kappa_fixed,
            scale_init=1.0,
        )
        new_layers.append(vq)

    model.hrq.vq_layers = torch.nn.ModuleList(new_layers)
    print(f"  Replaced {len(new_layers)} VQ layers with FreeCurvVectorQuantizationMixedCurvWithScale")
    print(f"  kappa_fixed = {kappa_fixed}, alpha_l = {model.hrq.vq_layers[0].alpha_l.item():.4f}, "
          f"scale_l = {model.hrq.vq_layers[0].scale_l.tolist()}")


def train(model: FreeCurvHRQVAE, dl: DataLoader, device: torch.device,
          n_epochs: int, lr: float, ckpt_dir: Path, eval_step: int = 5,
          log_every: int = 50) -> dict:
    """Train FreeCurvHRQVAE with Issue #55 Riemannian AdamW (for codebook params)
    + standard Adam (for encoder/decoder + α_l + s_l params)."""
    model.train()

    # Separate params:
    # - codebook params (embeddings): use RiemannianAdamW (Issue #55)
    # - encoder/decoder + α_l + s_l + theta: use standard AdamW
    codebook_params = []
    other_params = []
    for name, p in model.named_parameters():
        if "embeddings" in name:
            codebook_params.append(p)
        else:
            other_params.append(p)

    print(f"  Codebook params (RiemannianAdamW): {sum(p.numel() for p in codebook_params)}")
    print(f"  Other params (AdamW): {sum(p.numel() for p in other_params)}")

    optimizer_codebook = RiemannianAdamW(codebook_params, lr=lr, weight_decay=0.0)
    optimizer_other = torch.optim.AdamW(other_params, lr=lr, weight_decay=0.01)

    history = {"epoch_loss": [], "epoch_recon": [], "epoch_quant": [],
               "epoch_alpha_l": [], "epoch_scale_l": []}

    best_loss = float('inf')
    best_epoch = 0
    best_ckpt_path = ckpt_dir / "best_loss_model.pth"

    for epoch in range(1, n_epochs + 1):
        epoch_loss_sum = 0.0
        epoch_recon_sum = 0.0
        epoch_quant_sum = 0.0
        n_batch = 0
        t0 = time.time()

        for batch_idx, (x,) in enumerate(dl):
            x = x.to(device, non_blocking=True)
            out, rq_loss, indices = model(x, use_sk=True)
            loss, recon_loss = model.compute_loss(out, rq_loss, xs=x)

            optimizer_codebook.zero_grad()
            optimizer_other.zero_grad()
            loss.backward()
            optimizer_other.step()
            optimizer_codebook.step()

            epoch_loss_sum += loss.item()
            epoch_recon_sum += recon_loss.item()
            epoch_quant_sum += rq_loss.item()
            n_batch += 1

            if (batch_idx + 1) % log_every == 0:
                print(f"  [ep {epoch} batch {batch_idx+1}/{len(dl)}] "
                      f"loss={loss.item():.4f} recon={recon_loss.item():.4f} "
                      f"quant={rq_loss.item():.4f}", flush=True)

        avg_loss = epoch_loss_sum / n_batch
        avg_recon = epoch_recon_sum / n_batch
        avg_quant = epoch_quant_sum / n_batch
        alpha_l_per_layer = [vq.alpha_l.detach().cpu().tolist() for vq in model.hrq.vq_layers]
        scale_l_per_layer = [vq.scale_l.detach().cpu().tolist() for vq in model.hrq.vq_layers]

        history["epoch_loss"].append(avg_loss)
        history["epoch_recon"].append(avg_recon)
        history["epoch_quant"].append(avg_quant)
        history["epoch_alpha_l"].append(alpha_l_per_layer)
        history["epoch_scale_l"].append(scale_l_per_layer)

        elapsed = time.time() - t0
        print(f"  [Epoch {epoch}/{n_epochs}] avg_loss={avg_loss:.4f} "
              f"recon={avg_recon:.4f} quant={avg_quant:.4f} "
              f"alpha_l={alpha_l_per_layer[0]} scale_l={scale_l_per_layer[0]} "
              f"({elapsed:.1f}s)", flush=True)

        # R12: ckpt saving per epoch
        if epoch % eval_step == 0 or epoch == n_epochs:
            if avg_loss < best_loss:
                if best_ckpt_path.exists():
                    best_ckpt_path.unlink()
                best_loss = avg_loss
                best_epoch = epoch
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_codebook_state_dict": optimizer_codebook.state_dict(),
                    "optimizer_other_state_dict": optimizer_other.state_dict(),
                    "loss": avg_loss,
                    "alpha_l": alpha_l_per_layer,
                    "scale_l": scale_l_per_layer,
                }, best_ckpt_path)
                print(f"    [R12 ckpt saved] best_loss={best_loss:.4f} @ epoch {epoch}", flush=True)

    # Final ckpt save
    final_ckpt_path = ckpt_dir / "final_model.pth"
    if final_ckpt_path.exists():
        final_ckpt_path.unlink()
    torch.save({
        "epoch": n_epochs,
        "model_state_dict": model.state_dict(),
        "loss": avg_loss,
        "alpha_l": alpha_l_per_layer,
        "scale_l": scale_l_per_layer,
    }, final_ckpt_path)
    print(f"  [Final ckpt saved] {final_ckpt_path}", flush=True)

    history["best_loss"] = best_loss
    history["best_epoch"] = best_epoch
    return history


def main():
    parser = argparse.ArgumentParser(description="Task #157 Issue #55 Stage 1 training")
    parser.add_argument("--data_path", type=str, default=str(EMB_PATH_DEFAULT))
    parser.add_argument("--ckpt_dir", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/products/task157/ckpt/Instruments")
    parser.add_argument("--num_emb_list", type=int, nargs='+', default=[64, 128, 256])
    parser.add_argument("--e_dim", type=int, default=32)
    parser.add_argument("--M", type=int, default=1)
    parser.add_argument("--kappa_fixed", type=float, default=1.0,
                        help="Fixed κ for Issue #55 (跟 HG-Rec baseline c=1.0 一致)")
    parser.add_argument("--layers", type=int, nargs='+', default=[512, 256, 128, 64])
    parser.add_argument("--dropout_prob", type=float, default=0.0)
    parser.add_argument("--bn", type=bool, default=False)
    parser.add_argument("--loss_type", type=str, default="mse")
    parser.add_argument("--quant_loss_weight", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--kmeans_init", type=bool, default=True)
    parser.add_argument("--kmeans_iters", type=int, default=10)
    parser.add_argument("--sk_epsilons", type=float, nargs='+', default=[0.0, 0.0, 0.000])
    parser.add_argument("--sk_iters", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_step", type=int, default=5)
    parser.add_argument("--log_every", type=int, default=50)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    print(f"  Device: {device}")

    emb = load_item_embeddings(Path(args.data_path))
    in_dim = emb.shape[1]

    model = FreeCurvHRQVAE(
        in_dim=in_dim, num_emb_list=args.num_emb_list, e_dim=args.e_dim,
        M=args.M, kappa_max=args.kappa_fixed,
        layers=args.layers, dropout_prob=args.dropout_prob, bn=args.bn,
        loss_type=args.loss_type, quant_loss_weight=args.quant_loss_weight,
        beta=args.beta, kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_eps=args.sk_epsilons, sk_iters=args.sk_iters,
    ).to(device)

    replace_vq_with_mixed_curv_scale(model, kappa_fixed=args.kappa_fixed)
    model = model.to(device)

    ds = TensorDataset(emb)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Starting training: {args.epochs} epochs, lr={args.lr}, bs={args.batch_size}")
    print(f"  Recipe: num_emb_list={args.num_emb_list}, kappa_fixed={args.kappa_fixed}")
    print(f"  Ckpt dir: {ckpt_dir}\n")

    history = train(model, dl, device, n_epochs=args.epochs, lr=args.lr,
                    ckpt_dir=ckpt_dir, eval_step=args.eval_step,
                    log_every=args.log_every)

    history_path = ckpt_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2, default=str)
    print(f"  Training history saved: {history_path}")
    print(f"\n  Best loss: {history['best_loss']:.4f} @ epoch {history['best_epoch']}")
    print(f"  Final alpha_l (layer 0): {history['epoch_alpha_l'][-1][0]}")
    print(f"  Final scale_l (layer 0): {history['epoch_scale_l'][-1][0]}")


if __name__ == "__main__":
    main()