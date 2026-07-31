#!/usr/bin/env python3
"""Task #354 — Issue #63 Stage 1 训练 (vanilla FreeCurvHRQVAE, GPU 0).

Issue #63 Gate 1 spec:
- 三层 κ 原位曲率感知同步重校准 (vanilla FreeCurvHRQVAE, no MixedCurv)
- Recipe 跟 Task #84 baseline + Issue #55/#56 对比:
  - num_emb_list=[64, 128, 256]
  - e_dim=32, M=1, kappa_max=2.0 (learnable per-layer)
  - layers=[512, 256, 128]
  - sk_eps=[0, 0, 0] (OFF per Gate 1 — Sinkhorn 仅 Stage 2 推断时用)
  - beta=0.25
  - epochs=200 (Issue #63 spec)
  - batch_size=1024, lr=1e-3
  - seed=42

USAGE-KILL @ epoch 30: 任一层 utilization < 90% → KILL.
R12 ckpt 强制保存 (每个 eval_step epoch 保存 best_loss + 删除旧).
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

from model.hrqvae_free_curv import FreeCurvHRQVAE

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


def compute_layer_utilization(model: FreeCurvHRQVAE, dl: DataLoader, device: torch.device) -> dict:
    """Per-layer codebook utilization over full dataset."""
    model.eval()
    layer_codes = {i: [] for i in range(len(model.hrq.vq_layers))}
    with torch.no_grad():
        for batch_idx, (x,) in enumerate(dl):
            x = x.to(device, non_blocking=True)
            indices = model.get_indices(x, use_sk=False)
            for i in range(len(model.hrq.vq_layers)):
                layer_codes[i].append(indices[:, i].cpu())
    utils = {}
    for i, codes_list in layer_codes.items():
        all_codes = torch.cat(codes_list)
        K = model.hrq.vq_layers[i].n_e
        used = len(torch.unique(all_codes))
        utils[f"L{i}_util"] = used / K
        utils[f"L{i}_used"] = used
        utils[f"L{i}_total"] = K
        # collision rate: fraction of samples whose ANY layer collides with another
    # Cross-sample collision: each layer's duplicate ratio
    total_samples = sum(codes_list[0].numel() for codes_list in layer_codes.values()) // len(layer_codes)
    for i in range(len(model.hrq.vq_layers)):
        all_codes = torch.cat(layer_codes[i])
        hist = torch.bincount(all_codes)
        dup = (hist > 1).sum().item()  # codes used more than once
        utils[f"L{i}_collision_rate"] = dup / model.hrq.vq_layers[i].n_e
    return utils


def train(model: FreeCurvHRQVAE, dl: DataLoader, device: torch.device,
          n_epochs: int, lr: float, ckpt_dir: Path, eval_step: int = 5,
          log_every: int = 50, usage_kill_epoch: int = 30,
          usage_kill_threshold: float = 0.9) -> dict:
    """Train FreeCurvHRQVAE with Issue #63 vanilla κ-decouple.

    Returns history + final epoch. If USAGE-KILL triggers, returns early with
    history['killed'] = True and reason.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"epoch_loss": [], "epoch_recon": [], "epoch_quant": [],
               "epoch_kappa_m": [], "epoch_util": [], "epoch_collision": []}

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
            out, rq_loss, indices = model(x, use_sk=False)
            loss, recon_loss = model.compute_loss(out, rq_loss, xs=x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
        kappa_m_per_layer = [vq.kappa_m().detach().cpu().tolist() for vq in model.hrq.vq_layers]

        history["epoch_loss"].append(avg_loss)
        history["epoch_recon"].append(avg_recon)
        history["epoch_quant"].append(avg_quant)
        history["epoch_kappa_m"].append(kappa_m_per_layer)

        elapsed = time.time() - t0
        print(f"  [Epoch {epoch}/{n_epochs}] avg_loss={avg_loss:.4f} "
              f"recon={avg_recon:.4f} quant={avg_quant:.4f} "
              f"kappa_m L0={kappa_m_per_layer[0]} L1={kappa_m_per_layer[1]} L2={kappa_m_per_layer[2]} "
              f"({elapsed:.1f}s)", flush=True)

        # R12: ckpt saving per eval_step epoch
        if epoch % eval_step == 0 or epoch == n_epochs:
            # Compute utilization + collision
            util_info = compute_layer_utilization(model, dl, device)
            history["epoch_util"].append({"epoch": epoch, **util_info})
            utils_str = " ".join(f"L{i}_util={util_info[f'L{i}_util']:.3f}" for i in range(3))
            coll_str = " ".join(f"L{i}_coll={util_info[f'L{i}_collision_rate']:.3f}" for i in range(3))
            print(f"    [Eval @ ep {epoch}] {utils_str} | {coll_str}", flush=True)

            # USAGE-KILL check
            if epoch >= usage_kill_epoch:
                min_util = min(util_info[f"L{i}_util"] for i in range(3))
                if min_util < usage_kill_threshold:
                    print(f"    ❌ USAGE-KILL @ ep {epoch}: min_util={min_util:.3f} < {usage_kill_threshold}", flush=True)
                    history["killed"] = True
                    history["killed_reason"] = f"min_util={min_util:.3f} < {usage_kill_threshold}"
                    history["killed_epoch"] = epoch
                    # Save ckpt before killing
                    final_ckpt_path = ckpt_dir / "killed_model.pth"
                    if final_ckpt_path.exists():
                        final_ckpt_path.unlink()
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": avg_loss,
                    }, final_ckpt_path)
                    print(f"    [Killed ckpt saved] {final_ckpt_path}", flush=True)
                    return history

            # Best-loss ckpt save (R12: delete old first)
            if avg_loss < best_loss:
                if best_ckpt_path.exists():
                    best_ckpt_path.unlink()
                best_loss = avg_loss
                best_epoch = epoch
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "kappa_m": kappa_m_per_layer,
                    "util": util_info,
                }, best_ckpt_path)
                print(f"    [R12 ckpt saved] best_loss={best_loss:.4f} @ epoch {epoch}", flush=True)

    # Final ckpt save
    final_ckpt_path = ckpt_dir / "final_model.pth"
    if final_ckpt_path.exists():
        final_ckpt_path.unlink()
    torch.save({
        "epoch": n_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
        "kappa_m": kappa_m_per_layer,
    }, final_ckpt_path)
    print(f"  [Final ckpt saved] {final_ckpt_path}", flush=True)

    history["best_loss"] = best_loss
    history["best_epoch"] = best_epoch
    history["killed"] = False
    return history


def main():
    parser = argparse.ArgumentParser(description="Task #354 Issue #63 Stage 1 training")
    parser.add_argument("--data_path", type=str, default=str(EMB_PATH_DEFAULT))
    parser.add_argument("--ckpt_dir", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/products/task354/ckpt/Instruments")
    parser.add_argument("--num_emb_list", type=int, nargs='+', default=[64, 128, 256])
    parser.add_argument("--e_dim", type=int, default=32)
    parser.add_argument("--M", type=int, default=1)
    parser.add_argument("--kappa_max", type=float, default=2.0,
                        help="Per-layer learnable κ upper bound (Issue #63 vanilla κ-decouple)")
    parser.add_argument("--layers", type=int, nargs='+', default=[512, 256, 128])
    parser.add_argument("--dropout_prob", type=float, default=0.0)
    parser.add_argument("--bn", type=bool, default=False)
    parser.add_argument("--loss_type", type=str, default="mse")
    parser.add_argument("--quant_loss_weight", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--kmeans_init", type=bool, default=True)
    parser.add_argument("--kmeans_iters", type=int, default=10)
    parser.add_argument("--sk_epsilons", type=float, nargs='+', default=[0.0, 0.0, 0.0],
                        help="OFF per Gate 1 spec — Sinkhorn 仅 Stage 2 推断时用")
    parser.add_argument("--sk_iters", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_step", type=int, default=5)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--usage_kill_epoch", type=int, default=30)
    parser.add_argument("--usage_kill_threshold", type=float, default=0.9)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    print(f"  Device: {device}")

    # Load embeddings
    emb = load_item_embeddings(Path(args.data_path))
    in_dim = emb.shape[1]

    # Build FreeCurvHRQVAE (vanilla κ-decouple)
    model = FreeCurvHRQVAE(
        in_dim=in_dim, num_emb_list=args.num_emb_list, e_dim=args.e_dim,
        M=args.M, kappa_max=args.kappa_max,
        layers=args.layers, dropout_prob=args.dropout_prob, bn=args.bn,
        loss_type=args.loss_type, quant_loss_weight=args.quant_loss_weight,
        beta=args.beta, kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_eps=args.sk_epsilons, sk_iters=args.sk_iters,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: FreeCurvHRQVAE, params={n_params:,}")
    print(f"  num_emb_list={args.num_emb_list}, e_dim={args.e_dim}, kappa_max={args.kappa_max}")
    print(f"  per-layer κ learnable: {[vq.theta_m.requires_grad for vq in model.hrq.vq_layers]}")

    # Data loader
    ds = TensorDataset(emb)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True)
    eval_dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)

    # Train
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Starting training: {args.epochs} epochs, lr={args.lr}, bs={args.batch_size}")
    print(f"  Recipe: num_emb_list={args.num_emb_list}, kappa_max={args.kappa_max}")
    print(f"  Ckpt dir: {ckpt_dir}\n")

    history = train(model, dl, device, n_epochs=args.epochs, lr=args.lr,
                    ckpt_dir=ckpt_dir, eval_step=args.eval_step,
                    log_every=args.log_every,
                    usage_kill_epoch=args.usage_kill_epoch,
                    usage_kill_threshold=args.usage_kill_threshold)

    # Save history
    history_path = ckpt_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2, default=str)
    print(f"  Training history saved: {history_path}")

    if history.get("killed"):
        print(f"\n  ❌ KILLED @ epoch {history['killed_epoch']}: {history['killed_reason']}")
        print(f"     USAGE-KILL 触发. Issue #63 Gate 1 NO-GO.")
    else:
        print(f"\n  ✅ Training completed: {args.epochs} epochs")
        print(f"     Best loss: {history['best_loss']:.4f} @ epoch {history['best_epoch']}")
        print(f"     Final kappa_m: L0={history['epoch_kappa_m'][-1][0]} "
              f"L1={history['epoch_kappa_m'][-1][1]} L2={history['epoch_kappa_m'][-1][2]}")


if __name__ == "__main__":
    main()