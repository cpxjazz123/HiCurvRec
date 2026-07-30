#!/usr/bin/env python3
"""Task #160 — Issue #56 Stage 2 inference.

从 task156_issue56 Stage 1 trained ckpt (FreeCurvHRQVAE + MixedCurv VQ)
生成 (N, 4) SID npy, 跟下游 Stage 3 训练兼容.

Recipe:
  - Load ckpt: products/task156/ckpt/Instruments/best_loss_model.pth
  - Model: FreeCurvHRQVAE with FreeCurvVectorQuantizationMixedCurv layers
  - Input: HG-Rec/dataset/Instruments/item_emb.parquet
  - Output: products/task160/sid/Instruments_t5_hrqvae_issue56.npy (shape (9922, 4))
  - 算法: 跟 baseline 一致 — argmin(d) per layer + 4th digit dedup
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO / "HG-Rec"))

from model.hrqvae_free_curv import FreeCurvHRQVAE
from model.hrqvae_issue55_56 import FreeCurvVectorQuantizationMixedCurv, FreeCurvVectorQuantizationMixedCurvWithScale
from model.utils import EmbDataset

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


def replace_vq_with_mixed_curv(model: FreeCurvHRQVAE, kappa_fixed: float, with_scale: bool = False) -> None:
    """替换 hrq.vq_layers 为 FreeCurvVectorQuantizationMixedCurv (Issue #56)."""
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
        if with_scale:
            vq = FreeCurvVectorQuantizationMixedCurvWithScale(
                n_e=n_e, e_dim=e_dim, M=M,
                kappa_max=kappa_fixed, beta=beta,
                kmeans_init=kmeans_init, kmeans_iters=kmeans_iters,
                sk_eps=sk_eps, sk_iters=sk_iters,
                kappa_fixed=kappa_fixed, scale_init=1.0,
            )
        else:
            vq = FreeCurvVectorQuantizationMixedCurv(
                n_e=n_e, e_dim=e_dim, M=M,
                kappa_max=kappa_fixed, beta=beta,
                kmeans_init=kmeans_init, kmeans_iters=kmeans_iters,
                sk_eps=sk_eps, sk_iters=sk_iters,
                kappa_fixed=kappa_fixed,
            )
        new_layers.append(vq)

    model.hrq.vq_layers = torch.nn.ModuleList(new_layers)


def infer_sid(model: FreeCurvHRQVAE, emb: torch.Tensor, device: torch.device,
              batch_size: int = 1024) -> np.ndarray:
    """推理 SID per item.

    Returns: (N, num_layers) int array
    """
    model.eval()
    N = emb.shape[0]
    num_layers = len(model.hrq.vq_layers)
    all_indices = torch.zeros(N, num_layers, dtype=torch.long, device=device)

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            x = emb[start:end].to(device)
            # encode
            z = model.encoder(x)
            # residual VQ
            residual = z
            for li, quantizer in enumerate(model.hrq.vq_layers):
                # Quantize per layer
                latent = residual.view(-1, model.e_dim)
                codebook = quantizer.embeddings.weight  # (K, e_dim)
                # Per-component dist
                d = quantizer._per_component_dist_sq(latent, codebook)  # (B, K)
                indices = torch.argmin(d, dim=-1)
                all_indices[start:end, li] = indices
                # Update residual
                x_q_hard = codebook.index_select(0, indices)
                residual = residual - x_q_hard
            if (start // batch_size) % 20 == 0:
                print(f"    [Stage 2] {end}/{N} items done", flush=True)

    return all_indices.cpu().numpy()


def dedup_4th_digit(sid: np.ndarray) -> np.ndarray:
    """Add 4th dedup digit: distinguishes same-code duplicates.

    sid[:, 3] = cumulative count of occurrences of (sid[:, 0], sid[:, 1], sid[:, 2]) up to and including this row.
    """
    N = sid.shape[0]
    seen = {}
    out = sid.copy()
    for i in range(N):
        key = tuple(sid[i, :3].tolist())
        seen[key] = seen.get(key, 0) + 1
        out[i, 3] = seen[key] - 1  # 0-indexed dedup counter
    return out


def main():
    parser = argparse.ArgumentParser(description="Task #160 Issue #56 Stage 2 inference")
    parser.add_argument("--ckpt_path", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/products/task156/ckpt/Instruments/best_loss_model.pth")
    parser.add_argument("--data_path", type=str, default=str(EMB_PATH_DEFAULT))
    parser.add_argument("--output_path", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/products/task160/sid/Instruments_t5_hrqvae_issue56.npy")
    parser.add_argument("--kappa_fixed", type=float, default=0.74)
    parser.add_argument("--num_emb_list", type=int, nargs='+', default=[64, 128, 256])
    parser.add_argument("--e_dim", type=int, default=32)
    parser.add_argument("--M", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--sk_eps", type=float, default=0.003)
    parser.add_argument("--sk_iters", type=int, default=3)
    parser.add_argument("--layers", type=int, nargs='+', default=[512, 256, 128, 64])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--with_scale", action="store_true",
                        help="Use FreeCurvVectorQuantizationMixedCurvWithScale (Issue #55 + #56)")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"  Device: {device}")
    print(f"  Ckpt: {args.ckpt_path}")
    print(f"  Output: {args.output_path}")

    # Load embeddings
    emb = load_item_embeddings(Path(args.data_path))
    in_dim = emb.shape[1]

    # Build FreeCurvHRQVAE
    model = FreeCurvHRQVAE(
        in_dim=in_dim, num_emb_list=args.num_emb_list, e_dim=args.e_dim,
        M=args.M, kappa_max=args.kappa_fixed,
        layers=args.layers, dropout_prob=0.0, bn=False,
        loss_type="mse", quant_loss_weight=1.0, beta=args.beta,
        kmeans_init=True, kmeans_iters=10,
        sk_eps=args.sk_eps, sk_iters=args.sk_iters,
    ).to(device)

    # Replace VQ layers with mixed_curv (or with scale)
    replace_vq_with_mixed_curv(model, kappa_fixed=args.kappa_fixed, with_scale=args.with_scale)
    model = model.to(device)
    print(f"  Replaced VQ layers with {'MixedCurvWithScale' if args.with_scale else 'MixedCurv'}")

    # Load ckpt
    if not Path(args.ckpt_path).exists():
        raise FileNotFoundError(f"Ckpt not found: {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    print(f"  Loaded ckpt: epoch={ckpt.get('epoch', '?')}, loss={ckpt.get('loss', '?')}")
    if "alpha_l" in ckpt:
        print(f"  alpha_l (per layer): {ckpt['alpha_l']}")
    if "scale_l" in ckpt:
        print(f"  scale_l (per layer): {ckpt['scale_l']}")

    # Inference
    print(f"\n  Starting Stage 2 inference...")
    t0 = time.time()
    sid = infer_sid(model, emb, device, batch_size=args.batch_size)
    print(f"  Stage 2 inference done in {time.time() - t0:.1f}s. SID shape: {sid.shape}")

    # 4th digit dedup
    sid_dedup = dedup_4th_digit(sid)
    print(f"  After 4th-digit dedup: shape={sid_dedup.shape}")
    print(f"  Per-layer stats: max={sid_dedup.max(axis=0).tolist()}, min={sid_dedup.min(axis=0).tolist()}")

    # Save
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, sid_dedup)
    print(f"  Saved SID to {output_path}")

    # Quick collision check
    unique_3 = len(set([tuple(sid[i, :3].tolist()) for i in range(sid.shape[0])]))
    unique_4 = len(set([tuple(sid_dedup[i].tolist()) for i in range(sid_dedup.shape[0])]))
    print(f"  Unique 3-digit SIDs: {unique_3} / {sid.shape[0]} ({100*unique_3/sid.shape[0]:.2f}%)")
    print(f"  Unique 4-digit SIDs: {unique_4} / {sid_dedup.shape[0]} ({100*unique_4/sid_dedup.shape[0]:.2f}%)")


if __name__ == "__main__":
    main()