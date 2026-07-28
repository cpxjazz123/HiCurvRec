"""Task #144 θ grad diagnostic — verify if Phase B θ.grad is exactly 0
(or just very small) when κ=0 at start.

Per user 2026-07-24 feedback: "如果是精确的0(不是很小的非零数), 说明解冻这个
操作本身可能没有把θ真正接回优化器的更新链路/或 torch.where branch selection
阻断了 θ 的梯度."

Plan:
  1. Load same config as Task #144 Arm B (M=1, kappa_max=2.0, theta_init=0.0)
  2. Run 5 epochs with θ forced unfrozen lr_theta=1e-3 (large to amplify)
  3. Print θ_m.grad at each epoch end → check if exactly 0 or small nonzero
  4. Also test: if we start with θ_init=0.5 (so κ≈0.76, sph branch active),
     does θ.grad become nonzero? → verifies the torch.where bug hypothesis

Output: prints θ_m.data + θ_m.grad at each epoch end.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')

from model.utils import EmbDataset
from model.hrqvae_free_curv import FreeCurvHRQVAE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta_init", type=float, default=0.0)
    parser.add_argument("--lr_theta", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:3")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"\n=== Task #144 θ grad diagnostic ===")
    print(f"theta_init={args.theta_init} (κ_init={2.0 * np.tanh(args.theta_init):.4f})")
    print(f"lr_theta={args.lr_theta} (forced UNFROZEN, NOT Task #144 scheduler)")
    print(f"epochs={args.epochs}, device={args.device}\n")

    data_path = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet"
    data = EmbDataset(data_path)
    print(f"Data: {len(data)} items, dim={data.dim}")

    model = FreeCurvHRQVAE(
        in_dim=data.dim,
        num_emb_list=[64, 128, 256],
        e_dim=32,
        M=1,
        kappa_max=2.0,
        layers=[2048, 1024, 512, data.dim],
        dropout_prob=0.0,
        bn=False,
        loss_type="mse",
        quant_loss_weight=1.0,
        beta=0.25,
        kmeans_init=True,
        kmeans_iters=50,
        sk_eps=[0.0, 0.0, 0.0],
        sk_iters=[0, 0, 0],
    ).to(device)

    # Apply theta_init override
    if args.theta_init != 0.0:
        for vq in model.hrq.vq_layers:
            vq.theta_m.data = torch.full((1,), args.theta_init, dtype=torch.float32, device=device)

    encoder_decoder_params = [p for n, p in model.named_parameters() if 'theta_m' not in n]
    theta_params = [p for n, p in model.named_parameters() if 'theta_m' in n]

    optim = torch.optim.Adam([
        {'params': encoder_decoder_params, 'lr': 1e-4},
        {'params': theta_params, 'lr': args.lr_theta},
    ])

    print(f"\n[Setup] theta_params: {len(theta_params)} tensors, initial lr_theta={args.lr_theta}")
    print(f"[Setup] θ_m init values:")
    for li, vq in enumerate(model.hrq.vq_layers):
        theta = vq.theta_m.detach().cpu().tolist()
        kappa = (vq.kappa_max * torch.tanh(vq.theta_m)).detach().cpu().tolist()
        print(f"  Layer {li}: θ={theta}, κ={kappa}")

    loader = DataLoader(data, num_workers=2, batch_size=256, shuffle=True, pin_memory=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for batch_idx, d in enumerate(loader):
            d = d.to(device)
            optim.zero_grad()
            out, rq_loss, indices = model(d)
            # FreeCurvHRQVAE.forward returns (out, rq_loss, indices), not (out, latents)
            # Use out for MSE reconstruction loss (matches train_HG-Rec.py compute_loss call)
            recon_loss = torch.nn.functional.mse_loss(out, d)
            quant_loss_weight = model.quant_loss_weight if hasattr(model, 'quant_loss_weight') else 1.0
            loss = recon_loss + quant_loss_weight * rq_loss
            loss.backward()

            # Print θ.grad AT FIRST BATCH after backward (before optimizer.step)
            if batch_idx == 0:
                for li, vq in enumerate(model.hrq.vq_layers):
                    theta_grad = vq.theta_m.grad
                    if theta_grad is None:
                        print(f"  [ep{epoch} batch 0 L{li}] θ.grad = None ⚠️ NO GRADIENT FLOW")
                    else:
                        grad_values = theta_grad.detach().cpu().tolist()
                        is_exactly_zero = all(g == 0.0 for g in grad_values)
                        is_very_small = all(abs(g) < 1e-10 for g in grad_values)
                        marker = "EXACTLY 0 ⚠️" if is_exactly_zero else ("very small" if is_very_small else "nonzero")
                        print(f"  [ep{epoch} batch 0 L{li}] θ.grad = {grad_values} [{marker}]")

            optim.step()
            epoch_loss += loss.item()

            if batch_idx >= 20:  # only first 20 batches per epoch for speed
                break

        # End of epoch: print θ value + util
        print(f"\n[ep{epoch}] epoch_loss={epoch_loss:.4f}")
        for li, vq in enumerate(model.hrq.vq_layers):
            theta = vq.theta_m.detach().cpu().tolist()
            kappa = (vq.kappa_max * torch.tanh(vq.theta_m)).detach().cpu().tolist()
            print(f"  Layer {li}: θ={theta}, κ={kappa}")


if __name__ == "__main__":
    main()