#!/usr/bin/env python3
"""
Task #73 — Extract SID tensor from a trained ETEGRec RQ-VAE checkpoint.

Loads the model + runs get_indices on the embedding → saves (N, num_codebooks) int64 tensor.

Usage:
    python task73_rqvae_extract_sid.py \\
        --ckpt_path /path/to/epoch_4999_collision_X_model.pth \\
        --data_path /path/to/sentence-t5-768d.npy \\
        --output_path /path/to/sid_tensor.pt
"""
import argparse
import os
import sys
import torch
import numpy as np

# Add ETEGRec RQVAE to path
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/external/ETEGRec/RQVAE")
from models.rqvae import RQVAE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading checkpoint: {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    model_args = ckpt["args"]
    print(f"  epoch={ckpt.get('epoch', '?')}, collision={ckpt.get('best_collision_rate', '?')}")
    print(f"  model args: num_emb={model_args.num_emb_list}, e_dim={model_args.e_dim}, layers={model_args.layers}")

    print(f"\nLoading data: {args.data_path}")
    data = np.load(args.data_path)
    print(f"  data shape: {data.shape}, dtype: {data.dtype}")

    print(f"\nBuilding RQVAE model...")
    model = RQVAE(
        args=model_args,
        in_dim=data.shape[1],
    )
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model = model.to(device)
    model.eval()

    # Run inference in batches
    print(f"\nExtracting SID on {device}...")
    x = torch.from_numpy(data).float().to(device)
    batch_size = 4096
    all_indices = []
    for i in range(0, len(x), batch_size):
        batch = x[i:i + batch_size]
        with torch.no_grad():
            indices = model.get_indices(batch)
        all_indices.append(indices.cpu())
        if i % (batch_size * 10) == 0:
            print(f"  Processed {i + len(batch)}/{len(x)} items")

    all_indices = torch.cat(all_indices, dim=0)
    print(f"\nSID tensor shape: {all_indices.shape}, dtype: {all_indices.dtype}")
    print(f"  per-layer unique codes: {[len(torch.unique(all_indices[:, i])) for i in range(all_indices.shape[1])]}")

    # Save
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(all_indices, args.output_path)
    print(f"\nSaved to {args.output_path}")

    # Sanity: conflict rate
    strs = ["-".join(map(str, row.tolist())) for row in all_indices]
    n_unique = len(set(strs))
    print(f"Conflict rate: {(len(strs) - n_unique) / len(strs):.4f} ({len(strs)} items, {n_unique} unique)")


if __name__ == "__main__":
    main()
