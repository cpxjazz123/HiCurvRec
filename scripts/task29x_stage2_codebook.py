#!/usr/bin/env python3
"""Task #290/#291/#292 Stage 2 — Parameterized Sinkhorn inference for FSQ/EMA/Restoration wrappers.

Loads Stage 1 ckpt, instantiates the correct wrapper (FSQ/EMA/Restoration),
runs Sinkhorn up to 30 iterations + 4th-digit dedup, saves (9922, 4) int npy.

Usage:
    python3 scripts/task29x_stage2_codebook.py \
        --task_id 290 --quantizer fsq --ckpt_path <path> --output_path <path>

R12: state_dict 包含 encoder + decoder + hrq wrapper 全套. 这里只用 encoder + quantizer.
"""
import argparse
import collections
import glob
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

# PYTHONPATH for `from model.X` import
HGREC = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec"
sys.path.insert(0, HGREC)

from model.utils import EmbDataset  # EmbDataset 定义在 model/utils.py
from model.hrqvae import HRQVAE


def check_collision(all_indices_str):
    return len(all_indices_str) == len(set(all_indices_str.tolist()))


def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)
    return [items for items in index2id.values() if len(items) > 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, required=True, help="290/291/292")
    parser.add_argument("--quantizer", type=str, required=True,
                        choices=["fsq", "ema", "restoration"])
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--data_path", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--sinkhorn_iters", type=int, default=30)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    print(f"[Task #{args.task_id}] Stage 2 Sinkhorn inference")
    print(f"  quantizer={args.quantizer}")
    print(f"  ckpt={args.ckpt_path}")
    print(f"  output={args.output_path}")

    # Load ckpt
    ckpt = torch.load(args.ckpt_path, weights_only=False, map_location=torch.device("cpu"))
    saved_args = ckpt["args"]
    state_dict = ckpt["state_dict"]
    print(f"  saved_args.num_emb_list={saved_args.num_emb_list} e_dim={saved_args.e_dim}")
    print(f"  state_dict keys sample: {list(state_dict.keys())[:5]} ... {list(state_dict.keys())[-5:]}")

    # Determine M from saved_args
    M = len(saved_args.num_emb_list)

    # Build HRQVAE skeleton (no quantizer wrapping yet)
    data = EmbDataset(args.data_path)
    model = HRQVAE(
        in_dim=data.dim,
        num_emb_list=saved_args.num_emb_list,
        e_dim=saved_args.e_dim,
        layers=saved_args.layers,
        dropout_prob=saved_args.dropout_prob,
        bn=saved_args.bn,
        loss_type=saved_args.loss_type,
        quant_loss_weight=saved_args.quant_loss_weight,
        beta=saved_args.beta,
        kmeans_init=saved_args.kmeans_init,
        kmeans_iters=saved_args.kmeans_iters,
        sk_eps=saved_args.sk_epsilons,
        sk_iters=saved_args.sk_iters,
    )

    # Monkey-patch model.hrq with the right wrapper (mirror train_hrqvae.py post-init)
    if args.quantizer == "fsq":
        from model.fsq_quantizer import FSQWithKappaDecouple
        # FSQ levels: use 4 (32 fours = 4^32 grid, encoder only outputs in small range)
        fsq_levels = [4] * saved_args.e_dim
        wrapper = FSQWithKappaDecouple(
            n_e_list=saved_args.num_emb_list,
            e_dim=saved_args.e_dim,
            M=M,
            kappa_init=0.0,
            levels=fsq_levels,
        )
        model.hrq = wrapper
        print(f"  [FSQ] wrapper installed, levels={fsq_levels}")

    elif args.quantizer == "ema":
        from model.hrqvae_free_curv import FreeCurvResidualVectorQuantization
        from model.ema_codebook_quantizer import EMAQuantizerWrapper
        base_rq = FreeCurvResidualVectorQuantization(
            n_e_list=saved_args.num_emb_list,
            e_dim=saved_args.e_dim,
            M=M,
            kappa_max=2.0,
            sk_eps=saved_args.sk_epsilons,
            beta=0.5,
            kmeans_init=True,
            kmeans_iters=100,
            sk_iters=30,
        )
        wrapper = EMAQuantizerWrapper(base_rq, decay=0.99)
        model.hrq = wrapper
        print(f"  [EMA] wrapper installed")

    elif args.quantizer == "restoration":
        from model.hrqvae_free_curv import FreeCurvResidualVectorQuantization
        from model.restoration_quantizer import RestorationQuantizerWrapper
        base_rq = FreeCurvResidualVectorQuantization(
            n_e_list=saved_args.num_emb_list,
            e_dim=saved_args.e_dim,
            M=M,
            kappa_max=2.0,
            sk_eps=saved_args.sk_epsilons,
            beta=0.5,
            kmeans_init=True,
            kmeans_iters=100,
            sk_iters=30,
        )
        wrapper = RestorationQuantizerWrapper(
            base_rq,
            decay=0.99,
            revive_threshold=1,
            revive_ratio=0.1,
            revive_freq_epochs=5,
        )
        model.hrq = wrapper
        print(f"  [Restoration] wrapper installed")

    # Now load state_dict with strict=False (some keys may not match — wrapper internals)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"  load_state_dict: missing={len(missing_keys)}, unexpected={len(unexpected_keys)}")
    if missing_keys:
        print(f"    missing sample: {missing_keys[:3]}")
    if unexpected_keys:
        print(f"    unexpected sample: {unexpected_keys[:3]}")

    model = model.to(device)
    model.eval()

    data_loader = DataLoader(
        data,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
    )

    all_indices = []
    all_indices_str = []
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]

    from tqdm import tqdm
    for d in tqdm(data_loader, desc=f"Stage 2 [{args.quantizer}]"):
        d = d.to(device)
        with torch.no_grad():
            indices = model.get_indices(d, use_sk=False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for index in indices:
            code = [prefix[i].format(int(ind)) for i, ind in enumerate(index)]
            all_indices.append(code)
            all_indices_str.append(str(code))

    all_indices = np.array(all_indices)
    all_indices_str = np.array(all_indices_str)

    # Sinkhorn loop up to N iters to resolve collisions
    tt = 0
    while not check_collision(all_indices_str) and tt < args.sinkhorn_iters:
        collision_groups = get_collision_item(all_indices_str)
        print(f"  iter {tt}: {len(collision_groups)} collision groups")
        for collision_items in collision_groups:
            d = data[collision_items].to(device)
            with torch.no_grad():
                indices = model.get_indices(d, use_sk=True)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for item, index in zip(collision_items, indices):
                code = [prefix[i].format(int(ind)) for i, ind in enumerate(index)]
                all_indices[item] = code
                all_indices_str[item] = str(code)
        tt += 1

    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    print(f"  Sinkhorn done in {tt} iters. Collision Rate = {(tot_item - tot_indice) / tot_item:.4f}")

    # Add 4th-digit dedup column
    codes_array = np.array(
        [[int(item.split("_")[1].strip(">")) for item in row] for row in all_indices.tolist()]
    )
    codes_array = np.hstack((codes_array, np.zeros((codes_array.shape[0], 1), dtype=int)))
    unique_codes, counts = np.unique(codes_array, axis=0, return_counts=True)
    duplicates = unique_codes[counts > 1]
    if len(duplicates) > 0:
        print(f"  Resolving {len(duplicates)} 4th-digit duplicates...")
        for duplicate in duplicates:
            dup_idx = np.where((codes_array == duplicate).all(axis=1))[0]
            for i, idx in enumerate(dup_idx):
                codes_array[idx, -1] = i

    new_unique, new_counts = np.unique(codes_array, axis=0, return_counts=True)
    if len(new_unique[new_counts > 1]) > 0:
        print(f"  ⚠️ Still {len(new_unique[new_counts > 1])} duplicates after dedup")
    else:
        print(f"  ✅ All {len(codes_array)} codes unique")

    np.save(args.output_path, codes_array)
    print(f"  ✅ Saved codes to {args.output_path} shape={codes_array.shape}")
    print(f"  first 5 codes: {codes_array[:5].tolist()}")


if __name__ == "__main__":
    main()