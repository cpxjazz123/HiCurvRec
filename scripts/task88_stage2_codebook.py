#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #88 — Stage 2 codebook generation (HRQ-VAE ckpt -> SID tensor .npy).

Fork of scripts/task84_hgrec_stage2_codebook.py with parametric ckpt/output paths.
Used for all 6 grid curvature combinations (Task #88 per-layer curvature experiment).

R88 (Task #88): 通用化 Task #84 Stage 2 script.
- ckpt path: --ckpt (default: latest grid-specific best_loss_model.pth)
- output path: --output (default: {dataset}_{suffix}_t5_hrqvae_poincare.npy)
- dataset name: --dataset_name (default Instruments)
- prefix: <a_/b_/c_/d_/e_> (HG-Rec upstream 5 layers, we use first 3)

Output: numpy array shape (N, 3) int64 with codes [a_idx, b_idx, c_idx].
Stage 3 (T5) expects this shape + the matching prefix string format.
"""
from __future__ import annotations

import argparse
import collections
import glob
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.utils import EmbDataset
from model.hrqvae import HRQVAE


def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item == tot_indice


def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count


def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)
    collision_item_group = []
    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_group.append(index2id[index])
    return collision_item_group


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='Path to best_loss_model.pth from Task #88 grid run')
    p.add_argument('--output', required=True, help='Output .npy path (e.g. {dataset}_curv_X_Y_Z_t5_hrqvae_poincare.npy)')
    p.add_argument('--dataset_name', default='Instruments')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--max_collision_iter', type=int, default=30)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"=== Task #88 Stage 2 codebook generation ===")
    print(f"  ckpt: {args.ckpt}")
    print(f"  output: {args.output}")
    print(f"  device: {args.device}")

    ckpt = torch.load(args.ckpt, weights_only=False, map_location=torch.device('cpu'))
    train_args = ckpt['args']
    state_dict = ckpt['state_dict']

    # R88: ensure curvature_list is loaded from ckpt args
    print(f"  ckpt args: num_emb_list={train_args.num_emb_list}, e_dim={train_args.e_dim}, loss_type={train_args.loss_type}")
    print(f"  curvatures={getattr(train_args, 'curvatures', None)}")

    data = EmbDataset(train_args.data_path)

    # R88: pass curvature_list to HRQVAE if present (else default None = upstream behavior)
    curvature_list = getattr(train_args, 'curvatures', None)
    model = HRQVAE(
        in_dim=data.dim,
        num_emb_list=train_args.num_emb_list,
        e_dim=train_args.e_dim,
        layers=train_args.layers,
        dropout_prob=train_args.dropout_prob,
        bn=train_args.bn,
        loss_type=train_args.loss_type,
        quant_loss_weight=train_args.quant_loss_weight,
        beta=train_args.beta,
        kmeans_init=train_args.kmeans_init,
        kmeans_iters=train_args.kmeans_iters,
        sk_eps=train_args.sk_epsilons,
        sk_iters=train_args.sk_iters,
        curvature_list=curvature_list,
    )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    data_loader = DataLoader(
        data, num_workers=train_args.num_workers,
        batch_size=args.batch_size, shuffle=True, pin_memory=True,
    )

    all_indices = []
    all_indices_str = []
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]

    for d in data_loader:
        d = d.to(device)
        indices = model.get_indices(d, use_sk=False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for index in indices:
            code = []
            for i, ind in enumerate(index):
                code.append(prefix[i].format(int(ind)))
            all_indices.append(code)
            all_indices_str.append(str(code))

    all_indices = np.array(all_indices, dtype=object)
    all_indices_str = np.array(all_indices_str)

    # Collision resolution (up to max_collision_iter rounds with use_sk=True)
    tt = 0
    while True:
        if tt >= args.max_collision_iter or check_collision(all_indices_str):
            break
        collision_item_groups = get_collision_item(all_indices_str)
        print(f"  iter {tt}: {len(collision_item_groups)} collision groups")
        for collision_items in collision_item_groups:
            d = data[collision_items].to(device)
            indices = model.get_indices(d, use_sk=True)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for item, index in zip(collision_items, indices):
                code = []
                for i, ind in enumerate(index):
                    code.append(prefix[i].format(int(ind)))
                all_indices[item] = code
                all_indices_str[item] = str(code)
        tt += 1

    print(f"  All indices number: {len(all_indices)}")
    print(f"  Max number of conflicts: {max(get_indices_count(all_indices_str).values())}")
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    print(f"  Collision Rate: {(tot_item - tot_indice) / tot_item:.6f}")

    # Convert prefix strings to int codes (3 layer codebook)
    codes = []
    for key, value in enumerate(all_indices.tolist()):
        # value is list of strings like ["<a_42>", "<b_17>", "<c_123>"]
        code = [int(item.split('_')[1].strip('>')) for item in value]
        codes.append(code)
    codes_array = np.array(codes, dtype=np.int64)

    # R88: Add 4th column (placeholder for codebook_size[3]=1 EOS-like token)
    # (Task #84 Stage 2 fork convention, required by downstream item_to_code)
    codes_array = np.hstack((codes_array, np.zeros((codes_array.shape[0], 1), dtype=np.int64)))

    # R88: Resolve duplicate codes by incrementing last dimension (Task #84 fork logic)
    unique_codes, counts = np.unique(codes_array, axis=0, return_counts=True)
    duplicates = unique_codes[counts > 1]
    if len(duplicates) > 0:
        print(f"  Resolving {len(duplicates)} duplicate code groups...")
        for duplicate in duplicates:
            duplicate_indices = np.where((codes_array == duplicate).all(axis=1))[0]
            for i, idx in enumerate(duplicate_indices):
                codes_array[idx, -1] = i  # increment last digit

    new_unique_codes, new_counts = np.unique(codes_array, axis=0, return_counts=True)
    new_duplicates = new_unique_codes[new_counts > 1]
    if len(new_duplicates) > 0:
        print(f"  ⚠️ Still {len(new_duplicates)} duplicates after resolution (codebook_size[3]=1 limit)")
    else:
        print(f"  ✅ All codes unique after resolution.")

    print(f"  codes.shape: {codes_array.shape}, dtype: {codes_array.dtype}")

    # Save (R88: now saving codes_array with 4th col + dedup)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.save(args.output, codes_array)
    print(f"  Saved to: {args.output}")

    # R88: write metadata JSON
    meta_path = args.output.replace('.npy', '_meta.json')
    import json
    meta = {
        'ckpt': args.ckpt,
        'dataset_name': args.dataset_name,
        'curvatures': curvature_list,
        'num_items': len(codes_array),
        'code_shape': list(codes_array.shape),
        'collision_rate': (tot_item - tot_indice) / tot_item,
        'prefix_format': prefix[:codes_array.shape[1]],
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Meta saved to: {meta_path}")


if __name__ == '__main__':
    main()