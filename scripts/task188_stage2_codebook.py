#!/usr/bin/env python3
"""
Task #188 Phase 2 — Generate 4 codebooks from 4 collision rate tiers.
Usage:
  python3 scripts/task188_stage2_codebook.py --ckpt_path <path> --output_path <path> --device cuda:<0..3>
"""
import argparse
import collections
import json
import logging
import sys
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/HG-Rec")
sys.path.insert(0, os.getcwd())
import numpy as np
import torch
import glob
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.utils import *
from model.hrqvae import HRQVAE
print(f"[task188-stage2] sys.path = {os.getcwd()}")


def check_collision(all_indices_str):
    return len(all_indices_str) == len(set(all_indices_str.tolist()))


def get_indices_count(all_indices_str):
    return collections.Counter(all_indices_str.tolist())


def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        index2id.setdefault(index, []).append(i)
    return [items for items in index2id.values() if len(items) > 1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args_cli = parser.parse_args()

    print(f"==== Task #188 codebook generation ====")
    print(f"ckpt: {args_cli.ckpt_path}")
    print(f"output: {args_cli.output_path}")
    print(f"device: {args_cli.device}")

    ckpt = torch.load(args_cli.ckpt_path, weights_only=False, map_location=torch.device("cpu"))
    train_args = ckpt["args"]
    state_dict = ckpt["state_dict"]

    data = EmbDataset(train_args.data_path)
    device = torch.device(args_cli.device)

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
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    data_loader = DataLoader(
        data,
        num_workers=train_args.num_workers,
        batch_size=64,
        shuffle=True,
        pin_memory=True,
    )

    all_indices = []
    all_indices_str = []
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]

    for d in tqdm(data_loader):
        d = d.to(device)
        indices = model.get_indices(d, use_sk=False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for index in indices:
            code = [prefix[i].format(int(ind)) for i, ind in enumerate(index)]
            all_indices.append(code)
            all_indices_str.append(str(code))

    all_indices = np.array(all_indices)
    all_indices_str = np.array(all_indices_str)

    # Sinkhorn 30 轮 dedup 循环 (跟 Task #181 完全一致)
    tt = 0
    while True:
        if tt >= 30 or check_collision(all_indices_str):
            break
        collision_item_groups = get_collision_item(all_indices_str)
        print(f"  Sinkhorn iter {tt}: {len(collision_item_groups)} collision groups")
        for collision_items in collision_item_groups:
            d = data[collision_items].to(device)
            indices = model.get_indices(d, use_sk=True)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for item, index in zip(collision_items, indices):
                code = [prefix[i].format(int(ind)) for i, ind in enumerate(index)]
                all_indices[item] = code
                all_indices_str[item] = str(code)
        tt += 1

    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    collision_rate = (tot_item - tot_indice) / tot_item
    print(f"Final collision rate: {collision_rate:.6f} (iter={tt})")

    codes = []
    for arr in all_indices:
        codes.append([int(s.split("_")[1].strip(">")) for s in arr])

    codes_array = np.array(codes)
    codes_array = np.hstack((codes_array, np.zeros((codes_array.shape[0], 1), dtype=int)))

    unique_codes, counts = np.unique(codes_array, axis=0, return_counts=True)
    duplicates = unique_codes[counts > 1]
    if len(duplicates) > 0:
        for duplicate in duplicates:
            duplicate_indices = np.where((codes_array == duplicate).all(axis=1))[0]
            for i, idx in enumerate(duplicate_indices):
                codes_array[idx, -1] = i

    new_unique_codes, new_counts = np.unique(codes_array, axis=0, return_counts=True)
    duplicates_after = new_unique_codes[new_counts > 1]
    if len(duplicates_after) > 0:
        print(f"⚠️ Still {len(duplicates_after)} duplicates after 4-digit dedup: {duplicates_after[:3]}")
    else:
        print("✅ All codes unique after 4-digit dedup")

    print(f"Saving to {args_cli.output_path}, shape={codes_array.shape}")
    np.save(args_cli.output_path, codes_array)
    print("Done.")
