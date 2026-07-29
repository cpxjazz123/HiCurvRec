"""
Task #276 Stage 2 — A2 curriculum SID inference (Sinkhorn + dedup)
2026-07-29

输入: A2_extend_ep50 best_collision_model.pth (L0=89.1%, collision=0.1532)
       或任何 Task #270 系列 ckpt (A1/A2/A3)
输出: products/task276/stage2/A2_t5_hrqvae_poincare.npy (9922, 4) int array

逻辑:
  1. 读 ckpt['args'] 自动推断 model kwargs (product_manifold, angular_dim, radial_dim, etc.)
  2. instantiate HRQVAE 跟 train_hrqvae.py 一致
  3. load_state_dict (strict=True — 跟 train ckpt 1:1 对应)
  4. get_indices + Sinkhorn 30 iter + dedup (跟 task84 同样的 process)
  5. save (9922, 4) int .npy

R12: 读 best_collision (低 collision SID) 而不是 best_loss
R11.3: stage2 30 iter Sinkhorn + 4th-digit dedup (跟 task84 一致)
"""

import os
import sys
import json
import glob
import argparse
import collections
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO = "/home/wlia0047/ar57/wenyu/GeneRec"
sys.path.insert(0, os.path.join(REPO, "HG-Rec"))

from model.hrqvae import HRQVAE  # 跟 task84 一样
from model.utils import EmbDataset  # EmbDataset 实际在 model/utils.py

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--data_path", default=f"{REPO}/HG-Rec/dataset/Instruments/item_emb.parquet")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--sk_max_iters", type=int, default=30, help="Sinkhorn max iters")
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"[Task #276] ckpt_path={args.ckpt_path}")
    print(f"[Task #276] output_path={args.output_path}")
    print(f"[Task #276] data_path={args.data_path}")

    # 1. Load ckpt + auto-detect model kwargs
    ckpt = torch.load(args.ckpt_path, weights_only=False, map_location=torch.device("cpu"))
    cargs = ckpt["args"]
    state_dict = ckpt["state_dict"]

    print(f"[Task #276] ckpt args: product_manifold={cargs.product_manifold}, angular_dim={cargs.angular_dim}, radial_dim={cargs.radial_dim}, beta={cargs.beta}, num_emb_list={cargs.num_emb_list}, e_dim={cargs.e_dim}, layers={cargs.layers}, bn={getattr(cargs,'bn',False)}")

    # 2. Load data
    data = EmbDataset(args.data_path)

    # 3. Instantiate HRQVAE (跟 train_hrqvae.py 完整 kwargs 一致)
    # assignment_mode_list 在 train 时是 str (comma-separated), 需要 parse 成 List[str]
    aml = getattr(cargs, "assignment_mode_list", None)
    if aml is None:
        aml_list = None
    elif isinstance(aml, str):
        aml_list = [s.strip() for s in aml.split(",")]
    else:
        aml_list = aml

    model = HRQVAE(
        in_dim=data.dim,
        num_emb_list=cargs.num_emb_list,
        e_dim=cargs.e_dim,
        layers=cargs.layers,
        dropout_prob=getattr(cargs, "dropout_prob", 0.0),
        bn=getattr(cargs, "bn", False),
        loss_type=cargs.loss_type,
        quant_loss_weight=getattr(cargs, "quant_loss_weight", 1.0),
        beta=cargs.beta,
        kmeans_init=cargs.kmeans_init,
        kmeans_iters=cargs.kmeans_iters,
        sk_eps=cargs.sk_epsilons,
        sk_iters=cargs.sk_iters,
        product_manifold=getattr(cargs, "product_manifold", False),
        angular_dim=getattr(cargs, "angular_dim", 4),
        radial_dim=getattr(cargs, "radial_dim", 32),
        assignment_mode_list=aml_list,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"[Task #276] model loaded successfully")
    print(model)

    # 4. Get all indices (no Sinkhorn yet)
    data_loader = DataLoader(
        data,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
    )

    all_indices = []
    all_indices_str = []
    prefix = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>","<e_{}>"]

    for d in tqdm(data_loader, desc="get_indices"):
        d = d.to(device)
        indices = model.get_indices(d, use_sk=False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for index in indices:
            code = []
            for i, ind in enumerate(index):
                code.append(prefix[i].format(int(ind)))
            all_indices.append(code)
            all_indices_str.append(str(code))

    all_indices = np.array(all_indices)
    all_indices_str = np.array(all_indices_str)

    print(f"[Task #276] initial unique SID: {len(set(all_indices_str.tolist()))}/{len(all_indices_str)}")

    # 5. Sinkhorn iterative collision resolve (跟 task84 一样, 最多 30 iter)
    tt = 0
    while True:
        if tt >= args.sk_max_iters or check_collision(all_indices_str):
            break

        collision_item_groups = get_collision_item(all_indices_str)
        print(f"[Task #276] iter {tt}: {len(collision_item_groups)} collision groups")
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

    print(f"[Task #276] Sinkhorn done after {tt} iters")
    print(f"[Task #276] final unique SID: {len(set(all_indices_str.tolist()))}/{len(all_indices_str)}")

    # 6. Dedup with 4th digit (跟 task84 一致)
    codes = []
    for key, value in enumerate(all_indices.tolist()):
        code = [int(item.split('_')[1].strip('>')) for item in value]
        codes.append(code)

    codes_array = np.array(codes)
    codes_array = np.hstack((codes_array, np.zeros((codes_array.shape[0], 1), dtype=int)))

    unique_codes, counts = np.unique(codes_array, axis=0, return_counts=True)
    duplicates = unique_codes[counts > 1]
    if len(duplicates) > 0:
        print(f"[Task #276] Resolving {len(duplicates)} duplicate groups via 4th-digit increment")
        for duplicate in duplicates:
            duplicate_indices = np.where((codes_array == duplicate).all(axis=1))[0]
            for i, idx in enumerate(duplicate_indices):
                codes_array[idx, -1] = i

    new_unique_codes, new_counts = np.unique(codes_array, axis=0, return_counts=True)
    duplicates = new_unique_codes[new_counts > 1]
    if len(duplicates) > 0:
        print(f"[Task #276] WARN: {len(duplicates)} duplicate groups remain after dedup")
    else:
        print(f"[Task #276] All SIDs unique after dedup")

    # 7. Save
    print(f"[Task #276] saving codes to {args.output_path}")
    print(f"[Task #276] first 5 codes: {codes_array[:5]}")
    print(f"[Task #276] shape: {codes_array.shape}, dtype: {codes_array.dtype}")
    np.save(args.output_path, codes_array)
    print(f"[Task #276] DONE")


if __name__ == "__main__":
    main()