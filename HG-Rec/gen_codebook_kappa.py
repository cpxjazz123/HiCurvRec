"""Generate SID using HRQVAE_Kappa ckpt — for Stage 2 κ learning → Stage 3 transfer.

用法:
  python3 gen_codebook_kappa.py --ckpt <best_collision_model.pth> --output <.npy>

约束:
  - 不改 train/valid/test 数据 (R40)
  - 输出格式与 baseline gen_codebook 一致: (N, 4) int64 array, 第 4 列=PAD=0
  - 处理碰撞 (collision) 与 baseline 一致: 重复 get_indices 直到无碰撞 或 重试 30 次
"""
import argparse
import collections
import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from model.hrqvae_kappa import HRQVAE_Kappa, EmbDataset


def check_collision(all_indices_str):
    return len(all_indices_str) == len(set(all_indices_str.tolist()))


def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)
    return [v for v in index2id.values() if len(v) > 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_retry", type=int, default=30)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[SID] loading ckpt: {args.ckpt}")
    ckpt = torch.load(args.ckpt, weights_only=False, map_location="cpu")
    saved_args = ckpt["args"]
    print(f"[SID] ckpt epoch: {ckpt.get('epoch')}, best_collision: {ckpt.get('best_collision_rate')}")

    data = EmbDataset(saved_args["data_path"])
    print(f"[SID] dataset: {saved_args['data_path']} ({len(data)} items, dim={data.dim})")

    model = HRQVAE_Kappa(
        in_dim=data.dim,
        num_emb_list=saved_args["num_emb_list"],
        e_dim=saved_args["e_dim"],
        layers=saved_args["layers"],
        dropout_prob=saved_args["dropout_prob"],
        bn=saved_args["bn"],
        loss_type=saved_args["loss_type"],
        quant_loss_weight=saved_args["quant_loss_weight"],
        beta=saved_args["beta"],
        kmeans_init=saved_args["kmeans_init"],
        kmeans_iters=saved_args["kmeans_iters"],
        sk_eps=saved_args["sk_epsilons"],
        sk_iters=saved_args["sk_iters"],
        learnable_c=saved_args.get("learnable_c", False),
        c_init=saved_args.get("c_init", 1.0),
        c_min=saved_args.get("c_min", 0.05),
        c_max=saved_args.get("c_max", 20.0),
    )
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model = model.to(device)
    model.eval()

    # 显示学到的曲率
    curvs = [q.get_c().item() for q in model.hrq.vq_layers]
    print(f"[SID] learned κ: L0={curvs[0]:.4f}, L1={curvs[1]:.4f}, L2={curvs[2]:.4f}")

    data_loader = DataLoader(
        data, num_workers=4, batch_size=args.batch_size, shuffle=True, pin_memory=True
    )

    all_indices = []
    all_indices_str = []
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]

    for d in data_loader:
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

    tt = 0
    while True:
        if tt >= args.max_retry or check_collision(all_indices_str):
            break
        collision_groups = get_collision_item(all_indices_str)
        print(f"[SID] retry {tt}: {len(collision_groups)} collisions")
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

    print(f"[SID] All indices: {len(all_indices)}")
    counts = collections.defaultdict(int)
    for s in all_indices_str:
        counts[s] += 1
    print(f"[SID] Max conflicts: {max(counts.values())}")

    tot_item = len(all_indices_str)
    tot_unique = len(set(all_indices_str.tolist()))
    print(f"[SID] Collision Rate: {(tot_item - tot_unique) / tot_item:.4f}")

    all_indices_dict = {item: list(map(int, idx)) for item, indices in enumerate(all_indices.tolist())
                         for idx in [indices]}

    codes = []
    for item in range(len(all_indices_dict)):
        value = all_indices_dict[item]
        code = [int(s.split('_')[1].strip('>')) for s in
                [prefix[i].format(v) for i, v in enumerate(value)]]
        codes.append(code)

    codes_array = np.array(codes)
    codes_array = np.hstack((codes_array, np.zeros((codes_array.shape[0], 1), dtype=int)))

    unique_codes, code_counts = np.unique(codes_array, axis=0, return_counts=True)
    duplicates = unique_codes[code_counts > 1]

    if len(duplicates) > 0:
        print(f"[SID] Resolving {len(duplicates)} duplicates...")
        for duplicate in duplicates:
            dup_indices = np.where((codes_array == duplicate).all(axis=1))[0]
            for i, idx in enumerate(dup_indices):
                codes_array[idx, -1] = i

    new_unique_codes, new_counts = np.unique(codes_array, axis=0, return_counts=True)
    duplicates = new_unique_codes[new_counts > 1]
    if len(duplicates) > 0:
        print(f"[SID] WARNING: still {len(duplicates)} duplicates after resolution!")
    else:
        print(f"[SID] No duplicates after resolution.")

    print(f"[SID] Saving to {args.output}")
    print(f"[SID] First 5 codes: {codes_array[:5]}")
    np.save(args.output, codes_array)
    print(f"[SID] Done. Shape: {codes_array.shape}, dtype: {codes_array.dtype}")


if __name__ == "__main__":
    main()
