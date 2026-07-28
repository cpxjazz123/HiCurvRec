"""Task #179 Stage 2 fork — 纯欧式 RQ-VAE Sinkhorn codebook inference.

Mirror of scripts/task178_hgrec_fixed_stage2_codebook.py but with EuclideanHRQVAE.
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')

from model.utils import EmbDataset
from model.hrqvae_euclidean import EuclideanHRQVAE


def check_collision(all_indices_str):
    return len(all_indices_str) == len(set(all_indices_str.tolist()))


def get_indices_count(all_indices_str):
    return collections.Counter(all_indices_str.tolist())


def get_collision_item(all_indices_str):
    idx2ids = collections.defaultdict(list)
    for i, s in enumerate(all_indices_str):
        idx2ids[s].append(i)
    return [ids for s, ids in idx2ids.items() if len(ids) > 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--max_sinkhorn_iters", type=int, default=30)
    parser.add_argument("--num_emb_list", type=int, nargs='+', default=[32, 64, 256])
    parser.add_argument("--e_dim", type=int, default=32)
    parser.add_argument("--layers", type=int, nargs='+', default=[512, 256, 128, 64])
    parser.add_argument("--loss_type", type=str, default="mse")
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--sk_epsilons", type=float, nargs='+', default=[0.003, 0.003, 0.003])
    parser.add_argument("--sk_iters", type=int, default=50)
    parser.add_argument("--kmeans_init", type=bool, default=True)
    parser.add_argument("--kmeans_iters", type=int, default=1000)
    parser.add_argument("--bn", type=bool, default=True)
    parser.add_argument("--data_path", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet")
    args = parser.parse_args()

    device = torch.device(args.device)
    log = logging.getLogger("task179_stage2")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    log.info(f"Loading ckpt: {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, weights_only=False, map_location=torch.device('cpu'))
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    data = EmbDataset(args.data_path)
    log.info(f"  Data: {len(data)} items, dim={data.dim}")

    model = EuclideanHRQVAE(
        in_dim=data.dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=0.0,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=1.0,
        beta=args.beta,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_eps=args.sk_epsilons,
        sk_iters=args.sk_iters,
    )
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    data_loader = DataLoader(data, num_workers=2, batch_size=64, shuffle=True, pin_memory=True)

    all_indices = []
    all_indices_str = []
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]
    n_layers = len(args.num_emb_list)

    from tqdm import tqdm
    for d in tqdm(data_loader, desc="Task #179 Stage 2 forward"):
        d = d.to(device)
        indices = model.get_indices(d, use_sk=False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for index in indices:
            code = [prefix[i].format(int(ind)) for i, ind in enumerate(index)]
            all_indices.append(code)
            all_indices_str.append(str(code))

    all_indices = np.array(all_indices)
    all_indices_str = np.array(all_indices_str)

    tt = 0
    while tt < args.max_sinkhorn_iters and not check_collision(all_indices_str):
        collision_groups = get_collision_item(all_indices_str)
        log.info(f"  Sinkhorn iter {tt}: {len(collision_groups)} collision groups")
        for collision_items in collision_groups:
            d = data[collision_items].to(device)
            indices = model.get_indices(d, use_sk=True)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for idx, item in zip(indices, collision_items):
                code = [prefix[i].format(int(ind)) for i, ind in enumerate(idx)]
                all_indices[item] = code
                all_indices_str[item] = str(code)
        tt += 1

    log.info(f"  Total: {len(all_indices_str)}, unique: {len(set(all_indices_str.tolist()))}")
    counter = get_indices_count(all_indices_str)
    util_per_layer = [0] * n_layers
    for s in counter:
        codes = eval(s)
        for li, ind in enumerate(codes):
            ind_id = int(ind.split('_')[1].rstrip('>'))
            util_per_layer[li] = max(util_per_layer[li], ind_id + 1)
    log.info(f"  Per-layer codebook usage: {util_per_layer} / {args.num_emb_list}")

    all_indices_dict = {item: list(idx) for item, idx in enumerate(all_indices.tolist())}
    codes = []
    for value in all_indices_dict.values():
        code = [int(t.split('_')[1].strip('>')) for t in value]
        codes.append(code)
    codes_array = np.array(codes)

    unique_codes, counts = np.unique(codes_array, axis=0, return_counts=True)
    duplicates = unique_codes[counts > 1]
    if len(duplicates) > 0:
        log.info(f"  Resolving {len(duplicates)} duplicates with 4th-digit increment (dedup)")
        if codes_array.shape[1] == 3:
            dedup_col = np.zeros((codes_array.shape[0], 1), dtype=codes_array.dtype)
            codes_array = np.hstack([codes_array, dedup_col])
        for duplicate in duplicates:
            duplicate_indices = np.where((codes_array[:, :3] == duplicate).all(axis=1))[0]
            for i, idx in enumerate(duplicate_indices):
                codes_array[idx, -1] = i
    else:
        if codes_array.shape[1] == 3:
            dedup_col = np.zeros((codes_array.shape[0], 1), dtype=codes_array.dtype)
            codes_array = np.hstack([codes_array, dedup_col])

    new_unique_codes, _ = np.unique(codes_array, axis=0, return_counts=True)
    final_dups = new_unique_codes[np.unique(codes_array, axis=0, return_counts=True)[1] > 1]
    log.info(f"  POST-resolve: {len(final_dups)} duplicates remain (expected 0)")

    n_total_cols = codes_array.shape[1]
    for li in range(n_total_cols):
        codes_li = codes_array[:, li]
        unique_codes_li = np.unique(codes_li)
        cap = args.num_emb_list[li] if li < n_layers else "∞(dedup)"
        log.info(f"  Layer {li} utilization: {len(unique_codes_li)} / {cap}")

    log.info(f"  first 5 codes (int): {codes_array[:5].tolist()}")
    log.info(f"  codes shape: {codes_array.shape}")
    np.save(args.output_path, codes_array)
    log.info(f"  Saved: {args.output_path}")

    diag_path = args.output_path.replace(".npy", "_diagnostic.json")
    diag = {
        "ckpt_path": args.ckpt_path,
        "task": "task179_pure_euclidean",
        "n_items": int(len(all_indices_str)),
        "n_unique_sid": int(len(np.unique(codes_array, axis=0))),
        "per_layer_utilization": {
            f"layer_{li}": {
                "unique": int(len(np.unique(codes_array[:, li]))),
                "total_capacity": int(args.num_emb_list[li]) if li < n_layers else -1,
                "fraction": float(len(np.unique(codes_array[:, li])) / (args.num_emb_list[li] if li < n_layers else 1)),
            }
            for li in range(n_total_cols)
        },
        "sinkhorn_iters": tt,
        "recipe": {
            "num_emb_list": args.num_emb_list,
            "e_dim": args.e_dim,
            "beta": args.beta,
            "loss_type": args.loss_type,
            "sk_epsilons": args.sk_epsilons,
            "euclidean_only": True,
        },
    }
    with open(diag_path, 'w') as f:
        json.dump(diag, f, indent=2)
    log.info(f"  Saved diagnostic: {diag_path}")


if __name__ == '__main__':
    main()