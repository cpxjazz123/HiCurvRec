"""方向 F Stage 2: M-arm v11/v12 ep4 ckpt → (9922, 4) SID .npy for Stage 3 T5.

HRQVAE-based (用 HRQVAE.from_args 重建), 不是 RQVAE.
Pattern 抄 task144_stage2_codebook.py 但 model 换 HRQVAE.

Usage:
    python3 scripts/m_arm_step3_stage2_codebook.py --ckpt_path <ep4 ckpt> --out_name <name>
"""
import argparse
import collections
import glob
import logging
import os
import sys
from inspect import signature
from time import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')

from model.utils import EmbDataset
from model.hrqvae import HRQVAE


def check_collision(all_indices_str):
    return len(all_indices_str) == len(set(all_indices_str.tolist()))


def get_indices_count(all_indices_str):
    return collections.Counter(all_indices_str.tolist())


def get_collision_item(all_indices_str):
    idx2ids = collections.defaultdict(list)
    for i, s in enumerate(all_indices_str):
        idx2ids[s].append(i)
    return [ids for s, ids in idx2ids.items() if len(ids) > 1]


def build_model(ckpt):
    sig = signature(HRQVAE.__init__)
    args_d = vars(ckpt['args'])
    mkw = {'in_dim': 768, 'num_emb_list': [64, 128, 256], 'e_dim': 34}
    for k, v in args_d.items():
        nk = k.replace('-', '_')
        if nk in sig.parameters:
            mkw[nk] = v
    if 'norm_target' in args_d and 'r_target_norm_list' in sig.parameters:
        mkw['r_target_norm_list'] = list(args_d['norm_target'])
    for k in ['r_target_list']:
        v = mkw.get(k)
        if isinstance(v, str):
            mkw[k] = [float(x) for x in v.split(',')]
    return HRQVAE(**mkw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--out_name", type=str, required=True,
                        help="output .npy name (relative to /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_sinkhorn_iters", type=int, default=30)
    args = parser.parse_args()

    device = torch.device(args.device)
    log = logging.getLogger(f"m_arm_stage2")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    ckpt_path = args.ckpt_path
    log.info(f"Loading ckpt: {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=torch.device('cpu'))
    ckpt_args = ckpt['args']
    state_dict = ckpt['state_dict']
    log.info(f"  num_emb_list={ckpt_args.num_emb_list}, e_dim={ckpt_args.e_dim}")
    log.info(f"  product_manifold={ckpt_args.product_manifold}, angular_dim={ckpt_args.angular_dim}, radial_dim={ckpt_args.radial_dim}")
    log.info(f"  best_loss={ckpt.get('best_loss', '?')}, epoch={ckpt.get('epoch', '?')}, best_collision_rate={ckpt.get('best_collision_rate', '?')}")

    data = EmbDataset(ckpt_args.data_path)
    log.info(f"  Data: {len(data)} items, dim={data.dim}")

    t0 = time()
    model = build_model(ckpt)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    log.info(f"  Model loaded in {time()-t0:.1f}s")

    data_loader = DataLoader(data, num_workers=2, batch_size=64, shuffle=False, pin_memory=True)

    all_indices = []
    all_indices_str = []
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]

    from tqdm import tqdm
    for d in tqdm(data_loader, desc="Stage 2 forward (use_sk=False)"):
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

    # Collision before Sinkhorn (trainer-reported) — likely matches ckpt name
    n_unique = len(set(all_indices_str.tolist()))
    log.info(f"  PRE-Sinkhorn: items={len(all_indices_str)}, unique={n_unique}, collision={1 - n_unique/len(all_indices_str):.4f}")

    # Sinkhorn resolve (max 30 iter)
    tt = 0
    while tt < args.max_sinkhorn_iters and not check_collision(all_indices_str):
        collision_groups = get_collision_item(all_indices_str)
        log.info(f"  Sinkhorn iter {tt}: {len(collision_groups)} collision groups")
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
    log.info(f"  POST-Sinkhorn ({tt} iter): items={tot_item}, unique={tot_indice}, collision={1 - tot_indice/tot_item:.4f}")

    # Decode back to int array, append 4th digit for deduplication
    all_indices_dict = {item: list(idx) for item, idx in enumerate(all_indices.tolist())}
    codes = []
    for value in all_indices_dict.values():
        code = [int(t.split('_')[1].strip('>')) for t in value]
        codes.append(code)

    codes_array = np.array(codes)
    codes_array = np.hstack((codes_array, np.zeros((codes_array.shape[0], 1), dtype=int)))

    unique_codes, counts = np.unique(codes_array, axis=0, return_counts=True)
    duplicates = unique_codes[counts > 1]
    if len(duplicates) > 0:
        log.info(f"  Resolving {len(duplicates)} duplicates with 4th-digit increment")
        for duplicate in duplicates:
            duplicate_indices = np.where((codes_array == duplicate).all(axis=1))[0]
            for i, idx in enumerate(duplicate_indices):
                codes_array[idx, -1] = i

    new_unique_codes, new_counts = np.unique(codes_array, axis=0, return_counts=True)
    final_dups = new_unique_codes[new_counts > 1]
    log.info(f"  POST-4th-digit: {len(final_dups)} duplicates remain (expected 0)")

    # Per-layer utilization diagnostic (only 3 RQ-VAE layers, 4th column is dedup digit)
    n_rq_layers = len(ckpt_args.num_emb_list)
    for li in range(n_rq_layers):
        codes_li = codes_array[:, li]
        unique_codes_li = np.unique(codes_li)
        log.info(f"  Layer {li} utilization: {len(unique_codes_li)} / {ckpt_args.num_emb_list[li]} "
                 f"({100 * len(unique_codes_li) / ckpt_args.num_emb_list[li]:.1f}%)")

    output_path = f"/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/{args.out_name}.npy"
    log.info(f"Saving codes to {output_path}")
    log.info(f"  shape: {codes_array.shape}, first 5: {codes_array[:5].tolist()}")
    np.save(output_path, codes_array)

    # Diagnostic JSON
    diag_path = output_path.replace(".npy", "_diagnostic.json")
    diag = {
        "ckpt_path": ckpt_path,
        "n_items": int(tot_item),
        "n_unique_sid": int(len(np.unique(codes_array, axis=0))),
        "collision_rate_post_sinkhorn": float(1 - tot_indice / tot_item),
        "sinkhorn_iters_used": tt,
        "per_layer_utilization": {
            f"layer_{li}": {
                "unique": int(len(np.unique(codes_array[:, li]))),
                "total_capacity": int(ckpt_args.num_emb_list[li]),
                "fraction": float(len(np.unique(codes_array[:, li])) / ckpt_args.num_emb_list[li]),
            }
            for li in range(n_rq_layers)
        },
    }
    import json
    with open(diag_path, 'w') as f:
        json.dump(diag, f, indent=2)
    log.info(f"  Diagnostic: {diag_path}")


if __name__ == "__main__":
    main()