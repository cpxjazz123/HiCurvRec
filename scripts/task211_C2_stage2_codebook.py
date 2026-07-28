#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #210 Phase B B1 — Stage 2 SID inference fork.

Mirror of scripts/task84_hgrec_stage2_codebook.py with 3 changes:
  - ckpt_path: task210/hrqvae_B1/Jul-26-2026_19-43-26_*/best_collision_model.pth
  - output_path: Instruments/Instruments_t5_rqvae_task210_B1.npy
  - pass product_manifold / angular_dim / radial_dim kwargs to HRQVAE instantiation
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
from model.hrqvae import HRQVAE


def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item == tot_indice


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
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_sinkhorn_iters", type=int, default=30)
    args = parser.parse_args()

    device = torch.device(args.device)
    log = logging.getLogger("task210_B1_stage2")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    log.info(f"Loading ckpt: {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, weights_only=False, map_location=torch.device('cpu'))
    ckpt_args = vars(ckpt['args']) if not isinstance(ckpt['args'], dict) else ckpt['args']
    state_dict = ckpt['state_dict']
    log.info(f"  num_emb_list={ckpt_args['num_emb_list']}, e_dim={ckpt_args['e_dim']}, "
             f"product_manifold={ckpt_args.get('product_manifold', False)}, "
             f"angular_dim={ckpt_args.get('angular_dim', None)}, "
             f"radial_dim={ckpt_args.get('radial_dim', None)}")
    log.info(f"  best_loss={ckpt.get('best_loss', '?')}, epoch={ckpt.get('epoch', '?')}, "
             f"best_collision={ckpt.get('best_collision_rate', '?')}")

    data = EmbDataset(ckpt_args['data_path'])
    log.info(f"  Data: {len(data)} items, dim={data.dim}")

    # HRQVAE instantiation — 传 product_manifold 系列参数
    model = HRQVAE(
        in_dim=data.dim,
        num_emb_list=ckpt_args['num_emb_list'],
        e_dim=ckpt_args['e_dim'],
        layers=ckpt_args['layers'],
        dropout_prob=ckpt_args.get('dropout_prob', 0.0),
        bn=ckpt_args.get('bn', False),
        loss_type=ckpt_args['loss_type'],
        quant_loss_weight=ckpt_args.get('quant_loss_weight', 1.0),
        beta=ckpt_args['beta'],
        kmeans_init=ckpt_args.get('kmeans_init', False),
        kmeans_iters=ckpt_args.get('kmeans_iters', 100),
        sk_eps=ckpt_args['sk_epsilons'],
        sk_iters=ckpt_args['sk_iters'],
        product_manifold=ckpt_args.get('product_manifold', False),
        angular_dim=ckpt_args.get('angular_dim', None),
        radial_dim=ckpt_args.get('radial_dim', None),
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Per-layer κ_m diagnostic (if available)
    try:
        for li, vq in enumerate(model.hrq.vq_layers):
            kappas = vq.kappa_m().detach().cpu().tolist()
            log.info(f"  Layer {li} κ_m = [{', '.join(f'{k:+.4f}' for k in kappas)}]")
    except Exception as e:
        log.info(f"  κ_m diagnostic unavailable: {e}")

    data_loader = DataLoader(data, num_workers=2, batch_size=64, shuffle=True, pin_memory=True)

    all_indices = []
    all_indices_str = []
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]

    from tqdm import tqdm
    for d in tqdm(data_loader, desc="B1 Stage 2 forward"):
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
            for item, index in zip(collision_items, indices):
                code = [prefix[i].format(int(ind)) for i, ind in enumerate(index)]
                all_indices[item] = code
                all_indices_str[item] = str(code)
        tt += 1

    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    log.info(f"  PRE-resolve: items={tot_item}, unique={tot_indice}, "
             f"collision_rate={(tot_item - tot_indice) / tot_item:.4f}")

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
    log.info(f"  POST-resolve: {len(final_dups)} duplicates remain (expected 0)")

    n_rq_layers = len(ckpt_args['num_emb_list'])
    for li in range(n_rq_layers):
        codes_li = codes_array[:, li]
        unique_codes_li = np.unique(codes_li)
        log.info(f"  Layer {li} utilization: {len(unique_codes_li)} / {ckpt_args['num_emb_list'][li]} "
                 f"({100 * len(unique_codes_li) / ckpt_args['num_emb_list'][li]:.1f}%)")

    log.info(f"  first 5 codes (int): {codes_array[:5].tolist()}")
    np.save(args.output_path, codes_array)
    log.info(f"  Saved: {args.output_path}")

    # Diagnostic JSON
    diag_path = args.output_path.replace(".npy", "_diagnostic.json")
    diag = {
        "arm": "B1",
        "task": "task211_phase1_C2",
        "ckpt_path": args.ckpt_path,
        "n_items": int(tot_item),
        "n_unique_sid": int(len(np.unique(codes_array, axis=0))),
        "collision_rate_pre_resolve": float((tot_item - tot_indice) / tot_item),
        "per_layer_utilization": {
            f"layer_{li}": {
                "unique": int(len(np.unique(codes_array[:, li]))),
                "total_capacity": int(ckpt_args['num_emb_list'][li]),
                "fraction": float(len(np.unique(codes_array[:, li])) / ckpt_args['num_emb_list'][li]),
            }
            for li in n_rq_layers and [] or range(n_rq_layers)
        },
        "sinkhorn_iters": tt,
        "best_loss": float(ckpt.get("best_loss", -1.0)),
        "best_collision_rate": float(ckpt.get("best_collision_rate", -1.0)),
        "epoch": int(ckpt.get("epoch", -1)),
    }
    # 修上面那一行奇怪的 ternary, 直接重写
    diag["per_layer_utilization"] = {
        f"layer_{li}": {
            "unique": int(len(np.unique(codes_array[:, li]))),
            "total_capacity": int(ckpt_args['num_emb_list'][li]),
            "fraction": float(len(np.unique(codes_array[:, li])) / ckpt_args['num_emb_list'][li]),
        }
        for li in range(n_rq_layers)
    }
    with open(diag_path, "w") as f:
        json.dump(diag, f, indent=2)
    log.info(f"  Diagnostic saved: {diag_path}")


if __name__ == '__main__':
    main()