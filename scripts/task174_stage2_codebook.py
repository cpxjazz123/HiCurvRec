#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #174 Stage 2 fork: 用 MCKGGatingHRQVAE 加载 D 臂 ckpt (含 gate_net 参数).

Mirror of scripts/task164_stage2_codebook.py with 3 changes:
  - import MCKGGatingHRQVAE instead of FreeCurvHRQVAE
  - instantiate with use_gating=True, gate_hidden=32
  - load_state_dict with strict=False (some params may be in unexpected positions)
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')

from model.utils import EmbDataset
from model.hrqvae_mckg_gating import MCKGGatingHRQVAE


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
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_sinkhorn_iters", type=int, default=30)
    args = parser.parse_args()

    device = torch.device(args.device)
    log = logging.getLogger("task174_stage2")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    log.info(f"Loading ckpt: {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, weights_only=False, map_location=torch.device('cpu'))
    ckpt_args = ckpt['args']
    state_dict = ckpt['state_dict']
    log.info(f"  M={ckpt_args['M']}, kappa_max={ckpt_args['kappa_max']}, num_emb_list={ckpt_args['num_emb_list']}")
    log.info(f"  best_loss={ckpt.get('best_loss', '?')}, epoch={ckpt.get('epoch', '?')}")
    log.info(f"  gating_enabled={ckpt_args.get('gating_enabled', '?')}, gate_hidden={ckpt_args.get('gate_hidden', '?')}")

    data = EmbDataset(ckpt_args['data_path'])
    log.info(f"  Data: {len(data)} items, dim={data.dim}")

    # D 臂 use MCKGGatingHRQVAE
    model = MCKGGatingHRQVAE(
        in_dim=data.dim,
        num_emb_list=ckpt_args['num_emb_list'],
        e_dim=ckpt_args['e_dim'],
        M=ckpt_args['M'],
        kappa_max=ckpt_args['kappa_max'],
        layers=ckpt_args['layers'],
        dropout_prob=0.0,
        bn=False,
        loss_type=ckpt_args['loss_type'],
        quant_loss_weight=ckpt_args['quant_loss_weight'],
        beta=ckpt_args['beta'],
        kmeans_init=ckpt_args['kmeans_init'],
        kmeans_iters=ckpt_args['kmeans_iters'],
        sk_eps=ckpt_args['sk_epsilons'],
        sk_iters=ckpt_args['sk_iters'],
        use_gating=ckpt_args.get('gating_enabled', True),
        gate_hidden=ckpt_args.get('gate_hidden', 32),
    )
    # use strict=False 因为 gate_net 可能跟 ckpt 里顺序略有差异
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    for li, vq in enumerate(model.hrq.vq_layers):
        kappas = vq.kappa_m().detach().cpu().tolist()
        log.info(f"  Layer {li} κ_m = [{', '.join(f'{k:+.4f}' for k in kappas)}]")
        if vq.use_gating:
            with torch.no_grad():
                # Sample gating weights on a few real items
                sample_x = data[:32].to(device)
                z = model.encoder(sample_x)
                w = vq.get_gating_weights(z)
                log.info(f"    L{li} gating weights sample mean: "
                         f"[{', '.join(f'{v:.3f}' for v in w.mean(0).tolist())}]")

    data_loader = DataLoader(data, num_workers=2, batch_size=64, shuffle=True, pin_memory=True)

    all_indices = []
    all_indices_str = []
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]

    from tqdm import tqdm
    for d in tqdm(data_loader, desc="Task #174 Stage 2 forward"):
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
    util_per_layer = [0, 0, 0]
    for s in counter:
        codes = eval(s)
        for li, ind in enumerate(codes):
            ind_id = int(ind.split('_')[1].rstrip('>'))
            util_per_layer[li] = max(util_per_layer[li], ind_id + 1)
    log.info(f"  Per-layer codebook usage: {util_per_layer} / [32, 64, 256]")

    # Post-process: convert str → int + append EOS column (4th digit) + dedup
    # (mirror of task164_stage2_codebook.py line 121-145, required for GenRecDataset.item2code)
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

    new_unique_codes, _ = np.unique(codes_array, axis=0, return_counts=True)
    final_dups = new_unique_codes[np.unique(codes_array, axis=0, return_counts=True)[1] > 1]
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
        "ckpt_path": args.ckpt_path,
        "n_items": int(len(all_indices_str)),
        "n_unique_sid": int(len(np.unique(codes_array, axis=0))),
        "per_layer_utilization": {
            f"layer_{li}": {
                "unique": int(len(np.unique(codes_array[:, li]))),
                "total_capacity": int(ckpt_args['num_emb_list'][li]),
                "fraction": float(len(np.unique(codes_array[:, li])) / ckpt_args['num_emb_list'][li]),
            }
            for li in range(n_rq_layers)
        },
        "kappa_m_final": [
            [float(k) for k in vq.kappa_m().detach().cpu().tolist()]
            for vq in model.hrq.vq_layers
        ],
        "sinkhorn_iters": tt,
        "best_loss": float(ckpt.get("best_loss", -1.0)),
        "epoch": int(ckpt.get("epoch", -1)),
    }
    with open(diag_path, "w") as f:
        json.dump(diag, f, indent=2)
    log.info(f"  Diagnostic saved: {diag_path}")


if __name__ == '__main__':
    main()