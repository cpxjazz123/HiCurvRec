#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task #297 / Issue #25 Gate 2 — Stage 2 SID codebook from Gate 1 ckpt

- 加载 Gate 1 ckpt: products/task297/hrqvae_issue25_gate1_phase_b/best_loss_model.pth
- Sinkhorn 5 iter + 4-digit dedup (跟 Gate 0 一致)
- 输出: (9922, 4) int .npy 给 Stage 3 T5
- 通过条件 (Issue #25 §Gate 2): unique >= 9500 + per-layer util 偏差 <= 5pp
"""
import argparse
import collections
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger("task297_issue25_gate2")

from model.hrqvae import HRQVAE
from model.utils import EmbDataset


def _get_args(args, name, default=None):
    if isinstance(args, dict):
        return args.get(name, default)
    return getattr(args, name, default)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/products/task297/hrqvae_issue25_gate1_phase_b/best_loss_model.pth")
    parser.add_argument("--data_path", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet")
    parser.add_argument("--output_path", type=str,
                        default="/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_issue25_gate2_k0128.npy")
    parser.add_argument("--max_sinkhorn_iters", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")
    args_cli = parser.parse_args()

    log.info("=" * 70)
    log.info("Task #297 / Issue #25 Gate 2 — SID codebook from Gate 1 ckpt")
    log.info("=" * 70)
    log.info(f"CKPT: {args_cli.ckpt_path}")
    log.info(f"OUT : {args_cli.output_path}")
    log.info(f"max_sinkhorn_iters: {args_cli.max_sinkhorn_iters}")

    device = torch.device(args_cli.device)
    ckpt = torch.load(args_cli.ckpt_path, map_location=torch.device('cpu'), weights_only=False)
    cfg = ckpt['args']
    state_dict = ckpt['state_dict']

    log.info(f"  num_emb_list={_get_args(cfg, 'num_emb_list')}, e_dim={_get_args(cfg, 'e_dim')}, "
             f"beta={_get_args(cfg, 'beta')}")

    data = EmbDataset(_get_args(cfg, 'data_path'))
    log.info(f"  Data: {len(data)} items, dim={data.dim}")

    model = HRQVAE(
        in_dim=data.dim,
        num_emb_list=_get_args(cfg, 'num_emb_list'),
        e_dim=_get_args(cfg, 'e_dim'),
        layers=_get_args(cfg, 'layers', [512, 256, 128, _get_args(cfg, 'e_dim')]),
        dropout_prob=_get_args(cfg, 'dropout_prob', 0.0),
        bn=_get_args(cfg, 'bn', False),
        loss_type=_get_args(cfg, 'loss_type'),
        quant_loss_weight=_get_args(cfg, 'quant_loss_weight', 1.0),
        beta=_get_args(cfg, 'beta'),
        kmeans_init=False,
        kmeans_iters=100,
        sk_eps=_get_args(cfg, 'sk_epsilons'),
        sk_iters=_get_args(cfg, 'sk_iters'),
    )
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    data_loader = DataLoader(data, batch_size=64, shuffle=False, num_workers=2)

    all_indices = []
    all_indices_str = []
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]

    log.info("\n=== 推断所有 items (use_sk=False) ===")
    t0 = time.time()
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
    log.info(f"  推断时间: {time.time()-t0:.1f}s, 总 items: {len(all_indices)}")

    all_indices = np.array(all_indices)
    all_indices_str = np.array(all_indices_str)

    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    log.info(f"\n  3-digit unique (no SK): {tot_indice}/{tot_item} = {tot_indice/tot_item*100:.2f}%")
    log.info(f"  3-digit collision: {(tot_item - tot_indice)/tot_item:.4f}")

    # Sinkhorn 5 iter 解决 collision (task84 模式: 多次迭代直到 collision 解决)
    log.info(f"\n=== Sinkhorn (max {args_cli.max_sinkhorn_iters} iter) ===")
    tt = 0
    while tt < args_cli.max_sinkhorn_iters:
        tot_item = len(all_indices_str)
        tot_indice = len(set(all_indices_str.tolist()))
        if tot_item == tot_indice:
            log.info(f"  iter {tt}: 0 collision, 提前终止")
            break
        collision_groups = collections.defaultdict(list)
        for i, s in enumerate(all_indices_str.tolist()):
            collision_groups[s].append(i)
        collision_groups = [v for v in collision_groups.values() if len(v) > 1]
        log.info(f"  iter {tt}: {len(collision_groups)} 个 3-digit 碰撞, 共 {sum(len(v) for v in collision_groups)} items")

        for collision_items in collision_groups:
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

    # 4-digit dedup
    all_indices_dict = {}
    for item, indices in enumerate(all_indices.tolist()):
        all_indices_dict[item] = list(indices)

    codes = []
    for key, value in all_indices_dict.items():
        code = [int(item.split('_')[1].strip('>')) for item in value]
        codes.append(code)

    codes_array = np.array(codes)
    codes_array = np.hstack((codes_array, np.zeros((codes_array.shape[0], 1), dtype=int)))

    unique_codes, counts = np.unique(codes_array, axis=0, return_counts=True)
    duplicates = unique_codes[counts > 1]
    if len(duplicates) > 0:
        log.info(f"\n  Resolving 4-digit duplicates ({len(duplicates)} groups)...")
        for duplicate in duplicates:
            duplicate_indices = np.where((codes_array == duplicate).all(axis=1))[0]
            for i, idx in enumerate(duplicate_indices):
                codes_array[idx, -1] = i

    new_unique_codes, new_counts = np.unique(codes_array, axis=0, return_counts=True)
    duplicates = new_unique_codes[new_counts > 1]
    if len(duplicates) > 0:
        log.info(f"  ⚠️  仍有 {len(duplicates)} duplicates: {duplicates[:5]}")
    else:
        log.info(f"  ✅ 4-digit 全 unique")

    log.info(f"\n  codes_array shape: {codes_array.shape}")
    log.info(f"  前 5 codes: {codes_array[:5]}")

    # Per-layer util 验证
    n_e_list = _get_args(cfg, 'num_emb_list')
    per_layer_util = []
    for li in range(len(n_e_list)):
        n_unique = len(set(codes_array[:, li].tolist()))
        per_layer_util.append(n_unique / n_e_list[li])
        log.info(f"  L{li} (K={n_e_list[li]}): unique={n_unique}/{n_e_list[li]} = {per_layer_util[li]*100:.2f}%")

    n_unique_4 = len(set([tuple(c) for c in codes_array.tolist()]))
    log.info(f"\n  4-digit unique: {n_unique_4}/{len(codes_array)} = {n_unique_4/len(codes_array)*100:.2f}%")

    # Gate 2 通过条件
    log.info("\n" + "=" * 70)
    log.info("Issue #25 Gate 2 决策")
    log.info("=" * 70)
    log.info(f"  L0/L1/L2 utilization: {per_layer_util[0]*100:.2f}% / "
             f"{per_layer_util[1]*100:.2f}% / {per_layer_util[2]*100:.2f}%")
    log.info(f"  4-digit unique: {n_unique_4}")

    util_pass = all(u >= 0.90 for u in per_layer_util)
    unique_pass = n_unique_4 >= 9500
    gate2_passed = util_pass and unique_pass
    log.info(f"\n  通过条件: 三层 util >= 90% = {util_pass}")
    log.info(f"  通过条件: 4-digit unique >= 9500 = {unique_pass}")
    if gate2_passed:
        log.info("\n✅ Issue #25 Gate 2 PASS")
    else:
        log.info("\n❌ Issue #25 Gate 2 FAIL")

    # 落盘
    Path(args_cli.output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(args_cli.output_path, codes_array)
    log.info(f"\nOK 落盘: {args_cli.output_path}")
    log.info(f"  shape={codes_array.shape}, unique={n_unique_4}/{len(codes_array)}")


if __name__ == "__main__":
    main()
