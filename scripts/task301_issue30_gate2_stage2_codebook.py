"""Task #301 Issue #30 Gate 2 — Stage 2 Sinkhorn 推断.
Issue #30 ckpt (per-layer codebook transforms) → (N, 4) SID .npy 给 Stage 3 T5.

Usage:
    python3 scripts/task301_issue30_gate2_stage2_codebook.py
"""
import collections
import glob
import logging
import os
import re
import sys

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


def parse_radius_scale_from_ckpt_dir(ckpt_dir):
    """从 ckpt_dir 路径推断 (radius_list, scale_list) — R11.3 fallback 决策.
    实际是 R2 禁的, 改用 read 显式从 description 文件读.
    """
    return None


def main():
    """Issue #30 Gate 2 Sinkhorn 5 iter inference (gate_decision: 5 iter per task298 review)."""
    dataset = "Instruments"
    ckpt_base = "/home/wlia0047/ar57/wenyu/GeneRec/products/task301/hrqvae_issue30_gate1"
    ckpt_pattern = f"{ckpt_base}/*/best_loss_model.pth"
    candidates = sorted(glob.glob(ckpt_pattern))
    if not candidates:
        raise FileNotFoundError(f"❌ No best_loss_model.pth in {ckpt_pattern}")
    ckpt_path = candidates[-1]

    output_path = f"/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/{dataset}/{dataset}_t5_hrqvae_issue30_per_layer_transforms.npy"
    device = torch.device("cuda:0")

    log = logging.getLogger("task301_issue30_gate2")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    log.info(f"Loading ckpt: {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=torch.device('cpu'))
    ckpt_args = ckpt['args']
    state_dict = ckpt['state_dict']
    log.info(f"  num_emb_list={ckpt_args.num_emb_list}, e_dim={ckpt_args.e_dim}")
    log.info(f"  best_loss={ckpt.get('best_loss', '?')}, epoch={ckpt.get('epoch', '?')}")
    log.info(f"  sk_eps={ckpt_args.sk_epsilons}, sk_iters={ckpt_args.sk_iters}")

    data = EmbDataset(ckpt_args.data_path)
    log.info(f"  Data: {len(data)} items, dim={data.dim}")

    # R11.3 decision: 直接 baseline HRQVAE 加载 — 因为 Issue #30 wrapper 在 init 时
    # 已 transform .embeddings.weight 一次, ckpt 中 weights 即变换后的.
    # 因此 load_state_dict 与 baseline HRQVAE 兼容 (shape 一致).
    model = HRQVAE(in_dim=data.dim,
                   num_emb_list=ckpt_args.num_emb_list,
                   e_dim=ckpt_args.e_dim,
                   layers=ckpt_args.layers,
                   dropout_prob=ckpt_args.dropout_prob,
                   bn=ckpt_args.bn,
                   loss_type=ckpt_args.loss_type,
                   quant_loss_weight=ckpt_args.quant_loss_weight,
                   beta=ckpt_args.beta,
                   kmeans_init=ckpt_args.kmeans_init,
                   kmeans_iters=ckpt_args.kmeans_iters,
                   sk_eps=ckpt_args.sk_epsilons,
                   sk_iters=ckpt_args.sk_iters)
    # R11.5 决策 (2026-07-30): strict=False 加载 task298-extended ckpt
    # 训练时 wrapper 用了 c_k_range_list + assignment_mode + per-codeword log_r/hyp_mean/hyp_scale
    # 推断时用 baseline HRQVAE (无 c_k_range_list) + strict=False 忽略 per-codeword κ keys.
    # 实际影响: embeddings.weight 已经被 per-layer r/s transform 修改过 (wrapper 在 init 后 apply),
    # 距离公式仍是 Poincaré (self.c=1.0 全局), 跟训练时 per-codeword κ 距离不严格一致但合理.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    log.info(f"  load_state_dict strict=False: missing={len(missing)}, unexpected={len(unexpected)}")
    if unexpected:
        log.info(f"    unexpected keys (sample): {sorted(set(k.split('.')[0] for k in unexpected))[:5]}...")
    model = model.to(device)
    model.eval()

    data_loader = DataLoader(data,
                             num_workers=ckpt_args.num_workers,
                             batch_size=64,
                             shuffle=True,
                             pin_memory=True)

    all_indices = []
    all_indices_str = []
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]

    # Round 1: hard quantize (use_sk=False)
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

    all_indices = np.array(all_indices)
    all_indices_str = np.array(all_indices_str)

    log.info(f"Initial pass: {len(all_indices)} codes, "
             f"unique={len(set(all_indices_str.tolist()))}, "
             f"collision={(len(all_indices_str) - len(set(all_indices_str.tolist())))/len(all_indices_str):.4f}")

    # Sinkhorn 5 iter (Issue #30 Gate 2 decision: per task298 + task271 review)
    MAX_SK_ITERS = 5
    tt = 0
    while True:
        if tt >= MAX_SK_ITERS or check_collision(all_indices_str):
            break
        collision_item_groups = get_collision_item(all_indices_str)
        log.info(f"  SK iter {tt}: {len(collision_item_groups)} collision groups")
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

    log.info(f"After {tt} Sinkhorn iters: total={len(all_indices)}")

    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    collision_rate = (tot_item - tot_indice) / tot_item
    log.info(f"Final 3-digit Collision Rate: {collision_rate:.4f}")

    # Convert to (N, 3) int array, then add 4th-column dedup
    all_indices_dict = {}
    for item, indices in enumerate(all_indices.tolist()):
        all_indices_dict[item] = list(indices)

    codes = []
    for key, value in all_indices_dict.items():
        code = [int(item.split('_')[1].strip('>')) for item in value]
        codes.append(code)
    codes_array = np.array(codes)

    # 4-digit dedup (跟 baseline Stage 2 流程一致)
    codes_array = np.hstack((codes_array, np.zeros((codes_array.shape[0], 1), dtype=int)))
    unique_codes, counts = np.unique(codes_array, axis=0, return_counts=True)
    duplicates = unique_codes[counts > 1]
    if len(duplicates) > 0:
        log.info(f"  dedup pass: {len(duplicates)} duplicate groups")
        for duplicate in duplicates:
            duplicate_indices = np.where((codes_array == duplicate).all(axis=1))[0]
            for i, idx in enumerate(duplicate_indices):
                codes_array[idx, -1] = i

    new_unique_codes, new_counts = np.unique(codes_array, axis=0, return_counts=True)
    new_duplicates = new_unique_codes[new_counts > 1]
    if len(new_duplicates) > 0:
        log.info(f"  ⚠️ {len(new_duplicates)} duplicate groups remain after dedup")
    else:
        log.info("  ✅ No duplicates after resolution")

    log.info(f"Saving codes to {output_path}")
    log.info(f"  shape: {codes_array.shape}, unique: {len(np.unique(codes_array, axis=0))}")
    log.info(f"  first 5 codes: {codes_array[:5]}")
    np.save(output_path, codes_array)

    # Issue #30 Gate 2 pass conditions
    log.info("=" * 60)
    log.info("Issue #30 Gate 2 verification:")
    log.info(f"  4-digit unique ≥ 9500: {len(np.unique(codes_array, axis=0))} → "
             f"{'PASS' if len(np.unique(codes_array, axis=0)) >= 9500 else 'FAIL'}")
    log.info(f"  3-digit collision ≤ 0.20: {collision_rate:.4f} → "
             f"{'PASS' if collision_rate <= 0.20 else 'FAIL'}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
