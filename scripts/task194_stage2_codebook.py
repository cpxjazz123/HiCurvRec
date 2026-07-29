#!/usr/bin/env python3
"""Task #194 Stage 2 — fork of task84_hgrec_stage2_codebook.py with --ckpt_path/--output_path.

用法:
  python3 scripts/task194_stage2_codebook.py \
      --ckpt_path <path/to/best_collision_model.pth> \
      --output_path <path/to/Instruments_t5_rqvae_k0<K0>.npy> \
      --device cuda:0 \
      --max_sinkhorn_iters 30
"""
import argparse
import collections
import logging
import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
from model.utils import EmbDataset
from model.hrqvae import HRQVAE


def check_collision(all_indices_str):
    return len(all_indices_str) == len(set(all_indices_str.tolist()))


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
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    # R11 fix (user 2026-07-25): override sk_eps > 0 to actually enable Sinkhorn path
    # ckpt was trained with sk_epsilons=[0,0,0], so HVectorQuantization.forward
    # always takes the argmin branch (sk_eps <= 0 → argmin). 30 Sinkhorn iters were no-ops.
    # Override: 0.003 = author's recommended default; 0.5 = author's leftover path value.
    parser.add_argument("--sk_eps_override", type=float, default=0.003,
                        help="Override vq.sk_eps for all 3 layers. 0.0 = disabled (argmin only).")
    args = parser.parse_args()

    device = torch.device(args.device)
    log = logging.getLogger("task194_stage2")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    log.info(f"Loading ckpt: {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, weights_only=False, map_location=torch.device('cpu'))
    ckpt_args = ckpt['args']
    state_dict = ckpt['state_dict']
    # R11 fix: ckpt['args'] 可能是 Namespace (task84) 或 dict (task89 vars(args))
    def _get_arg(name, default=None):
        if isinstance(ckpt_args, dict):
            return ckpt_args.get(name, default)
        return getattr(ckpt_args, name, default)
    num_emb_list = _get_arg('num_emb_list', None)
    e_dim = _get_arg('e_dim', None)
    loss_type = _get_arg('loss_type', '?')
    beta = _get_arg('beta', '?')
    log.info(f"  num_emb_list={num_emb_list}, e_dim={e_dim}, "
             f"loss_type={loss_type}, beta={beta}")

    # EmbDataset path: ckpt 中 args.data_path 已是绝对路径 (task89 default = /home/wlia0047/.../item_emb.parquet)
    # 兼容 task84 相对路径 './dataset/Instruments/item_emb.parquet'
    data_path = _get_arg('data_path', None)
    if not data_path:
        raise RuntimeError("ckpt['args'] missing data_path")
    if not os.path.isabs(data_path):
        data_path = os.path.join('/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec', data_path)
    log.info(f"  data_path={data_path}")
    data = EmbDataset(data_path)
    log.info(f"  dataset size={len(data)}, dim={data.dim}")

    model = HRQVAE(in_dim=data.dim,
                   num_emb_list=num_emb_list,
                   e_dim=e_dim,
                   layers=_get_arg('layers', [512, 256, 128, 64]),
                   dropout_prob=_get_arg('dropout_prob', 0.0),
                   bn=_get_arg('bn', False),
                   loss_type=loss_type,
                   quant_loss_weight=_get_arg('quant_loss_weight', 1.0),
                   beta=beta,
                   kmeans_init=_get_arg('kmeans_init', True),
                   kmeans_iters=_get_arg('kmeans_iters', 1000),
                   sk_eps=_get_arg('sk_epsilons', [0.0, 0.0, 0.0]),
                   sk_iters=_get_arg('sk_iters', 50))

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # R11 fix (user 2026-07-25): override sk_eps to actually enable Sinkhorn.
    # ckpt's sk_epsilons=[0,0,0] → all 3 vq_layers had sk_eps=0, so Sinkhorn path was
    # always skipped (use_sk=False or sk_eps<=0 → argmin). Force-override after load.
    if args.sk_eps_override > 0:
        for li, vq in enumerate(model.hrq.vq_layers):
            vq.sk_eps = args.sk_eps_override
            log.info(f"  [Sinkhorn fix] layer {li}: sk_eps overridden → {args.sk_eps_override}")
        log.info(f"  [Sinkhorn fix] All {len(model.hrq.vq_layers)} layers enabled with sk_eps={args.sk_eps_override}")
    else:
        log.warning(f"  [Sinkhorn fix] sk_eps_override=0, Sinkhorn DISABLED (argmin only)")

    data_loader = DataLoader(data,
                             num_workers=args.num_workers,
                             batch_size=args.batch_size,
                             shuffle=True,
                             pin_memory=True)

    all_indices = []
    all_indices_str = []
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]

    from tqdm import tqdm
    for d in tqdm(data_loader):
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

    log.info(f"Initial collision_rate: "
             f"{1 - len(set(all_indices_str.tolist()))/len(all_indices_str):.4f}")

    tt = 0
    while True:
        if tt >= args.max_sinkhorn_iters or check_collision(all_indices_str):
            break

        collision_item_groups = get_collision_item(all_indices_str)
        log.info(f"iter {tt}: {len(collision_item_groups)} collision groups, fixing...")
        for collision_items in collision_item_groups:
            d = data[collision_items].to(device)
            indices = model.get_indices(d, use_sk=True)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for j, ci in enumerate(collision_items):
                code = []
                for i, ind in enumerate(indices[j]):
                    code.append(prefix[i].format(int(ind)))
                all_indices[ci] = code
                all_indices_str[ci] = str(code)
        tt += 1

    # Final collision
    final_collision = 1 - len(set(all_indices_str.tolist()))/len(all_indices_str)
    log.info(f"Final collision_rate: {final_collision:.4f} after {tt} Sinkhorn iters")

    # R11 fix: convert strings (e.g. "<a_5>") back to int codes, add 4th-digit dedup column
    # 镜像 task84_hgrec_stage2_codebook.py line 140-167
    all_indices_dict = {}
    for item, indices in enumerate(all_indices.tolist()):
        all_indices_dict[item] = list(indices)

    codes = []
    for key, value in all_indices_dict.items():
        # value = ["<a_5>", "<b_3>", "<c_7>", "<d_2>"] → [5, 3, 7, 2]
        code = [int(item.split('_')[1].strip('>')) for item in value]
        codes.append(code)

    codes_array = np.array(codes, dtype=int)

    # Add 4th-digit dedup column (initialize 0)
    codes_array = np.hstack((codes_array, np.zeros((codes_array.shape[0], 1), dtype=int)))

    # Resolve duplicates by incrementing the last dimension
    unique_codes, counts = np.unique(codes_array, axis=0, return_counts=True)
    duplicates = unique_codes[counts > 1]
    if len(duplicates) > 0:
        log.info(f"Resolving {len(duplicates)} duplicate code groups via 4th-digit dedup...")
        for duplicate in duplicates:
            duplicate_indices = np.where((codes_array == duplicate).all(axis=1))[0]
            for i, idx in enumerate(duplicate_indices):
                codes_array[idx, -1] = i  # Increment the last digit for resolving duplicates

    new_unique_codes, new_counts = np.unique(codes_array, axis=0, return_counts=True)
    new_dups = new_unique_codes[new_counts > 1]
    if len(new_dups) > 0:
        log.warning(f"{len(new_dups)} duplicates remain after 4th-digit dedup (Sinkhorn insufficient)")
    else:
        log.info("All codes unique after 4th-digit dedup")

    # Save
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    np.save(args.output_path, codes_array)
    log.info(f"Saved SID codebook to {args.output_path} (shape={codes_array.shape}, dtype={codes_array.dtype})")
    log.info(f"First 5 codes: {codes_array[:5].tolist()}")


if __name__ == "__main__":
    main()