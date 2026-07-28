#!/usr/bin/env python3
"""
Task #200 Stage 2 — 双码本 ckpt 推断 → SID .npy
支持 v3 (F.mse_loss) + v4 (sum 版 rec_align) ckpt.

用法 (等用户拍清单 4 选项 D 启动):
  python3 scripts/task200_stage2_codebook.py --ckpt_path <best_collision_model.pth> --output_suffix dual_v4

ckpt 路径:
  v3: products/task200/dual_arm_C_v3/Jul-26-2026_01-34-43_*/best_collision_model.pth
  v4: products/task200/dual_arm_C_v4/Jul-26-2026_01-41-57_*/best_collision_model.pth

输出 .npy 路径: HG-Rec/dataset/Instruments/Instruments_t5_rqvae_{suffix}.npy
Shape: (9922, 4) [3 hierarchies + 1 dedup digit]
"""
import collections, json, logging, numpy as np, torch, os, glob, argparse, sys
from time import time
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader

REPO = "/home/wlia0047/ar57/wenyu/GeneRec"
sys.path.insert(0, f"{REPO}/HG-Rec")

from model.utils import *
from model.hrqvae import HRQVAE

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_suffix", type=str, required=True,
                        help="e.g. 'dual_v3' / 'dual_v4' → Inferred SID file name")
    parser.add_argument("--dataset", type=str, default="Instruments")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    output_path = f"{REPO}/HG-Rec/dataset/{args.dataset}/{args.dataset}_t5_rqvae_{args.output_suffix}.npy"
    device = torch.device(args.device)

    print(f"[Stage 2 Task #200] loading ckpt from {args.ckpt_path}")
    print(f"[Stage 2 Task #200] output to {output_path}")

    ckpt = torch.load(args.ckpt_path, weights_only=False, map_location=torch.device('cpu'))
    saved_args = ckpt['args']
    state_dict = ckpt['state_dict']

    data = EmbDataset(saved_args.data_path)

    # 双码本参数从 ckpt args 读, 缺省值 false / None (兼容 v3/v4)
    use_centering_list = None
    if getattr(saved_args, "centering_layers", None):
        centering_idx = set()
        for s in saved_args.centering_layers.split(","):
            s = s.strip()
            if not s:
                continue
            centering_idx.add(int(s))
        use_centering_list = [i in centering_idx for i in range(len(saved_args.num_emb_list))]

    model = HRQVAE(in_dim=data.dim,
                  num_emb_list=saved_args.num_emb_list,
                  e_dim=saved_args.e_dim,
                  layers=saved_args.layers,
                  dropout_prob=saved_args.dropout_prob,
                  bn=saved_args.bn,
                  loss_type=saved_args.loss_type,
                  quant_loss_weight=saved_args.quant_loss_weight,
                  beta=saved_args.beta,
                  kmeans_init=saved_args.kmeans_init,
                  kmeans_iters=saved_args.kmeans_iters,
                  sk_eps=saved_args.sk_epsilons,
                  sk_iters=saved_args.sk_iters,
                  curvature_list=saved_args.curvatures,
                  euclidean_qloss=saved_args.euclidean_qloss,
                  loss_mult_codebook=saved_args.loss_mult_codebook,
                  dual_codebook=getattr(saved_args, "dual_codebook", False),
                  use_centering_list=use_centering_list,
                  )

    # 旧 v3 ckpt 不含 z_mean → strict=False (允许缺 z_mean, 由 model 自身 init)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load_state_dict] missing keys (允许, 用 init): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[load_state_dict] unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    model = model.to(device)
    model.eval()
    print(model)

    data_loader = DataLoader(data,
                             num_workers=saved_args.num_workers,
                             batch_size=args.batch_size,
                             shuffle=True,
                             pin_memory=True)

    all_indices = []
    all_indices_str = []
    prefix = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>","<e_{}>"]

    # 第一次 forward: 无 Sinkhorn (argmin 路径, dual_codebook 走 emb_geo 方向)
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

    # 冲突解决: 用 Sinkhorn (sk_eps>0 才能解冲突), 最多 30 轮
    tt = 0
    while True:
        if tt >= 30 or check_collision(all_indices_str):
            break
        collision_item_groups = get_collision_item(all_indices_str)
        print(f"[Collision] iter {tt}: {len(collision_item_groups)} groups")
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

    print(f"[Final] all_indices number: {len(all_indices)}")
    print(f"[Final] max conflicts: {max(get_indices_count(all_indices_str).values())}")

    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    collision_rate = (tot_item - tot_indice) / tot_item
    print(f"[Final] collision_rate: {collision_rate:.4f}")

    all_indices_dict = {}
    for item, indices in enumerate(all_indices.tolist()):
        all_indices_dict[item] = list(indices)

    codes = []
    for key, value in all_indices_dict.items():
        code = [int(item.split('_')[1].strip('>')) for item in value]
        codes.append(code)

    codes_array = np.array(codes)
    # 追加 1 列 dedup digit (Stage 3/4 需要 4 列)
    codes_array = np.hstack((codes_array, np.zeros((codes_array.shape[0], 1), dtype=int)))

    unique_codes, counts = np.unique(codes_array, axis=0, return_counts=True)
    duplicates = unique_codes[counts > 1]

    if len(duplicates) > 0:
        print(f"[Dedup] resolving {len(duplicates)} duplicate code sequences...")
        for duplicate in duplicates:
            duplicate_indices = np.where((codes_array == duplicate).all(axis=1))[0]
            for i, idx in enumerate(duplicate_indices):
                codes_array[idx, -1] = i

    new_unique_codes, new_counts = np.unique(codes_array, axis=0, return_counts=True)
    duplicates = new_unique_codes[new_counts > 1]

    if len(duplicates) > 0:
        print(f"⚠️ 仍有 {len(duplicates)} duplicates: {duplicates[:5]}")
    else:
        print("✅ 4-digit dedup 后无重复")

    print(f"[Save] codes to {output_path}")
    print(f"[Save] shape: {codes_array.shape}, first 5: {codes_array[:5]}")
    np.save(output_path, codes_array)
    print("[Stage 2 Task #200] COMPLETE.")