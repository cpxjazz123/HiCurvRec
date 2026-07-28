#!/usr/bin/env python3
"""
Task #209 Phase 2a — A3 路径正则 Stage 1 ckpt → Stage 2 SID 推断.

Inputs:
  ckpt_path: products/task209/phase1_arm_A3/Jul-26-2026_18-26-17_*/best_loss_model.pth
  output_path: HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task209_A3.npy
  device: cuda:0 (R7: GPU 0/2/3 全空闲)

复用了 task181_stage2_codebook.py 的 Sinkhorn + dedup 主逻辑.
唯一区别: ckpt 来自 task209 A3 (collision 99.96% 高坍缩), 输出 task209 专用 SID 文件名,
不覆盖 #181 baseline 的 Instruments_t5_rqvae_paper_fix.npy.
"""
import collections, json, logging, numpy as np, torch, os, glob, sys
from time import time
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/scripts')

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

if __name__ == "__main__":
    dataset = "Instruments"
    ckpt_path = (
        "/home/wlia0047/ar57/wenyu/GeneRec/products/task209/phase1_arm_A3/"
        "Jul-26-2026_18-26-17_beta_0.500_codebook_[64,128,256]_sk_0.000/best_loss_model.pth"
    )
    output_path = (
        f"/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/{dataset}/"
        f"{dataset}_t5_rqvae_task209_A3.npy"
    )
    device = torch.device("cuda:0")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"❌ A3 ckpt missing: {ckpt_path}")
    print(f"Stage 2 (A3): loading ckpt from {ckpt_path}")
    print(f"Output: {output_path}")

    ckpt = torch.load(ckpt_path, weights_only=False, map_location=torch.device('cpu'))
    args = ckpt['args']
    state_dict = ckpt['state_dict']
    print(f"args: num_emb_list={args.num_emb_list} e_dim={args.e_dim} layers={args.layers}")
    print(f"      sk_epsilons={args.sk_epsilons} kmeans_init={args.kmeans_init}")

    data = EmbDataset(args.data_path)

    # 必须把 #209 新增的 path-reg / norm-target / dual-codebook 相关 flag 都传进去,
    # 否则 HRQVAE 会拒载 (state_dict 包含 r_target_norm_list 等参数).
    model = HRQVAE(
        in_dim=data.dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        beta=args.beta,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_eps=args.sk_epsilons,
        sk_iters=args.sk_iters,
        # Task #209 path reg args
        w_path=getattr(args, 'w_path', 0.0),
        path_geometry=getattr(args, 'path_geometry', 'hyp'),
        rho_targets_path=getattr(args, 'rho_targets_path', None),
        # Task #196/199/203 args
        scale_norm=getattr(args, 'scale_norm', 'none'),
        r_target_norm_list=getattr(args, 'norm_target', None),
        gamma_norm=getattr(args, 'gamma_norm', 0.0),
        kappa_mode=getattr(args, 'kappa_mode', 'fixed'),
        r_target_list=getattr(args, 'r_target_list', None),
        r_median_init=getattr(args, 'r_median_init', 1.0),
        theta_init=getattr(args, 'theta_init', 0.0),
    )
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    data_loader = DataLoader(data,
                             num_workers=args.num_workers,
                             batch_size=64,
                             shuffle=True,
                             pin_memory=True)

    all_indices = []
    all_indices_str = []
    prefix = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>","<e_{}>"]

    print("Stage 2 (A3): initial inference (use_sk=False)...")
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

    print(f"Initial: collision_rate = {(len(all_indices_str) - len(set(all_indices_str))) / len(all_indices_str):.4f}")

    # Sinkhorn-Knopp 解码 (30 轮 dedup)
    tt = 0
    while True:
        if tt >= 30 or check_collision(all_indices_str):
            break
        collision_item_groups = get_collision_item(all_indices_str)
        print(f"Iter {tt}: {len(collision_item_groups)} collision groups")
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

    print(f"After SK ({tt} iters): collision_rate = {(len(all_indices_str) - len(set(all_indices_str))) / len(all_indices_str):.4f}")
    print("All indices number:", len(all_indices))
    print("Max number of conflicts:", max(get_indices_count(all_indices_str).values()))

    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    collision_rate = (tot_item - tot_indice) / tot_item
    print(f"Stage 2 (A3) final collision_rate: {collision_rate}")

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
        print(f"Resolving {len(duplicates)} duplicates via 4th-digit dedup...")
        for duplicate in duplicates:
            duplicate_indices = np.where((codes_array == duplicate).all(axis=1))[0]
            for i, idx in enumerate(duplicate_indices):
                codes_array[idx, -1] = i

    new_unique_codes, new_counts = np.unique(codes_array, axis=0, return_counts=True)
    duplicates = new_unique_codes[new_counts > 1]

    if len(duplicates) > 0:
        print(f"⚠️  There still have duplicates: {len(duplicates)}")
    else:
        print("✅ There are no duplicates in the codes after resolution.")

    print(f"Saving codes to {output_path}")
    print(f"Shape: {codes_array.shape}, dtype: {codes_array.dtype}")
    print(f"First 5 codes: {codes_array[:5]}")
    np.save(output_path, codes_array)
    print("Stage 2 (A3) complete.")
