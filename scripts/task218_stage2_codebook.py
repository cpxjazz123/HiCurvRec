"""Task #218 Stage 2 — SID codebook generation from per_codeword_kappa + spread ckpt.

Fork of task84_hgrec_stage2_codebook.py:
  - 用 task218 ckpt (per_codeword_kappa + spread + NormCap + simvq, e_dim=32)
  - 输出 (9922, 4) .npy 到 /HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_pck_spread.npy
  - 跟 task84 Stage 3 脚本兼容 (--code_path 指向这个文件即可)
"""
import collections, json, logging
import numpy as np
import torch
import os, glob, sys

from torch.utils.data import DataLoader
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')
from model.utils import EmbDataset
from model.hrqvae import HRQVAE


def check_collision(all_indices_str):
    return len(all_indices_str) == len(set(all_indices_str.tolist()))


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
    # 找 task218 最佳 ckpt: 优先 best_loss, 备选 best_collision, 最后 epoch_199
    ckpt_base = "/home/wlia0047/ar57/wenyu/GeneRec/products/task218/stage1_pck_spread"
    candidates_loss = sorted(glob.glob(f"{ckpt_base}/*/best_loss_model.pth"))
    candidates_collision = sorted(glob.glob(f"{ckpt_base}/*/best_collision_model.pth"))
    candidates_epoch = sorted(glob.glob(f"{ckpt_base}/*/epoch_199_*_model.pth"))
    candidates = candidates_loss or candidates_collision or candidates_epoch
    if not candidates:
        raise FileNotFoundError(f"❌ No ckpt found in {ckpt_base}")
    ckpt_path = candidates[-1]
    print(f"[task218 Stage 2] loading ckpt: {ckpt_path}")

    output_path = f"/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/{dataset}/{dataset}_t5_hrqvae_pck_spread.npy"
    device = torch.device("cuda:0")

    ckpt = torch.load(ckpt_path, weights_only=False, map_location=torch.device('cpu'))
    args = ckpt['args']
    state_dict = ckpt['state_dict']

    data = EmbDataset(args.data_path)
    print(f"[task218 Stage 2] data: {data.dim}-dim, {len(data)} samples")

    # 重建 HRQVAE (跟 task84 launcher 一致)
    model = HRQVAE(in_dim=data.dim,
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
                   product_manifold=getattr(args, 'product_manifold', False),
                   angular_dim=getattr(args, 'angular_dim', None),
                   radial_dim=getattr(args, 'radial_dim', None),
                   assignment_mode=getattr(args, 'assignment_mode', 'shared'),
                   c_k_min=getattr(args, 'c_k_min', 0.5),
                   c_k_max=getattr(args, 'c_k_max', 20.0),
                   c_k_seed=getattr(args, 'c_k_seed', 42),
                   c_k_update_mode=getattr(args, 'c_k_update_mode', 'fixed'),
                   c_k_alpha=getattr(args, 'c_k_alpha', 0.3),
                   c_k_clamp_min=getattr(args, 'c_k_clamp_min', 0.5),
                   c_k_clamp_max=getattr(args, 'c_k_clamp_max', 2.0),
                   c_k_dead_thr=getattr(args, 'c_k_dead_thr', 20),
                   distance_mode=getattr(args, 'distance_mode', 'sphere'),
                   use_normcap=getattr(args, 'use_normcap', False),
                   normcap_target=getattr(args, 'normcap_target', 0.4),
                   w_anchor=getattr(args, 'w_anchor', 0.0),
                   rho_min=getattr(args, 'rho_min', 1.5),
                   rho_max=getattr(args, 'rho_max', 2.9),
                   anti_collapse=getattr(args, 'anti_collapse', 'none'))

    result = model.load_state_dict(state_dict, strict=False)
    print(f"[task218 Stage 2] ckpt loaded: missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}")
    if result.missing_keys:
        print(f"  missing keys (sample): {result.missing_keys[:5]}")
    if result.unexpected_keys:
        print(f"  unexpected keys (sample): {result.unexpected_keys[:5]}")

    model = model.to(device)
    model.eval()

    data_loader = DataLoader(data, num_workers=0, batch_size=64, shuffle=False, pin_memory=True)
    all_indices = []
    all_indices_str = []
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]

    from tqdm import tqdm
    for d in tqdm(data_loader, desc="encoding"):
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
    while True:
        if tt >= 30 or check_collision(all_indices_str):
            break
        collision_item_groups = get_collision_item(all_indices_str)
        print(f"[collision pass {tt}] {len(collision_item_groups)} groups")
        for collision_items in collision_item_groups:
            d = data[collision_items].to(device)
            indices = model.get_indices(d, use_sk=True)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for item, index in zip(collision_items, indices):
                code = [prefix[i].format(int(ind)) for i, ind in enumerate(index)]
                all_indices[item] = code
                all_indices_str[item] = str(code)
        tt += 1

    print(f"All indices: {len(all_indices)}")
    print(f"Max conflicts per code: {max(get_indices_count(all_indices_str).values())}")
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    print(f"Collision Rate: {(tot_item-tot_indice)/tot_item}")

    codes = []
    for indices in all_indices.tolist():
        code = [int(item.split('_')[1].strip('>')) for item in indices]
        codes.append(code)
    codes_array = np.array(codes)
    codes_array = np.hstack((codes_array, np.zeros((codes_array.shape[0], 1), dtype=int)))

    unique_codes, counts = np.unique(codes_array, axis=0, return_counts=True)
    duplicates = unique_codes[counts > 1]
    if len(duplicates) > 0:
        print(f"Resolving {len(duplicates)} duplicates...")
        for duplicate in duplicates:
            duplicate_indices = np.where((codes_array == duplicate).all(axis=1))[0]
            for i, idx in enumerate(duplicate_indices):
                codes_array[idx, -1] = i
    new_unique_codes, new_counts = np.unique(codes_array, axis=0, return_counts=True)
    duplicates = new_unique_codes[new_counts > 1]
    if len(duplicates) > 0:
        print(f"⚠ still have duplicates: {len(duplicates)}")
    else:
        print("✅ No duplicates after dedup.")

    print(f"Saving to {output_path}")
    print(f"First 5 codes: {codes_array[:5]}")
    np.save(output_path, codes_array)
