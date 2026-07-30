"""
Task #336 — Issue #43 Gate 2b Stage 2 SID inference.

Loads best_collision HRQVAEWithHypPre checkpoint from Stage 1, runs
Sinkhorn-Knopp decoding (up to 30 rounds), resolves duplicates with 4th digit,
outputs (9922, 4) token array for Stage 3 T5 training.

Usage:
  python3 scripts/task336_issue43_gate2b_stage2_infer.py
"""
import collections
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys, os
HGREC = "/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec"
sys.path.insert(0, HGREC)
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/GeneRec/scripts")

from model.utils import EmbDataset
from model.hrqvae import HRQVAE
from task334_issue43_gate2a_hyp_pre_encoder import HRQVAEWithHypPre


def check_collision(all_indices_str):
    return len(all_indices_str) == len(set(all_indices_str.tolist()))


def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)
    return [v for v in index2id.values() if len(v) > 1]


if __name__ == "__main__":
    dataset = "Instruments"
    ckpt_base = "/home/wlia0047/ar57/wenyu/GeneRec/products/task336/stage1/Instruments"
    ckpt_pattern = f"{ckpt_base}/*/best_collision_model.pth"
    candidates = sorted(glob.glob(ckpt_pattern))
    if not candidates:
        raise FileNotFoundError(f"No best_collision_model.pth in {ckpt_pattern}")
    ckpt_path = candidates[-1]
    output_path = f"/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/{dataset}/{dataset}_t5_hrqvae_hyp_pre.npy"
    device = torch.device("cuda:1")  # GPU 1 (R7: GPU 0 free for other use)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=torch.device('cpu'))
    args = ckpt['args']
    state_dict = ckpt['state_dict']

    data = EmbDataset(args.data_path)
    print(f"  Dataset: {len(data)} items, dim={data.dim}")

    # Build base HRQVAE + wrap with HypPreEncoder (same as Stage 1 training)
    base = HRQVAE(
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
    )
    model = HRQVAEWithHypPre(base, c=0.74, enabled=True)

    # Load state_dict directly (has `base.` prefix — matches HRQVAEWithHypPre keys)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"  Model loaded (HypPreEncoder ENABLED, c=0.74)")

    loader = DataLoader(data, num_workers=4, batch_size=64, shuffle=False, pin_memory=True)

    all_indices = []
    all_indices_str = []
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]

    # Phase 1: argmin (no Sinkhorn)
    print("Phase 1: argmin encoding...")
    for d in tqdm(loader):
        d = d.to(device)
        indices = model.get_indices(d, use_sk=False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for index in indices:
            code = [prefix[i].format(int(ind)) for i, ind in enumerate(index)]
            all_indices.append(code)
            all_indices_str.append(str(code))

    all_indices = np.array(all_indices)
    all_indices_str = np.array(all_indices_str)

    # Phase 2: resolve collisions with Sinkhorn (up to 30 rounds)
    print("Phase 2: Sinkhorn collision resolution...")
    for tt in range(30):
        if check_collision(all_indices_str):
            print(f"  All collisions resolved at round {tt}")
            break

        groups = get_collision_item(all_indices_str)
        print(f"  Round {tt}: {len(groups)} collision groups")
        for collision_items in groups:
            d = data[collision_items].to(device)
            indices = model.get_indices(d, use_sk=True)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for item, index in zip(collision_items, indices):
                code = [prefix[i].format(int(ind)) for i, ind in enumerate(index)]
                all_indices[item] = code
                all_indices_str[item] = str(code)

    print(f"Total indices: {len(all_indices)}")
    all_indices_dict = {item: list(indices) for item, indices in enumerate(all_indices.tolist())}
    codes = [[int(v.split('_')[1].strip('>')) for v in code] for code in all_indices_dict.values()]
    codes_array = np.array(codes)

    # Phase 3: 4th-digit dedup
    codes_array = np.hstack((codes_array, np.zeros((codes_array.shape[0], 1), dtype=int)))
    unique_codes, counts = np.unique(codes_array, axis=0, return_counts=True)
    duplicates = unique_codes[counts > 1]
    if len(duplicates) > 0:
        print(f"Resolving {len(duplicates)} duplicate codes via 4th digit...")
        for dup in duplicates:
            dup_idx = np.where((codes_array == dup).all(axis=1))[0]
            for i, idx in enumerate(dup_idx):
                codes_array[idx, -1] = i

    final_unique, final_counts = np.unique(codes_array, axis=0, return_counts=True)
    final_dups = final_unique[final_counts > 1]
    if len(final_dups) > 0:
        print(f"  ⚠️ {len(final_dups)} duplicates REMAIN after 4th-digit dedup")
    else:
        print(f"  ✅ All duplicates resolved")

    print(f"Saving to {output_path}")
    print(f"  First 5 codes: {codes_array[:5]}")
    np.save(output_path, codes_array)
    print("Stage 2 complete.")
