#!/usr/bin/env python3
"""Task #260 — Issue #10 方向 A 准备: Sinkhorn 强度曲线预扫.

加载 Task #84 baseline vanilla codebook ckpt (1 次 GPU 加载), 扫 max_sinkhorn_iters
∈ {0, 5, 10, 20, 30}, 输出 collision + utilization 曲线. Stage 2 推断 (CPU forward) 多次,
不跑 Stage 3 训练 (R11.4 用户决策).
"""
import sys, os, json, argparse
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec')

import torch
import numpy as np
from torch.utils.data import DataLoader
from model.utils import EmbDataset
from train_hrqvae import HRQVAE

CKPT = '/home/wlia0047/ar57/wenyu/GeneRec/products/task84/ckpt/Instruments/Jul-23-2026_20-08-06_beta_1.000_codebook_[64,128,256]_sk_0.000/best_loss_model.pth'
DATA_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet'
OUT_JSON = '/home/wlia0047/ar57/wenyu/GeneRec/verdicts/task260_issue10_sinkhorn_strength_sweep.json'

MAX_ITERS_LIST = [0, 5, 10, 20, 30]
DEVICE = 'cuda:0'

def check_collision(str_array):
    return len(set(str_array.tolist())) == len(str_array)

def get_collision_item(str_array):
    seen = {}
    groups = []
    for i, s in enumerate(str_array.tolist()):
        if s in seen:
            groups.append([seen[s], i])
            seen[s] = i
        else:
            seen[s] = i
    grouped = {}
    for a, b in groups:
        grouped.setdefault(a, [a]).append(b)
    return list(grouped.values())

def main():
    log = []
    log.append(f"[Task #260] ckpt={CKPT}")
    log.append(f"[Task #260] data={DATA_PATH}, iters={MAX_ITERS_LIST}")
    print('\n'.join(log))

    # 1) load ckpt once
    ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict']
    ckpt_args = vars(ckpt['args']) if not isinstance(ckpt['args'], dict) else ckpt['args']
    print(f"  num_emb_list={ckpt_args['num_emb_list']}, e_dim={ckpt_args['e_dim']}, "
          f"beta={ckpt_args['beta']}, product_manifold={ckpt_args.get('product_manifold', False)}")

    data = EmbDataset(ckpt_args['data_path'])
    print(f"  Data: {len(data)} items, dim={data.dim}")

    # 2) build model
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
        assignment_mode=ckpt_args.get('assignment_mode', 'shared'),
        c_k_min=ckpt_args.get('c_k_min', 0.5),
        c_k_max=ckpt_args.get('c_k_max', 20.0),
        c_k_seed=ckpt_args.get('c_k_seed', 42),
    )
    model.load_state_dict(state_dict, strict=False)
    model = model.to(DEVICE)
    model.eval()

    data_loader = DataLoader(data, num_workers=2, batch_size=64, shuffle=False, pin_memory=True)

    # 3) first forward: no Sinkhorn (max_iters=0 baseline)
    all_indices = []
    all_indices_str = []
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]

    print("\n=== Forward pass (1×, shared by all iters) ===")
    with torch.no_grad():
        for d in data_loader:
            d = d.to(DEVICE)
            indices = model.get_indices(d, use_sk=False)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for index in indices:
                code = [prefix[i].format(int(ind)) for i, ind in enumerate(index)]
                all_indices.append(code)
                all_indices_str.append(str(code))

    all_indices = np.array(all_indices)
    all_indices_str = np.array(all_indices_str)
    print(f"  Forward done: {len(all_indices)} items")

    # 4) Snapshot baseline (max_iters=0)
    results = {}
    for target_iters in MAX_ITERS_LIST:
        print(f"\n=== max_iters = {target_iters} ===")
        ai = all_indices.copy()
        ais = all_indices_str.copy()

        tt = 0
        while tt < target_iters and not check_collision(ais):
            collision_groups = get_collision_item(ais)
            print(f"  Sinkhorn iter {tt}: {len(collision_groups)} collision groups")
            for collision_items in collision_groups:
                d = data[collision_items].to(DEVICE)
                with torch.no_grad():
                    indices = model.get_indices(d, use_sk=True)
                indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
                for item, index in zip(collision_items, indices):
                    code = [prefix[i].format(int(ind)) for i, ind in enumerate(index)]
                    ai[item] = code
                    ais[item] = str(code)
            tt += 1

        tot_item = len(ais)
        tot_unique = len(set(ais.tolist()))
        collision_rate = (tot_item - tot_unique) / tot_item

        # decode back to int array
        codes = []
        for code in ai:
            codes.append([int(t.split('_')[1].strip('>')) for t in code])
        codes_array = np.array(codes)

        # 4-digit dedup (mimic task223 behavior)
        codes_array = np.hstack((codes_array, np.zeros((codes_array.shape[0], 1), dtype=int)))
        unique_codes, counts = np.unique(codes_array, axis=0, return_counts=True)
        duplicates = unique_codes[counts > 1]
        for dup in duplicates:
            dup_idx = np.where((codes_array == dup).all(axis=1))[0]
            for i, idx in enumerate(dup_idx):
                codes_array[idx, -1] = i

        new_unique_codes, new_counts = np.unique(codes_array, axis=0, return_counts=True)
        n_unique_sid_post = len(new_unique_codes)

        # per-layer utilization
        per_layer = {}
        for li in range(len(ckpt_args['num_emb_list'])):
            layer_unique = len(np.unique(codes_array[:, li]))
            layer_total = ckpt_args['num_emb_list'][li]
            per_layer[f'L{li}'] = {
                'unique': layer_unique,
                'total': layer_total,
                'utilization': layer_unique / layer_total,
            }

        results[target_iters] = {
            'n_items': tot_item,
            'n_unique_pre_resolve': tot_unique,
            'collision_rate_pre_resolve': collision_rate,
            'n_unique_post_resolve': n_unique_sid_post,
            'collision_rate_post_resolve': (tot_item - n_unique_sid_post) / tot_item,
            'iters_actually_run': tt,
            'per_layer_utilization': per_layer,
        }
        print(f"  → collision_pre={collision_rate:.4f}, n_unique_post={n_unique_sid_post}, iters_run={tt}")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n=== Saved to {OUT_JSON} ===")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()