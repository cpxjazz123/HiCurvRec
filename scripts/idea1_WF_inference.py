#!/usr/bin/env python3
"""Idea 1 WF Stage 2.2 inference — 直接重做 forward pass, 兼容 K_l=[256,64,16]

任务18 的 rqidx.pt 用 task4_unified_rerun.py 离线生成 (不依赖 src.inference).
仿造之: 加载 WF ckpt, 抽取 per-layer codebooks (各层 n_clusters 不同),
做 RKMeans forward pass, 保存 rqidx.pt 与碰撞率.
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import src.utils.decorators

import os, json
import numpy as np
import torch

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_2_cf_diagnostic'
os.makedirs(OUT_DIR, exist_ok=True)

EMB_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
WF_CKPT = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/train/runs/idea1_WF_s21_K256_64_16/checkpoints/checkpoint_000_003000.ckpt'


def run_rkmeans_forward(x, codebooks, gains=None):
    """Per the task14_unified_rerun pattern: r_l → argmin ‖r - c‖² → q_l → idx."""
    r_lst = [x.clone()]
    q_lst, idx_lst = [], []
    L = len(codebooks)
    for l in range(L):
        r = r_lst[-1]
        C = codebooks[l]
        if gains is not None and gains[l] is not None:
            eff_C = C * gains[l].unsqueeze(-1) if hasattr(gains[l], 'unsqueeze') else C
        else:
            eff_C = C
        # NN via squared dist trick
        r_norm2 = (r ** 2).sum(-1, keepdim=True)
        c_norm2 = (eff_C ** 2).sum(dim=-1)
        cross = r @ eff_C.T
        d2 = r_norm2 + c_norm2[None, :] - 2 * cross
        idx = d2.argmin(dim=-1)
        q = eff_C[idx]
        q_lst.append(q)
        idx_lst.append(idx)
        r_lst.append(r - q)
    return r_lst, q_lst, idx_lst


def main():
    print('=' * 70)
    print('Idea 1 WF Inference — load ckpt + forward pass + save rqidx')
    print('=' * 70)
    x = torch.load(EMB_PATH, map_location='cpu', weights_only=False).float()
    print(f'  x shape: {x.shape}')

    ck = torch.load(WF_CKPT, map_location='cpu', weights_only=False)
    sd = ck['state_dict']
    codebooks = []
    for l in range(3):
        key = f'quantization_layer_list.{l}.centroids'
        t = sd[key].float().cpu()
        print(f'  L{l+1} codebook shape: {t.shape}')
        codebooks.append(t)
    gains = [None, None, None]  # WF 不使用 gain tracking

    print('\nRunning RKMeans forward pass...')
    r_lst, q_lst, idx_lst = run_rkmeans_forward(x, codebooks, gains)
    for l in range(3):
        print(f'  L{l+1}: idx unique={idx_lst[l].unique().shape[0]}, '
              f'r_residual norm={(r_lst[l+1].norm(dim=-1).mean()):.4f}')

    # SID: 拼 3 层 → (N, 3), 末位 补 dedup (k=256 codes)
    sids = torch.stack([idx_lst[0], idx_lst[1], idx_lst[2]], dim=-1)
    print(f'\n  SIDs shape: {sids.shape}, dtype={sids.dtype}')
    n_unique = sids.unique(dim=0).shape[0]
    n_total = sids.shape[0]
    collision_rate = 1 - n_unique / n_total
    print(f'  unique SIDs: {n_unique}/{n_total} = {n_unique/n_total:.4f}')
    print(f'  collision rate: {collision_rate:.4f}')

    out = {
        'r_lst': r_lst,    # r[0]=x, r[1]=residual after L1, etc.
        'q_lst': q_lst,
        'idx_lst': idx_lst,
        'codebooks': codebooks,
        'gains': gains,
        'sids': sids,
        'K_per_layer': [c.shape[0] for c in codebooks],
        'collision_rate': float(collision_rate),
        'n_unique': int(n_unique),
        'n_total': int(n_total),
    }
    out_path = os.path.join(OUT_DIR, 'idea1_WF_rqidx.pt')
    torch.save(out, out_path)
    print(f'\nSaved → {out_path}')

    # Also save compact diagnostic
    diag = {
        'algo': 'idea1_WF_K_256_64_16',
        'K_per_layer': [int(c.shape[0]) for c in codebooks],
        'N_items': int(x.shape[0]),
        'n_unique_SIDs': int(n_unique),
        'collision_rate': float(collision_rate),
        'r_norm_per_layer': [float(r_lst[l].norm(dim=-1).mean()) for l in range(1, 4)],
        'q_norm_per_layer': [float(q_lst[l].norm(dim=-1).mean()) for l in range(3)],
    }
    diag_path = os.path.join(OUT_DIR, 'idea1_WF_inference_diag.json')
    with open(diag_path, 'w') as f:
        json.dump(diag, f, indent=2)
    print(f'Saved diag → {diag_path}')


if __name__ == '__main__':
    main()