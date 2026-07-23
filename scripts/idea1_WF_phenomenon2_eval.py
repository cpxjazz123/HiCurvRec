#!/usr/bin/env python3
"""Idea 1 现象 2 评估 — WF K_l=[256,64,16] vs AQ K_l=[256,256,256]

加载 WF 训练后的 ckpt, 与 A_baseline 对比:
1. Δ_l (跨层码本失配): WF 是否更平滑?
2. V-info (逐层 mutual info): WF 是否更集中?
3. Collision rate: K=[256,64,16] 实际碰撞率 vs [256,256,256]
4. 端到端 R@10 (需要完整 Stage 3 + 4 训练, 本脚本只做诊断量)

判定:
- WF Δ_l 比 AQ 平滑 (slope < AQ slope) → ✓ WF 有效
- WF V-info 集中度 > AQ (later layers contribute less) → ✓
"""

import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import src.utils.decorators

import os, json
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_2_cf_diagnostic'
os.makedirs(OUT_DIR, exist_ok=True)

# Paths
EMB_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'

# Two checkpoints: A_baseline (AQ) and idea1_WF (extreme)
CKPT_PATHS = {
    'A_AQ_(256,256,256)': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/train/runs/task3_group_a_s21/checkpoints/checkpoint_000_003000.ckpt',
    'Idea1_WF_(256,64,16)': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/train/runs/idea1_WF_s21_K256_64_16/checkpoints/checkpoint_000_003000.ckpt',
}

# RQ idx outputs (post-inference)
RQIDX_PATHS = {
    'A_AQ_(256,256,256)': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt',
    'Idea1_WF_(256,64,16)': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/idea1_WF_rqidx.pt',  # to be created
}

N_MI_SAMPLE = 2000
PCA_RED_DIM = 16


def fast_mi_bits(x, y, n_sample=N_MI_SAMPLE, seed=42, n_neighbors=5):
    if x.ndim == 2 and x.shape[1] > 32:
        x = PCA(n_components=PCA_RED_DIM, random_state=seed).fit_transform(x)
    N = x.shape[0]
    rng = np.random.default_rng(seed)
    if N > n_sample:
        idx = rng.choice(N, size=n_sample, replace=False)
        x_s, y_s = x[idx], y[idx]
    else:
        x_s, y_s = x, y
    mi_nats = mutual_info_classif(x_s, y_s, n_neighbors=n_neighbors,
                                   random_state=seed, discrete_features=False, n_jobs=4)
    return float(mi_nats.mean() / np.log(2))


def build_y(emb_path, K=20):
    x = torch.load(emb_path, map_location='cpu', weights_only=False).float()
    km = MiniBatchKMeans(n_clusters=K, random_state=42, n_init=3, batch_size=1024).fit(x.numpy())
    return km.labels_.astype(np.int64)


def main():
    print('=' * 70)
    print('Idea 1 现象 2 — WF vs AQ 评估')
    print('=' * 70)

    y = build_y(EMB_PATH, K=20)

    # Per-algo analysis
    all_results = {}
    for name, ckpt_p in CKPT_PATHS.items():
        if not os.path.exists(ckpt_p):
            print(f'\n{name}: ckpt not found, skip — {ckpt_p}')
            continue
        print(f'\n--- {name} ---')
        ck = torch.load(ckpt_p, map_location='cpu', weights_only=False)
        sd = ck['state_dict']

        # Per-layer centroid norms and ‖c_l - c_{l-1}‖_F mismatch (Δ_l)
        centroids = []
        for l in range(3):
            key = f'quantization_layer_list.{l}.centroids'
            if key in sd:
                t = sd[key].float().numpy()
                centroids.append(t)
            else:
                print(f'  WARNING: {key} not in state_dict')
                centroids.append(None)
        # Δ_l: norm of L1 centroid vs L2 centroids' mean vector (elbow mismatch proxy)
        for l in range(len(centroids)):
            if centroids[l] is not None:
                avg_norm = float(np.linalg.norm(centroids[l], axis=1).mean())
                print(f'  L{l+1} centroid norm (mean): {avg_norm:.4f}')

        # V-info via RQ idx (Stage 2.2 inference output)
        rqidx_p = RQIDX_PATHS.get(name)
        if rqidx_p and os.path.exists(rqidx_p):
            bundle = torch.load(rqidx_p, map_location='cpu', weights_only=False)
            r_lst = bundle['r_lst']
            sids = bundle.get('sids', None)
            n_unique = int(np.unique(sids.numpy() if torch.is_tensor(sids) else sids, axis=0).shape[0]) if sids is not None else None
            N = sids.shape[0] if sids is not None else r_lst[0].shape[0]
            print(f'  N items: {N}, unique SIDs: {n_unique}/{N} = {(n_unique or 0)/N:.4f}')
            # V-info per layer: I(r_l → Y)
            mi_v_per_layer = []
            for l in range(len(r_lst)):
                r_l = r_lst[l].numpy().astype(np.float32)
                mi_v = fast_mi_bits(r_l, y)
                mi_v_per_layer.append(mi_v)
                print(f'  V-info I(r_{l+1} → Y) = {mi_v:.4f} bits')
            # Δ_l between consecutive layers
            d_l = []
            for l in range(1, len(mi_v_per_layer)):
                d_l.append(mi_v_per_layer[l] - mi_v_per_layer[l-1])
            print(f'  V-info Δ_l (monotonic or not): {[f"{d:+.4f}" for d in d_l]}')
            all_results[name] = {
                'centroid_mean_norm': [float(np.linalg.norm(c, axis=1).mean()) if c is not None else None for c in centroids],
                'V_info_per_layer': mi_v_per_layer,
                'V_info_deltas': d_l,
                'N_items': N,
                'N_unique_SIDs': n_unique,
                'collision_rate': 1 - (n_unique or 0) / N if sids is not None else None,
            }
        else:
            print(f'  No rqidx file found at {rqidx_p}')

    # Verdict
    print('\n' + '=' * 70)
    print('VERDICT — Idea 1 现象 2')
    print('=' * 70)
    if 'A_AQ_(256,256,256)' in all_results and 'Idea1_WF_(256,64,16)' in all_results:
        a = all_results['A_AQ_(256,256,256)']
        w = all_results['Idea1_WF_(256,64,16)']
        # Slope of V-info
        a_slope = np.polyfit(range(len(a['V_info_per_layer'])), a['V_info_per_layer'], 1)[0]
        w_slope = np.polyfit(range(len(w['V_info_per_layer'])), w['V_info_per_layer'], 1)[0]
        print(f'  AQ V-info slope: {a_slope:+.4f} bits/layer')
        print(f'  WF V-info slope: {w_slope:+.4f} bits/layer')
        # WF should be "flatter" (slope close to 0): layer 2/3 contribute less
        if abs(w_slope) < abs(a_slope):
            verdict_v = '✓ WF V-info 更集中 (slope 更小)'
        else:
            verdict_v = '△ WF V-info 与 AQ 相似'
        print(f'  {verdict_v}')
        # Collision rate
        cr_a, cr_w = a.get('collision_rate'), w.get('collision_rate')
        if cr_a is not None and cr_w is not None:
            print(f'  AQ collision rate: {cr_a:.4f}')
            print(f'  WF collision rate: {cr_w:.4f}')
            if cr_w < cr_a:
                verdict_cr = '✓ WF 碰撞率更低'
            else:
                verdict_cr = '△ WF 碰撞率相似或更高'
            print(f'  {verdict_cr}')
    else:
        verdict_v = '数据不全, 跳过判定'
        print(f'  {verdict_v}')

    out_path = os.path.join(OUT_DIR, 'idea1_WF_phenomenon2.json')
    with open(out_path, 'w') as f:
        json.dump({
            'results': all_results,
            'verdict': verdict_v,
        }, f, indent=2)
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()