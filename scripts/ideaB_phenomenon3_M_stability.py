#!/usr/bin/env python3
"""Idea B 现象 3: M 构造稳定性

对比 4 种 M 构造方式下 Idea B 现象 2 的结论是否一致:
1. M_RQ_L1: RQ 码本第 1 层 centroids 协方差
2. M_RQ_all: RQ 码本 3 层 centroids 协方差之和
3. M_stage1: Stage 1 embedding 的样本协方差 (x x.T / N)
4. M_combined: M_RQ_L1 + α M_stage1 (α=0.5)

判定:
- 4 种构造下 causal_gain 方向 (sign) 一致 → 构造稳定
- 方向不一致 (有正有负) → Idea B 判死
"""

import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

# Pre-import to break circular import
import src.utils.decorators

import os, json
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/ideaB_causal'
os.makedirs(OUT_DIR, exist_ok=True)

EMB_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
RQIDX_PATHS = {
    'A_baseline': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt',
    'B_mmq':      '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/B_mmq_rqidx.pt',
    'C_gsrq':     '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/C_gsrq_rqidx.pt',
}
RQ_CKPT_PATHS = {
    'A_baseline': '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/train/runs/task3_group_a_s21/checkpoints/checkpoint_000_003000.ckpt',
    'B_mmq':      '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/train/runs/2026-07-11/13-15-28/checkpoints/checkpoint_000_003000.ckpt',
    'C_gsrq':     '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/train/runs/task3_group_c_s21/checkpoints/checkpoint_000_003000.ckpt',
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


def load_centroids(ckpt_path, layers=(0, 1, 2)):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ck['state_dict']
    out = {}
    for l in layers:
        key = f'quantization_layer_list.{l}.centroids'
        if key not in sd:
            for k in sd:
                if 'centroid' in k.lower() and str(l) in k:
                    key = k
                    break
        t = sd[key].float().numpy()
        if t.shape[0] == 256:
            t = t.T  # (2048, 256)
        out[l] = t
    return out


def make_M_RQ_L1(ckpt):
    cs = load_centroids(ckpt, layers=(0,))
    c = cs[0]
    return c @ c.T + 1e-3 * np.eye(c.shape[0])


def make_M_RQ_all(ckpt):
    cs = load_centroids(ckpt, layers=(0, 1, 2))
    M = sum(c @ c.T for c in cs.values())
    return M + 1e-3 * np.eye(M.shape[0])


def make_M_stage1(emb_path):
    """M = sample covariance of Stage 1 embedding x."""
    x = torch.load(emb_path, map_location='cpu', weights_only=False).float().numpy()
    x_c = x - x.mean(axis=0, keepdims=True)
    M = (x_c.T @ x_c) / x_c.shape[0]
    return M + 1e-3 * np.eye(M.shape[0])


def make_M_combined(ckpt, emb_path, alpha=0.5):
    M1 = make_M_RQ_L1(ckpt)
    M2 = make_M_stage1(emb_path)
    # Normalize traces
    M1_n = M1 / (np.trace(M1) + 1e-12)
    M2_n = M2 / (np.trace(M2) + 1e-12)
    M = M1_n + alpha * M2_n
    return M + 1e-3 * np.eye(M.shape[0])


def decouple_eucl(r_l, q_1):
    beta = np.linalg.pinv(q_1.T @ q_1 + 1e-4 * np.eye(q_1.shape[1])) @ q_1.T @ r_l
    return r_l - q_1 @ beta


def decouple_causal(r_l, q_1, M):
    Mr_l = r_l @ M.T
    Mq_1 = M @ q_1.T
    num = (Mr_l * q_1).sum(axis=1)
    den = (Mq_1 * q_1.T).sum(axis=0)
    proj_coef = num / (den + 1e-12)
    return r_l - proj_coef[:, None] * q_1


def main():
    print('=' * 70)
    print('Building Y and 4 M variants')
    print('=' * 70)
    y = build_y(EMB_PATH, K=20)
    M_variants = {}
    for name, ckpt in RQ_CKPT_PATHS.items():
        M_variants[name] = {}
        M_variants[name]['M_RQ_L1']    = make_M_RQ_L1(ckpt)
        M_variants[name]['M_RQ_all']   = make_M_RQ_all(ckpt)
        M_variants[name]['M_stage1']   = make_M_stage1(EMB_PATH)
        M_variants[name]['M_combined'] = make_M_combined(ckpt, EMB_PATH, alpha=0.5)
        print(f'  {name}:')
        for k, M in M_variants[name].items():
            print(f'    {k}: trace={np.trace(M):.2e}, frobenius={np.linalg.norm(M):.2e}, '
                  f'eig_min={np.linalg.eigvalsh(M).min():.2e}')

    print('\nLoading r_lst, q_1 for A/B/C...')
    bundles = {n: torch.load(p, map_location='cpu', weights_only=False)
               for n, p in RQIDX_PATHS.items()}

    results = {}
    for name in bundles:
        print(f'\n--- {name} ---')
        bundle = bundles[name]
        r_lst = bundle['r_lst']
        q_lst = bundle['q_lst']
        algo_res = {}
        for l in range(1, len(r_lst)):
            r_l = r_lst[l].numpy().astype(np.float32)
            q_1 = q_lst[0].numpy().astype(np.float32)
            mi_base = fast_mi_bits(r_l, y)
            r_perp_e = decouple_eucl(r_l, q_1)
            mi_e = fast_mi_bits(r_perp_e, y)
            delta_e = mi_e - mi_base
            layer_res = {'delta_eucl': delta_e, 'mi_base': mi_base}
            for m_name, M in M_variants[name].items():
                r_perp_m = decouple_causal(r_l, q_1, M)
                mi_m = fast_mi_bits(r_perp_m, y)
                delta_m = mi_m - mi_base
                causal_gain = delta_m - delta_e
                layer_res[m_name] = {
                    'mi': mi_m,
                    'delta_causal': delta_m,
                    'causal_gain_vs_eucl': causal_gain,
                }
                print(f'  L{l} {m_name:14s}: I_V={mi_m:.4f} (Δ={delta_m:+.4f}), '
                      f'gain_vs_E={causal_gain:+.4f}')
            algo_res[f'l{l}'] = layer_res
        results[name] = algo_res

    # Stability check: sign of causal_gain per (algo, layer) across M variants
    print('\n' + '=' * 70)
    print('STABILITY CHECK — sign of causal_gain across M variants')
    print('=' * 70)
    M_NAMES = ['M_RQ_L1', 'M_RQ_all', 'M_stage1', 'M_combined']
    stability = {}
    for name, ar in results.items():
        stability[name] = {}
        for lk, lr in ar.items():
            signs = []
            for mn in M_NAMES:
                g = lr[mn]['causal_gain_vs_eucl']
                signs.append(np.sign(g))
            n_pos = sum(1 for s in signs if s > 0)
            n_neg = sum(1 for s in signs if s < 0)
            n_zero = sum(1 for s in signs if s == 0)
            stable = (n_pos == 4 or n_neg == 4)
            consistency = max(n_pos, n_neg) / 4
            mean_gain = float(np.mean([lr[mn]['causal_gain_vs_eucl'] for mn in M_NAMES]))
            stability[name][lk] = {
                'signs': signs,
                'n_pos': n_pos, 'n_neg': n_neg, 'n_zero': n_zero,
                'consistent': stable,
                'consistency': consistency,
                'mean_gain': mean_gain,
            }
            sign_str = ' '.join(['+' if s > 0 else ('-' if s < 0 else '0') for s in signs])
            verdict = '✓ stable' if stable else '✗ unstable (kill)'
            print(f'  {name} {lk}: signs=[{sign_str}], mean_gain={mean_gain:+.4f}, '
                  f'consistency={consistency:.0%}  {verdict}')

    # Final verdict
    print('\n' + '=' * 70)
    print('FINAL VERDICT — Idea B 现象 3 (M 构造稳定性)')
    print('=' * 70)
    # Per algo overall stability
    overall = {}
    for name in stability:
        # Stable if ALL 3 layers consistent
        all_stable = all(stability[name][lk]['consistent'] for lk in stability[name])
        # Or: at least 2 of 3 layers consistent
        n_stable_layers = sum(1 for lk in stability[name] if stability[name][lk]['consistent'])
        overall[name] = {
            'all_layers_stable': all_stable,
            'n_stable_layers': n_stable_layers,
            'total_layers': len(stability[name]),
        }
        v = '✓ stable' if n_stable_layers >= 2 else '✗ unstable'
        print(f'  {name}: stable layers {n_stable_layers}/{len(stability[name])}  {v}')

    out_path = os.path.join(OUT_DIR, 'ideaB_phenomenon3.json')
    with open(out_path, 'w') as f:
        json.dump({
            'M_variants': M_NAMES,
            'results': results,
            'stability': stability,
            'overall': overall,
        }, f, indent=2)
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()