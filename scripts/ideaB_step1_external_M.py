#!/usr/bin/env python3
"""第一步: 排除循环论证 — 用外部固定 M 重新测 causal_gain

核心问题: 之前 M 是从 C_gsrq 自己的码本协方差构造的, 等于用自己的尺子量自己。

设计:
1. 固定 M_from_A_baseline, 测 A/B/C 三组 causal_gain
2. 固定 M_from_C_gsrq, 测 A/B/C 三组 causal_gain (交叉检验)

判定:
- 如果 C 在 "用 A 的尺子" 时 causal_gain 仍 ≈+0.08 → 真实优势
- 如果 C 在 "用 A 的尺子" 时 causal_gain 跌回 ~0 → 循环论证, Idea B 判死
- 反向: 如果在 "用 C 的尺子" 下 A/B 也表现出高 causal_gain → 谁用谁的 M 谁就赢 = 同义反复
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


def load_M_from_ckpt(ckpt_path, layer=0):
    """M = centroids.T @ centroids (D x D). Standardize: t shape (D, K)."""
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ck['state_dict']
    key = f'quantization_layer_list.{layer}.centroids'
    if key not in sd:
        for k in sd:
            if 'centroid' in k.lower() and str(layer) in k:
                key = k
                break
    t = sd[key].float().numpy()
    if t.shape[0] == 256:
        t = t.T  # (2048, 256)
    return t @ t.T + 1e-3 * np.eye(t.shape[0])


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
    print('Step 1 — External M (排除循环论证)')
    print('=' * 70)
    y = build_y(EMB_PATH, K=20)

    # Step 1a: Build 2 EXTERNAL M's: M_from_A and M_from_C
    M_external = {
        'M_from_A_baseline': load_M_from_ckpt(RQ_CKPT_PATHS['A_baseline'], layer=0),
        'M_from_C_gsrq':     load_M_from_ckpt(RQ_CKPT_PATHS['C_gsrq'],     layer=0),
    }
    for name, M in M_external.items():
        print(f'  {name}: trace={np.trace(M):.2e}, frobenius={np.linalg.norm(M):.2e}, '
              f'eig_max={np.linalg.eigvalsh(M).max():.2e}, '
              f'eig_min={np.linalg.eigvalsh(M).min():.2e}')

    # Also: M_self for comparison (each algo's own M)
    M_self = {
        name: load_M_from_ckpt(ckpt, layer=0) for name, ckpt in RQ_CKPT_PATHS.items()
    }

    print('\nLoading r_lst, q_1 for A/B/C...')
    bundles = {n: torch.load(p, map_location='cpu', weights_only=False)
               for n, p in RQIDX_PATHS.items()}

    # M sources to test
    M_sources = {
        'M_self_A_baseline': M_self['A_baseline'],
        'M_self_B_mmq':      M_self['B_mmq'],
        'M_self_C_gsrq':     M_self['C_gsrq'],
        'M_external_A':      M_external['M_from_A_baseline'],
        'M_external_C':      M_external['M_from_C_gsrq'],
    }

    results = {}
    for algo in ['A_baseline', 'B_mmq', 'C_gsrq']:
        print(f'\n--- {algo} ---')
        bundle = bundles[algo]
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
            for m_name, M in M_sources.items():
                r_perp_m = decouple_causal(r_l, q_1, M)
                mi_m = fast_mi_bits(r_perp_m, y)
                delta_m = mi_m - mi_base
                causal_gain = delta_m - delta_e
                layer_res[m_name] = {
                    'mi': mi_m,
                    'delta_causal': delta_m,
                    'causal_gain_vs_eucl': causal_gain,
                }
                print(f'  L{l} {m_name:24s}: Δ={delta_m:+.4f}, gain={causal_gain:+.4f}')
            algo_res[f'l{l}'] = layer_res
        results[algo] = algo_res

    # Verdict: the key table — for each (algo, M_source), causal_gain L1
    print('\n' + '=' * 70)
    print('CRITICAL TABLE — causal_gain for each (algo × M_source) at L1')
    print('=' * 70)
    headers = ['algo \\ M_source'] + list(M_sources.keys())
    print(f'  {"":25s} ' + ' '.join(f'{h:>20s}' for h in headers[1:]))
    table = {}
    for algo in ['A_baseline', 'B_mmq', 'C_gsrq']:
        row = [algo]
        cells = {}
        for m_name in M_sources:
            g = results[algo]['l1'][m_name]['causal_gain_vs_eucl']
            row.append(f'{g:+.4f}')
            cells[m_name] = g
        print(f'  {algo:25s} ' + ' '.join(f'{c:>20s}' for c in row[1:]))
        table[algo] = cells

    # Key check: C_gsrq with M_external_A
    c_with_ext_a = table['C_gsrq']['M_external_A']
    c_with_self_c = table['C_gsrq']['M_self_C_gsrq']
    print(f'\n  KEY CHECK: C_gsrq L1 causal_gain:')
    print(f'    with M_self_C_gsrq (loop):     {c_with_self_c:+.4f}')
    print(f'    with M_external_A_baseline:    {c_with_ext_a:+.4f}')
    print(f'    drop ratio: {(c_with_ext_a / c_with_self_c):.2%}')

    # Cross check: A with M_external_C
    a_with_ext_c = table['A_baseline']['M_external_C']
    a_with_self_a = table['A_baseline']['M_self_A_baseline']
    print(f'\n  CROSS CHECK: A_baseline L1 causal_gain:')
    print(f'    with M_self_A_baseline:        {a_with_self_a:+.4f}')
    print(f'    with M_external_C_gsrq:        {a_with_ext_c:+.4f}')

    # Final verdict
    print('\n' '=' * 70)
    print('VERDICT — Idea B Step 1 (循环论证检验)')
    print('=' * 70)
    # If C drops more than 50% with external M → circular
    drop_ratio = (c_with_ext_a / (c_with_self_c + 1e-12))
    if drop_ratio < 0.5:
        verdict = '✗ CIRCULAR (C 优势用自己 M 才能测出, 换 A 的 M 跌回)'
    elif abs(a_with_ext_c) > 0.05:
        verdict = '✗ 谁用谁的 M 谁就赢 (同义反复)'
    else:
        verdict = '✓ causal_gain 是真实优势 (不依赖 M 来源)'
    print(f'  {verdict}')

    out_path = os.path.join(OUT_DIR, 'ideaB_step1_external_M.json')
    with open(out_path, 'w') as f:
        json.dump({
            'M_sources': list(M_sources.keys()),
            'results': results,
            'critical_table': table,
            'C_with_self_C': c_with_self_c,
            'C_with_external_A': c_with_ext_a,
            'drop_ratio_C': drop_ratio,
            'A_with_self_A': a_with_self_a,
            'A_with_external_C': a_with_ext_c,
            'verdict': verdict,
        }, f, indent=2)
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()