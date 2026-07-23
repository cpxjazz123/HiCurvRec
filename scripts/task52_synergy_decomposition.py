#!/usr/bin/env python3
"""Task 310: 协同信息诊断 (lightweight structural proxy, 4 模型 V 分解)

不训 4 TIGER 模型, 而用 4 个 item-level 表示 + linear probe 估计 V 分解:
- M_0: random baseline
- M_A: semantic (FLAN-T5)
- M_B: collaborative (SVD b_beh)
- M_AB: 累计 codebook (semantic + cooc via item-id grouping)

Target: popularity (cooc row sum) 作为 proxy for "next item click"

V 分解:
- V_B|A = V(M_AB) - V(M_A)
- Syn_l = V(M_AB) - max(V(M_A), V(M_B))
- V_total = V(M_AB) - V(M_0)
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import os, json
import numpy as np
import torch
from scipy.sparse import load_npz
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task52_synergy'
os.makedirs(OUT_DIR, exist_ok=True)

EMBED_PT = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
RQIDX = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt'
COOCC = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/stage2_candidate2_graph_gate/cooccurrence_csr.npz'

SVD_DIM = 50
PCA_DIM = 50


def value_r2(X_feat, y_target):
    """Compute V (model value) = R² of Ridge regression on (X_feat -> y_target)."""
    if X_feat.ndim == 1:
        X_feat = X_feat.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X_feat, y_target, test_size=0.2, random_state=42)
    m = Ridge(alpha=1.0)
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    if y_target.ndim == 1:
        return float(r2_score(y_test, y_pred))
    else:
        return float(r2_score(y_test, y_pred, multioutput='raw_values').mean())


def main():
    print('=' * 70)
    print('Task 310: 协同信息诊断 (lightweight V decomposition proxy)')
    print('=' * 70)

    # Load data
    print('\n[1] 加载 data...')
    x_sem = torch.load(EMBED_PT, map_location='cpu', weights_only=False)
    if x_sem.dim() > 2:
        x_sem = x_sem.squeeze(0)
    if x_sem.shape[0] != 11924 and x_sem.shape[1] == 11924:
        x_sem = x_sem.t()
    N = x_sem.shape[0]
    x_sem_np = x_sem.float().numpy()

    bundle = torch.load(RQIDX, map_location='cpu', weights_only=False)
    codebooks = bundle['codebooks']
    idx_lst = bundle['idx_lst']

    cooc = load_npz(COOCC)
    svd = TruncatedSVD(n_components=SVD_DIM, random_state=42)
    b_beh = svd.fit_transform(cooc).astype(np.float32)
    print(f'  b_beh shape: {b_beh.shape}, var={svd.explained_variance_ratio_.sum():.3f}')

    # Target: popularity (cooc row sum) + log-transformed
    pop = np.log1p(np.array(cooc.sum(axis=1)).flatten()).astype(np.float32)

    # Cumulative codebook for h^(l)
    q_per_layer = []
    q_cum = np.zeros((N, x_sem_np.shape[1]), dtype=np.float32)
    for l in range(3):
        cb = codebooks[l]
        idx = idx_lst[l].numpy()
        q_l = cb[idx].float().numpy()
        q_cum += q_l
        q_per_layer.append(q_cum.copy())

    # 4 模型
    print('\n[2] 构造 4 模型...')
    # M_0: random features (no info)
    rng = np.random.default_rng(42)
    M_0 = rng.standard_normal((N, PCA_DIM)).astype(np.float32)

    # M_A: semantic (FLAN-T5 reduced)
    pca_sem = PCA(n_components=PCA_DIM, random_state=42).fit_transform(x_sem_np)
    M_A = pca_sem.astype(np.float32)

    # M_B: collaborative (SVD b_beh)
    M_B = b_beh.astype(np.float32)

    # M_AB: cumulative codebook (h^(l) for layer l)
    M_AB_per_layer = []
    for l in range(3):
        h_l = q_per_layer[l]
        h_pca = PCA(n_components=PCA_DIM, random_state=42).fit_transform(h_l)
        M_AB_per_layer.append(h_pca.astype(np.float32))

    # Compute V for each model
    print('\n[3] 计算 V 各模型 (target=log popularity)...')
    V_0 = value_r2(M_0, pop)
    V_A = value_r2(M_A, pop)
    V_B = value_r2(M_B, pop)
    V_AB_per_layer = []
    for l in range(3):
        V_AB = value_r2(M_AB_per_layer[l], pop)
        V_AB_per_layer.append(V_AB)

    # 4 个互补指标
    print('\n[4] 4 协同指标:')
    V_total = V_AB_per_layer[2] - V_0
    V_A_share = V_A - V_0
    V_B_share = V_B - V_0
    V_A_given_B = max(V_A, V_B) - V_0
    V_B_given_A_per_layer = []
    Syn_per_layer = []
    for l in range(3):
        V_AB = V_AB_per_layer[l]
        # V_B|A = V_AB - V_A (semantic baseline)
        V_b_given_a = V_AB - V_A
        V_B_given_A_per_layer.append(V_b_given_a)
        # Syn = V_AB - max(V_A, V_B)
        syn = V_AB - max(V_A, V_B)
        Syn_per_layer.append(syn)

    print(f'  V_0 (random):   {V_0:.4f}')
    print(f'  V_A (semantic): {V_A:.4f}')
    print(f'  V_B (collab):   {V_B:.4f}')
    for l in range(3):
        print(f'  V_AB L{l+1}:    {V_AB_per_layer[l]:.4f}')
        print(f'  V_B|A  L{l+1}:  {V_B_given_A_per_layer[l]:.4f}')
        print(f'  Syn    L{l+1}:  {Syn_per_layer[l]:.4f}')

    # Save
    out = {
        'n_items': N,
        'cooc_path': COOCC,
        'svd_dim': SVD_DIM,
        'target': 'log(popularity from cooc row sum)',
        'note': 'structural proxy: 4 模型用 4 类 item-level 表示. 完整版需训 4 TIGER 模型.',
        'V_models': {
            'V_0_random': V_0,
            'V_A_semantic': V_A,
            'V_B_collaborative': V_B,
            'V_AB_per_layer': V_AB_per_layer,
        },
        'V_B_given_A_per_layer': V_B_given_A_per_layer,
        'Syn_per_layer': Syn_per_layer,
    }
    with open(os.path.join(OUT_DIR, 'V_breakdown.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n[产物] {OUT_DIR}/V_breakdown.json')

    # Verdict
    with open(os.path.join(OUT_DIR, 'verdict.md'), 'w') as f:
        f.write('# Task 310 Verdict: 协同信息诊断 (lightweight V decomposition)\n\n')
        f.write(f'数据集: Toys (N={N}), target = log(popularity)\n\n')
        f.write('## 4 模型 V (model value = R² predicting popularity)\n\n')
        f.write('| Model | V (R²) |\n')
        f.write('|-------|--------|\n')
        f.write(f'| M_0 (random) | {V_0:.4f} |\n')
        f.write(f'| M_A (semantic) | {V_A:.4f} |\n')
        f.write(f'| M_B (collaborative) | {V_B:.4f} |\n')
        for l in range(3):
            f.write(f'| M_AB (L{l+1} codebook) | {V_AB_per_layer[l]:.4f} |\n')
        f.write('\n## 协同 / 冗余 / 增益分解\n\n')
        f.write('| Layer | V_B\\|A (collab given semantic) | Syn (AB - max(A,B)) |\n')
        f.write('|-------|--------------------------------|----------------------|\n')
        for l in range(3):
            f.write(f'| L{l+1} | {V_B_given_A_per_layer[l]:.4f} | {Syn_per_layer[l]:.4f} |\n')
        f.write('\n## 判读\n\n')
        # Compare V_A vs V_B (which contributes more)
        if V_A > V_B:
            dominant = 'semantic'
            dominant_v = V_A
        else:
            dominant = 'collaborative'
            dominant_v = V_B
        f.write(f'### 主导信号\n')
        f.write(f'- V_A (semantic) = {V_A:.4f}, V_B (collaborative) = {V_B:.4f}\n')
        f.write(f'- **{dominant}** 占主导 (V={dominant_v:.4f})\n\n')

        # Synergy analysis
        for l in range(3):
            syn = Syn_per_layer[l]
            v_b_a = V_B_given_A_per_layer[l]
            f.write(f'### L{l+1}\n')
            f.write(f'- Syn = {syn:.4f}\n')
            f.write(f'- V_B|A = {v_b_a:.4f}\n')
            if syn > 0.05:
                f.write(f'  → **Synergy positive**: 累计 codebook 提供的 value 超过 A 或 B 各自\n')
            elif syn < -0.05:
                f.write(f'  → **Negative interaction**: A+B 反而不如 max(A, B)\n')
            else:
                f.write(f'  → **Additive**: A+B ≈ max(A, B), 无明显 synergy\n')
            if v_b_a < 0.02:
                f.write(f'  → 协同信号 V_B|A 极小 → **协同未真正被吸收**\n')
            f.write('\n')

    print(f'[产物] {OUT_DIR}/verdict.md')


if __name__ == '__main__':
    main()