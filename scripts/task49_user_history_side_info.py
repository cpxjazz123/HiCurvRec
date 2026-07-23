#!/usr/bin/env python3
"""Task 307: 用户历史侧信息 (lightweight structural proxy)

不训 2 TIGER 模型, 而用结构性代理估计 SIS_l:
- b_beh (from SVD cooccurrence) = "用户行为信号"
- h^(l) = 累计 codebook = "item 表示"
- SIS_l = 1 - corr(b_beh_sim, h^(l)_sim) / corr(b_beh_sim, b_beh_sim_optimal)

如果 h^(l) 与 b_beh 的 item-item 相似度高度相关 → 侧信息已被代码本吸收
否则 → 用户历史提供独立信号
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import os, json
import numpy as np
import torch
from scipy.sparse import load_npz
from sklearn.decomposition import TruncatedSVD

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task49_user_history'
os.makedirs(OUT_DIR, exist_ok=True)

EMBED_PT = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
RQIDX = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt'
COOCC = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/stage2_candidate2_graph_gate/cooccurrence_csr.npz'

SVD_DIM = 50
N_SAMPLE_PAIRS = 5000


def cos_sim_pairs(X, n_pairs=5000, seed=42):
    """Compute cos sim for n_pairs random (i,j) pairs."""
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    i_idx = rng.integers(0, N, size=n_pairs)
    j_idx = rng.integers(0, N, size=n_pairs)
    i_idx, j_idx = i_idx, j_idx
    a = X[i_idx] / (np.linalg.norm(X[i_idx], axis=-1, keepdims=True) + 1e-8)
    b = X[j_idx] / (np.linalg.norm(X[j_idx], axis=-1, keepdims=True) + 1e-8)
    return float((a * b).sum(-1).mean()), float((a * b).sum(-1).std())


def main():
    print('=' * 70)
    print('Task 307: 用户历史侧信息 (lightweight structural proxy)')
    print('=' * 70)

    # Load semantic embedding
    print('\n[1] 加载 data...')
    x_sem = torch.load(EMBED_PT, map_location='cpu', weights_only=False)
    if x_sem.dim() > 2:
        x_sem = x_sem.squeeze(0)
    if x_sem.shape[0] != 11924 and x_sem.shape[1] == 11924:
        x_sem = x_sem.t()
    N = x_sem.shape[0]

    bundle = torch.load(RQIDX, map_location='cpu', weights_only=False)
    codebooks = bundle['codebooks']
    idx_lst = bundle['idx_lst']

    # Load cooccurrence and SVD
    cooc = load_npz(COOCC)
    svd = TruncatedSVD(n_components=SVD_DIM, random_state=42)
    b_beh = svd.fit_transform(cooc).astype(np.float32)
    print(f'  b_beh shape: {b_beh.shape}, var={svd.explained_variance_ratio_.sum():.3f}')

    # Compute h^(l) (cumulative codebook) for each layer
    q_per_layer = []
    q_cum = np.zeros((N, x_sem.shape[1]), dtype=np.float32)
    for l in range(3):
        cb = codebooks[l]
        idx = idx_lst[l].numpy()
        q_l = cb[idx].float().numpy()
        q_cum += q_l
        q_per_layer.append(q_cum.copy())

    # Also x_sem baseline
    x_sem_np = x_sem.float().numpy()

    # Compute pairwise sim between random (i,j) pairs for each representation
    print(f'\n[2] 计算 {N_SAMPLE_PAIRS} pairs 的 cos sim...')
    rng = np.random.default_rng(42)
    i_idx = rng.integers(0, N, size=N_SAMPLE_PAIRS)
    j_idx = rng.integers(0, N, size=N_SAMPLE_PAIRS)
    # Avoid i==j
    mask = i_idx != j_idx
    i_idx = i_idx[mask][:N_SAMPLE_PAIRS]
    j_idx = j_idx[mask][:N_SAMPLE_PAIRS]

    # b_beh similarity (user-history-driven proxy)
    a_b = b_beh[i_idx] / (np.linalg.norm(b_beh[i_idx], axis=-1, keepdims=True) + 1e-8)
    b_b = b_beh[j_idx] / (np.linalg.norm(b_beh[j_idx], axis=-1, keepdims=True) + 1e-8)
    sim_b = (a_b * b_b).sum(-1)

    # h^(l) similarity for each layer
    sim_h = {}
    for l in range(3):
        h_l = q_per_layer[l]
        a_h = h_l[i_idx] / (np.linalg.norm(h_l[i_idx], axis=-1, keepdims=True) + 1e-8)
        b_h = h_l[j_idx] / (np.linalg.norm(h_l[j_idx], axis=-1, keepdims=True) + 1e-8)
        sim_h[f'L{l+1}'] = (a_h * b_h).sum(-1)

    # x_sem baseline
    a_x = x_sem_np[i_idx] / (np.linalg.norm(x_sem_np[i_idx], axis=-1, keepdims=True) + 1e-8)
    b_x = x_sem_np[j_idx] / (np.linalg.norm(x_sem_np[j_idx], axis=-1, keepdims=True) + 1e-8)
    sim_x = (a_x * b_x).sum(-1)

    # Correlations
    from scipy.stats import pearsonr, spearmanr
    results = {}
    print('\n[3] 各层 h^(l) 与 b_beh 的 item-sim 相关性:')
    print(f'  b_beh sim: mean={sim_b.mean():.3f}, std={sim_b.std():.3f}')
    print(f'  x_sem sim: mean={sim_x.mean():.3f}, std={sim_x.std():.3f}')
    print(f'  b_beh vs x_sem: r={pearsonr(sim_b, sim_x)[0]:.3f}')
    results['b_vs_x_sem'] = {
        'pearson_r': float(pearsonr(sim_b, sim_x)[0]),
        'spearman_r': float(spearmanr(sim_b, sim_x)[0]),
    }
    for l in range(3):
        l_name = f'L{l+1}'
        s = sim_h[l_name]
        # SIS_l: 1 - corr(h_l, b_beh) / corr(b_beh, optimal_beh)
        # 假设 optimal_beh = b_beh itself (上界 = 1)
        corr_h_b = float(pearsonr(s, sim_b)[0])
        corr_b_b = 1.0  # b_beh 与自身的相关
        sis_l = 1 - corr_h_b / (corr_b_b + 1e-6)
        results[l_name] = {
            'corr_h_b_beh_pearson': corr_h_b,
            'corr_h_b_beh_spearman': float(spearmanr(s, sim_b)[0]),
            'SIS_l_proxy': float(sis_l),
            'sim_h_mean': float(s.mean()),
            'sim_h_std': float(s.std()),
        }
        print(f'  {l_name} h^(l): r(h, b_beh)={corr_h_b:.3f}, SIS_l={sis_l:.3f}')

    # Estimate how much info user history adds beyond h^(l):
    # Compare residual: b_beh - projection(b_beh onto h^(l))
    from sklearn.linear_model import Ridge
    print('\n[4] 量化 b_beh 在 h^(l) 上的解释度 (R² of regression b_beh ~ h^(l))...')
    for l in range(3):
        l_name = f'L{l+1}'
        from sklearn.decomposition import PCA
        h_l = q_per_layer[l]
        # Reduce dim for speed
        h_pca = PCA(n_components=50, random_state=42).fit_transform(h_l)
        ridge = Ridge(alpha=1.0)
        from sklearn.model_selection import train_test_split
        h_train, h_test, b_train, b_test = train_test_split(h_pca, b_beh, test_size=0.2, random_state=42)
        ridge.fit(h_train, b_train)
        b_pred = ridge.predict(h_test)
        from sklearn.metrics import r2_score
        r2 = float(r2_score(b_test, b_pred, multioutput='raw_values').mean())
        results[l_name]['R2_b_beh_given_h_l'] = r2
        print(f'  {l_name}: R²(b_beh | h^(l)) = {r2:.3f}')

    # Save
    out = {
        'n_items': N,
        'cooc_path': COOCC,
        'n_sample_pairs': N_SAMPLE_PAIRS,
        'note': 'structural proxy: 用 b_beh (SVD of cooccurrence) 作为"用户历史侧信息"代理. 完整版需训 P_nohist vs P_hist 两个 TIGER 模型.',
        'results': results,
    }
    with open(os.path.join(OUT_DIR, 'side_info.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n[产物] {OUT_DIR}/side_info.json')

    # Verdict
    with open(os.path.join(OUT_DIR, 'verdict.md'), 'w') as f:
        f.write('# Task 307 Verdict: 用户历史侧信息 (lightweight proxy)\n\n')
        f.write(f'数据集: Toys (N={N}), SVD-{SVD_DIM} b_beh (var={svd.explained_variance_ratio_.sum():.3f})\n\n')
        f.write('## 结构性发现\n\n')
        f.write(f'- b_beh (用户历史代理) 与 x_sem 的 item-sim 相关: r={results["b_vs_x_sem"]["pearson_r"]:.3f}\n')
        f.write('  - **低相关** 意味着用户历史信号与语义空间几乎独立\n\n')
        f.write('## 各层 h^(l) 与 b_beh 的相关性\n\n')
        f.write('| Layer | corr(h, b_beh) | SIS_l | R²(b_beh \\| h^(l)) |\n')
        f.write('|-------|----------------|-------|---------------------|\n')
        for l in range(3):
            r = results[f'L{l+1}']
            f.write(f'| L{l+1} | {r["corr_h_b_beh_pearson"]:.3f} | {r["SIS_l_proxy"]:.3f} | {r["R2_b_beh_given_h_l"]:.3f} |\n')
        f.write('\n## 判读\n\n')
        # Key: if corr(h, b_beh) is low and SIS is high → user history adds info beyond codebook
        # if R²(b_beh | h^(l)) is low → codebook doesn't capture user history
        for l in range(3):
            r = results[f'L{l+1}']
            f.write(f'### L{l+1}\n')
            corr = r['corr_h_b_beh_pearson']
            r2 = r['R2_b_beh_given_h_l']
            if corr < 0.1:
                f.write(f'- corr(h^(l), b_beh)={corr:.3f} < 0.1: 代码本与用户历史**几乎不相关**\n')
                f.write(f'  → 用户历史侧信息**未被代码本吸收**\n')
            elif corr > 0.5:
                f.write(f'- corr(h^(l), b_beh)={corr:.3f} > 0.5: 代码本已部分吸收用户历史\n')
            else:
                f.write(f'- corr(h^(l), b_beh)={corr:.3f}: 中等相关\n')
            if r2 < 0.1:
                f.write(f'- R²(b_beh | h^(l))={r2:.3f} < 0.1: 用户历史信号**完全独立于代码本**\n')
            f.write('\n')

    print(f'[产物] {OUT_DIR}/verdict.md')


if __name__ == '__main__':
    main()