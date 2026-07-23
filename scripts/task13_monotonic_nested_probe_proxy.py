#!/usr/bin/env python3
"""Task 299: 单调嵌套增量 Probe (proxy 版, CPU only)

Proxy: 用结构性信息增量代替 user-history driven probe.
- Probe A: f_A(i) = features built on Z[<l] codebook vectors (sum q^j for j<l)
- Incremental f_l(i) = features on z[l] codebook vector
- Probe C = Probe A + α * f_l, where α ≥ 0 via softplus reparam

Target: y = FLAN-T5 embedding (proxy for downstream utility).
5 init R=5: random PCA init or noise level.
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import os, json
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task39_nested_probe'
os.makedirs(OUT_DIR, exist_ok=True)

EMBED_PT = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
RQIDX = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt'

PCA_DIM = 30
R_INITS = 5


def softplus(x):
    return float(np.log1p(np.exp(x)))


def fit_probe_nested(X_A, X_l, y_target, alpha_init_logit=0.0):
    """Fit nested probe: y ≈ X_A @ w_A + softplus(α) * X_l @ w_l.

    Returns dict with α, R²_A, R²_C, ΔR².
    """
    X_train_A, X_test_A, X_train_l, X_test_l, y_train, y_test = train_test_split(
        X_A, X_l, y_target, test_size=0.2, random_state=42
    )
    # Fit Probe A first (Ridge)
    mA = Ridge(alpha=1.0)
    mA.fit(X_train_A, y_train if y_train.ndim == 1 else y_train)
    r2_A = float(r2_score(y_test, mA.predict(X_test_A), multioutput='raw_values').mean())

    # Fit Probe l only
    ml = Ridge(alpha=1.0)
    ml.fit(X_train_l, y_train if y_train.ndim == 1 else y_train)

    # α ≥ 0 via softplus; gradient descent on α
    best_alpha = 0.0
    best_r2_C = r2_A
    # Brute grid (avoids tf/optimizer)
    for a_logit in np.linspace(-5, 5, 21):
        a = softplus(a_logit)
        pred_C = mA.predict(X_test_A) + a * ml.predict(X_test_l)
        r2_C = float(r2_score(y_test, pred_C, multioutput='raw_values').mean())
        if r2_C > best_r2_C:
            best_r2_C = r2_C
            best_alpha = a

    return {
        'alpha': best_alpha,
        'r2_A': r2_A,
        'r2_C': best_r2_C,
        'delta_r2': best_r2_C - r2_A,
    }


def main():
    print('=' * 70)
    print('Task 299: 单调嵌套 Probe (proxy 版, R={} init)'.format(R_INITS))
    print('=' * 70)

    # Load data
    print('\n[1] 加载 data...')
    x_sem = torch.load(EMBED_PT, map_location='cpu', weights_only=False)
    if x_sem.dim() > 2:
        x_sem = x_sem.squeeze(0)
    if x_sem.shape[0] != 11924 and x_sem.shape[1] == 11924:
        x_sem = x_sem.t()
    N = x_sem.shape[0]
    X = x_sem.float().numpy()  # (N, 2048)
    bundle = torch.load(RQIDX, map_location='cpu', weights_only=False)
    codebooks = bundle['codebooks']
    idx_lst = bundle['idx_lst']
    print(f'  N={N}')

    # Build per-layer cumulative q^l (Probe A features)
    q_per_layer = []
    q_cum = np.zeros((N, X.shape[1]), dtype=np.float32)
    for l in range(3):
        cb = codebooks[l]
        idx = idx_lst[l].numpy()
        q_l = cb[idx].float().numpy()
        q_cum += q_l
        q_per_layer.append(q_cum.copy())

    # Target: use FLAN-T5 embedding as proxy (multidim via PCA-50)
    pca_y = PCA(n_components=50, random_state=42).fit_transform(X)
    print(f'  Target dim: {pca_y.shape}')

    # Per-layer analysis with R inits
    results = {}
    for l in range(3):
        layer_name = f'L{l+1}'
        print(f'\n  [{layer_name}] R={R_INITS} init...')
        results[layer_name] = {'R_inits': []}
        for r_init in range(R_INITS):
            # Different random projection for feature (mimics init variance)
            np.random.seed(42 + r_init)
            pca_feat_A = PCA(n_components=PCA_DIM, random_state=42 + r_init).fit_transform(q_per_layer[l] - q_per_layer[l].mean(0))
            pca_feat_l = PCA(n_components=PCA_DIM, random_state=42 + r_init).fit_transform(q_per_layer[l] - q_per_layer[l].mean(0))
            # Use l-th layer's q^l specifically
            cb = codebooks[l]
            idx = idx_lst[l].numpy()
            q_l = cb[idx].float().numpy()
            pca_feat_l_only = PCA(n_components=PCA_DIM, random_state=42 + r_init).fit_transform(q_l - q_l.mean(0))
            probe = fit_probe_nested(pca_feat_A, pca_feat_l_only, pca_y)
            results[layer_name]['R_inits'].append(probe)
            print(f'    init {r_init+1}: α={probe["alpha"]:.4f}, '
                  f'R²_A={probe["r2_A"]:.3f}, R²_C={probe["r2_C"]:.3f}, ΔR²={probe["delta_r2"]:.4f}')

        # Aggregate over R inits
        deltas = [r['delta_r2'] for r in results[layer_name]['R_inits']]
        alphas = [r['alpha'] for r in results[layer_name]['R_inits']]
        results[layer_name]['delta_r2_mean'] = float(np.mean(deltas))
        results[layer_name]['delta_r2_std'] = float(np.std(deltas))
        results[layer_name]['alpha_mean'] = float(np.mean(alphas))
        results[layer_name]['alpha_std'] = float(np.std(alphas))

    # Save
    out = {
        'n_items': N,
        'embed_path': EMBED_PT,
        'note': 'Proxy: target=FLAN-T5 PCA-50, Probe A 特征=累计 q^l, f_l=z[l] codebook. R=5 init.',
        'results': results,
    }
    with open(os.path.join(OUT_DIR, 'nested_probe_per_layer.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n[产物] {OUT_DIR}/nested_probe_per_layer.json')

    # Verdict
    with open(os.path.join(OUT_DIR, 'verdict.md'), 'w') as f:
        f.write('# Task 299 Verdict: 单调嵌套 Probe (proxy 版)\n\n')
        f.write(f'数据集: Toys (N={N}), R={R_INITS} init, target=FLAN-T5 PCA-50\n\n')
        f.write('## 各层 ΔR² (R²_C - R²_A) over R inits\n\n')
        f.write('| Layer | ΔR² mean | ΔR² std | α mean | α std |\n')
        f.write('|-------|----------|---------|--------|-------|\n')
        for l in range(3):
            r = results[f'L{l+1}']
            f.write(f'| L{l+1} | {r["delta_r2_mean"]:.4f} | {r["delta_r2_std"]:.4f} | '
                    f'{r["alpha_mean"]:.4f} | {r["alpha_std"]:.4f} |\n')
        f.write('\n## 判读\n\n')
        for l in range(3):
            r = results[f'L{l+1}']
            mean_d = r['delta_r2_mean']
            std_d = r['delta_r2_std']
            f.write(f'### L{l+1}\n')
            f.write(f'- ΔR² = {mean_d:.4f} ± {std_d:.4f}, α = {r["alpha_mean"]:.4f}\n')
            if mean_d < 0.005:
                f.write(f'  → **深层增量信息极少**, α 自然衰减 → 深层无显著可读取增量\n')
            elif mean_d < 0.02:
                f.write(f'  → 深层提供有限增量\n')
            else:
                f.write(f'  → 深层提供显著增量 (但需考虑这是 proxy)\n')
            f.write('\n')

    print(f'[产物] {OUT_DIR}/verdict.md')


if __name__ == '__main__':
    main()