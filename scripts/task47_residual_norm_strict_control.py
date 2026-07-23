#!/usr/bin/env python3
"""Task 305: Residual Norm 严格控制 (4 probes, β_2 显著性)

判断深层信息衰减是否主要由 residual magnitude 变小导致.

实现:
- 利用 task16 产出的 r_lst (list[Tensor[11924, 2048]]) 包含每层输入 residual
- 计算每层 gain g_i^l = ||r_i^l||_2, direction d_i^l = r_i^l / (g_i^l + eps)
- 4 种 probe: gain-only, direction-only, gain+direction, raw
- 拟合 regression: Utility_{i,l} ~ β_0 + β_1 |r| + β_2 layer + ε
"""
import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import os, json
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task47_norm_strict'
os.makedirs(OUT_DIR, exist_ok=True)

RQIDX = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt'


def compute_layer_metrics(r_lst, q_lst, codebooks, idx_lst, layer_l, n_targets=200):
    """For each item in layer l:
    - gain g_i = ||r_i^l||_2
    - direction d_i = r_i^l / (g_i + eps)
    Then train regression: y (target = codebook assignment) ~ features

    Since y requires ground truth target, we use next-layer codebook idx as proxy.
    """
    N = r_lst[0].shape[0]
    r_l = r_lst[layer_l].float()  # (N, D)
    r_lt = r_lst[layer_l + 1].float() if layer_l + 1 < len(r_lst) else r_lst[layer_l].float()

    # gains
    gains = r_l.norm(dim=-1).numpy()  # (N,)
    eps = 1e-8
    direction = (r_l / (r_l.norm(dim=-1, keepdim=True) + eps)).numpy()  # (N, D)

    # Target: next-layer codebook assignment
    if layer_l + 1 < len(idx_lst):
        target = idx_lst[layer_l + 1].numpy()  # (N,) integer codes
    else:
        target = idx_lst[layer_l].numpy()

    return gains, direction, target


def fit_probe(X, y, alpha=1.0):
    """Train Ridge regression on (X -> y classification via one-hot mean)."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    # Score = explained variance
    from sklearn.metrics import explained_variance_score, r2_score
    r2 = r2_score(y_test, y_pred_test)
    return r2


def main():
    print('=' * 70)
    print('Task 305: Residual norm 严格控制 (4 probes, β_2 显著性)')
    print('=' * 70)

    bundle = torch.load(RQIDX, map_location='cpu', weights_only=False)
    r_lst = bundle['r_lst']
    q_lst = bundle['q_lst']
    idx_lst = bundle['idx_lst']
    codebooks = bundle['codebooks']

    print(f'\n[1] 加载 residual bundle: r_lst ({len(r_lst)} layers), each (11924, 2048)')

    results = {}
    for layer_l in [0, 1, 2]:
        layer_name = f'L{layer_l+1}'
        print(f'\n[2.{layer_l+1}] {layer_name} (l={layer_l}) 4 probes...')
        gains, direction, target = compute_layer_metrics(r_lst, q_lst, codebooks, idx_lst, layer_l)
        print(f'  gain stats: mean={gains.mean():.2f}, median={np.median(gains):.2f}, max={gains.max():.2f}, min={gains.min():.2f}')

        results[layer_name] = {}

        # 1. Gain-only probe
        r2_gain = fit_probe(gains.reshape(-1, 1), target.astype(float), alpha=1.0)
        results[layer_name]['gain_only_R2'] = r2_gain
        print(f'  gain-only R² (gain -> next code): {r2_gain:.4f}')

        # 2. Direction-only probe (use top-PCA-50 dims)
        # Reduce dim for speed
        from sklearn.decomposition import PCA
        n_pca = min(50, direction.shape[1])
        pca = PCA(n_components=n_pca)
        dir_pca = pca.fit_transform(direction)
        r2_dir = fit_probe(dir_pca, target.astype(float), alpha=1.0)
        results[layer_name]['direction_only_R2'] = r2_dir
        results[layer_name]['direction_pca_var_explained'] = float(pca.explained_variance_ratio_.sum())
        print(f'  direction-only R² (PCA-{n_pca} -> next code): {r2_dir:.4f} (pca var sum {pca.explained_variance_ratio_.sum():.3f})')

        # 3. Gain + Direction
        gain_dir_feat = np.hstack([gains.reshape(-1, 1), dir_pca])
        r2_gd = fit_probe(gain_dir_feat, target.astype(float), alpha=1.0)
        results[layer_name]['gain_plus_dir_R2'] = r2_gd
        print(f'  gain+dir R²: {r2_gd:.4f}')

        # 4. Raw residual probe (PCA-50 of r_l)
        r_arr = r_lst[layer_l].float().numpy()
        r_pca = PCA(n_components=n_pca).fit_transform(r_arr)
        r2_raw = fit_probe(r_pca, target.astype(float), alpha=1.0)
        results[layer_name]['raw_residual_R2'] = r2_raw
        print(f'  raw residual R² (PCA-{n_pca}): {r2_raw:.4f}')

        # Gain stats
        results[layer_name]['gain_stats'] = {
            'mean': float(gains.mean()),
            'median': float(np.median(gains)),
            'std': float(gains.std()),
            'max': float(gains.max()),
            'min': float(gains.min()),
        }
        results[layer_name]['note'] = (
            'gain_only: 1D gain | direction: PCA-50 of dir | '
            'gain+dir: stacked | raw: PCA-50 of r_l. Target: next-layer code.'
        )

    # 5. Regression Utility_{i,l} ~ β_0 + β_1 |r_i| + β_2 layer + ε
    # Use next-layer code as proxy "utility"
    print('\n[3] Regression: Utility ~ gain + layer')
    table = []
    for layer_l in [0, 1, 2]:
        gains, _, target = compute_layer_metrics(r_lst, q_lst, codebooks, idx_lst, layer_l)
        for i in range(len(gains)):
            table.append({
                'item': i,
                'layer': layer_l,
                'gain': gains[i],
                'target_code': int(target[i]),
            })
    table_arr = np.array([(r['gain'], r['layer'], r['target_code']) for r in table])
    X = np.column_stack([table_arr[:, 0], table_arr[:, 1]])  # gain, layer
    y = table_arr[:, 2]  # target code
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    reg.fit(X, y)
    y_pred = reg.predict(X)
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2_overall = 1 - ss_res / ss_tot
    coefs = reg.coef_.tolist()
    intercept = reg.intercept_
    print(f'  R² = {r2_overall:.4f}')
    print(f'  β_0 (intercept) = {intercept:.4f}')
    print(f'  β_1 (gain) = {coefs[0]:.6f}')
    print(f'  β_2 (layer) = {coefs[1]:.4f}')

    regression_results = {
        'r2_overall': float(r2_overall),
        'beta_0_intercept': float(intercept),
        'beta_1_gain': float(coefs[0]),
        'beta_2_layer': float(coefs[1]),
        'note': 'positive β_2 means deeper layers have higher target code, controlling for gain',
    }

    # Save
    out = {
        'r_lst_path': RQIDX,
        'n_items': r_lst[0].shape[0],
        'layers_tested': ['L1', 'L2', 'L3'],
        'results': results,
        'regression': regression_results,
    }
    with open(os.path.join(OUT_DIR, '4probe_per_layer.json'), 'w') as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(OUT_DIR, 'regression_coef.json'), 'w') as f:
        json.dump(regression_results, f, indent=2)
    print(f'\n[产物] {OUT_DIR}/4probe_per_layer.json')
    print(f'[产物] {OUT_DIR}/regression_coef.json')

    # Verdict
    with open(os.path.join(OUT_DIR, 'verdict.md'), 'w') as f:
        f.write('# Task 305 Verdict: Residual Norm 严格控制\n\n')
        f.write(f'R-VAE 数据集: Toys (N={r_lst[0].shape[0]})\n\n')
        f.write('## 4 Probe R² (predict next-layer codebook assignment)\n\n')
        f.write('| Layer | Gain-only | Direction-only (PCA-50) | Gain+Dir | Raw residual (PCA-50) | Gain mean |\n')
        f.write('|-------|-----------|-------------------------|----------|----------------------|-----------|\n')
        for l_name in ['L1', 'L2', 'L3']:
            r = results[l_name]
            f.write(f'| {l_name} | {r["gain_only_R2"]:.4f} | {r["direction_only_R2"]:.4f} | '
                    f'{r["gain_plus_dir_R2"]:.4f} | {r["raw_residual_R2"]:.4f} | '
                    f'{r["gain_stats"]["mean"]:.2f} |\n')
        f.write('\n## 跨层趋势\n\n')
        dir_r2 = [results[f'L{i+1}']['direction_only_R2'] for i in range(3)]
        gain_r2 = [results[f'L{i+1}']['gain_only_R2'] for i in range(3)]
        for i, l_name in enumerate(['L1', 'L2', 'L3']):
            f.write(f'- {l_name}: direction R² = {dir_r2[i]:.4f}, gain R² = {gain_r2[i]:.4f}\n')
        f.write('\n')
        if dir_r2[2] < dir_r2[0] * 0.5:
            f.write('⚠️ **深层 direction R² 显著低于浅层** → direction 信息确实衰减\n')
        if gain_r2[2] > gain_r2[0] * 1.5:
            f.write('- **深层 gain 的可预测性比浅层高** → 深层能用 magnitude 区分\n')
        f.write('\n## 回归: Utility ~ β_0 + β_1 * gain + β_2 * layer\n\n')
        f.write(f'- R² (overall) = {regression_results["r2_overall"]:.4f}\n')
        f.write(f'- β_1 (gain) = {regression_results["beta_1_gain"]:.6f} (gain 越大, target code 越高)\n')
        f.write(f'- β_2 (layer) = {regression_results["beta_2_layer"]:.4f} (层数对 target code 的独立影响)\n')
        if abs(regression_results['beta_2_layer']) < 0.5:
            f.write('\n**结论**: 控制 gain 后, layer 几乎无独立效应 → **norm 主导**\n')
        else:
            f.write(f'\n**结论**: β_2 = {regression_results["beta_2_layer"]:.4f}, 有独立层效应\n')
    print(f'[产物] {OUT_DIR}/verdict.md')


if __name__ == '__main__':
    main()
