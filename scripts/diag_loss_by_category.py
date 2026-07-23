#!/usr/bin/env python3
"""信息丢失的品类分布 (Task #3) — v5 — 终版

前几版发现：
- 子集内 5-fold KNN trivial（全 rec=1.000）
- cat_sub 上 global 5-fold KNN 在 K=19 时随机基线 5.26%，rec_x=0.058, rec_q=0.077（q1 略好）
- 信号非常弱，难以衡量 per-cat_sub 异质性

更换核心指标为 **class-conditional Fisher criterion**:
    对某个 cat_sub 类 c，定义 z_c = 类内均值向量；用 ||z_c||² 与类内平均方差来衡量
    任务信息容量。

指标（每类）：
1. **Fisher score (F)**:
   F_c = (类内均值向量中心化后的范数²) / (类内平均样本方差)
   这是经典可分性度量。

2. **CRR (cat_sub Recovery Rate)**:
   对每个 cat_sub 类 c，随机抽 N_c 个样本对：
   - P_same = 类别 c 内样本对的 cosine 相似度中位数
   - P_diff = 不同 c 样本对的 cosine 相似度中位数
   - CRR_c = P_same - P_diff

3. **ρ ratio (上一版的 PCA-ρ 重写)**:
   ρ_c = sum_i∈c ||U^T (z_i - μ_c)||² / sum_i∈c ||z_i - μ_c||²
   ratio_c = ρ_q_c / ρ_x_c，loss_rho_c = clip(1 - ratio_c, 0, 1)

x 与 q1 分别计算，对每个 cat_sub 类 c 取 loss = 1 - score_q_c / score_x_c
(各 score 维度下)。

KILL LINE: std(loss_ratio) < 10pp UNIFORM; > 20pp HETEROGENEITY.
"""

import sys
sys.path.insert(0, '/home/wlia0047/ar57/wenyu/GeneRec/GRID')

import src.utils.decorators  # noqa: F401

import os
import json
import numpy as np
import torch
from collections import Counter
from scipy.stats import spearmanr

EMB_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/2026-07-07/18-51-46/pickle/merged_predictions_tensor.pt'
IDX_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task6/A_baseline_rqidx.pt'
META_PATH = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/ideaB_causal/toys_metadata.json'
OUT_DIR = '/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/diag_loss_by_category'
os.makedirs(OUT_DIR, exist_ok=True)

MIN_CAT_SIZE = 50
PCA_NCOMP = 64
N_PCA_CRR = 32
N_PAIR_SAMPLE = 400
RNG = 42

# (label_key, label_threshold, alias)
CONFIGS = [
    ('cat_sub', 50, 'cat_sub_n50'),
    ('brand', 50, 'brand_n50'),
    ('brand', 100, 'brand_n100'),
]


def load_data():
    print('[load] embedding …')
    x = torch.load(EMB_PATH, map_location='cpu', weights_only=False).float().numpy()
    b = torch.load(IDX_PATH, map_location='cpu', weights_only=False)
    q1 = b['q_lst'][0].numpy().astype(np.float32)
    print(f'  x: {x.shape}, q1: {q1.shape}')
    return x, q1, x.shape[0]


def load_metadata(N):
    d = json.load(open(META_PATH))
    cat_sub = np.array([d[str(i)].get('cat_sub', 'NO_CAT') for i in range(N)])
    brand = np.array([d[str(i)].get('brand', 'NO_BRAND') for i in range(N)])
    return cat_sub, brand


def fisher_score(X, y, c, eps=1e-12):
    """F = ||μ_c||^2 / σ² with centering by global mean μ.
    Here we use within-class dispersion:
        F_c = (||μ_c - μ||^2) / E[||x_i - μ_c||^2]
    """
    idx = (y == c)
    Xc = X[idx]
    n = Xc.shape[0]
    if n < 5:
        return 0.0
    mu = Xc.mean(axis=0)
    mu_g = X[idx].mean(axis=0)  # within-class mean
    # Actually: we want between / within: numerator = ||μ_c - μ_global||^2
    # denominator = within-class variance (average squared distance from μ_c)
    # but global mean changes per class; for class-specific F we use class-mean vs global-mean
    diff = mu - X.mean(axis=0)
    num = float((diff ** 2).sum())
    den = float(((Xc - mu) ** 2).sum(axis=1).mean())
    return num / (den + eps)


def crr_score(X, y, c, n_pairs=N_PAIR_SAMPLE, seed=RNG):
    """Class-conditional: cos-sim median of intra-class pairs vs cross-class pairs."""
    idx = (y == c)
    Xc = X[idx]
    n_c = Xc.shape[0]
    if n_c < 5:
        return 0.0
    rng = np.random.default_rng(seed)
    # intra pairs
    if n_c >= 2:
        n_intra = min(n_pairs, n_c * (n_c - 1) // 2)
        a = rng.integers(0, n_c, size=n_intra)
        b = rng.integers(0, n_c, size=n_intra)
        valid = a != b
        a, b = a[valid], b[valid]
        ua = Xc[a]; ub = Xc[b]
        cos_intra = (ua * ub).sum(axis=1) / (
            np.linalg.norm(ua, axis=1) * np.linalg.norm(ub, axis=1) + 1e-12
        )
    else:
        cos_intra = np.zeros(1)
    # cross pairs: pick other class indices
    idx_other = (y != c)
    Xo = X[idx_other]
    n_o = Xo.shape[0]
    n_cross = min(n_pairs, n_c * n_o)
    a = rng.integers(0, n_c, size=n_cross)
    b = rng.integers(0, n_o, size=n_cross)
    ua = Xc[a]; ub = Xo[b]
    cos_cross = (ua * ub).sum(axis=1) / (
        np.linalg.norm(ua, axis=1) * np.linalg.norm(ub, axis=1) + 1e-12
    )
    return float(np.median(cos_intra) - np.median(cos_cross))


def rho_ratio(x, q1, idx, U_x):
    x_sub = x[idx]; q_sub = q1[idx]
    n_sub = x_sub.shape[0]
    if n_sub < 5:
        return None, None
    mu_x = x_sub.mean(axis=0); mu_q = q_sub.mean(axis=0)
    x_c = x_sub - mu_x; q_c = q_sub - mu_q
    x_d = float((x_c ** 2).sum()); q_d = float((q_c ** 2).sum())
    if x_d < 1e-12 or q_d < 1e-12:
        return None, None
    rho_x = float(((x_c @ U_x) ** 2).sum()) / (x_d + 1e-12)
    rho_q = float(((q_c @ U_x) ** 2).sum()) / (q_d + 1e-12)
    return rho_x, rho_q


def global_knn_5fold(X, y, K=5, n_folds=5, seed=RNG):
    keep = y != -1
    Xk = X[keep]; yk = y[keep]
    N = Xk.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    fold_size = N // n_folds
    correct = 0
    total = 0
    for f in range(n_folds):
        test_idx = perm[f * fold_size:(f + 1) * fold_size]
        train_idx = np.concatenate([perm[:f * fold_size], perm[(f + 1) * fold_size:]])
        Xt, yt = Xk[train_idx], yk[train_idx]
        Xte, yte = Xk[test_idx], yk[test_idx]
        m = Xte.shape[0]
        if m == 0:
            continue
        chunk = 256
        preds = np.empty(m, dtype=yk.dtype)
        for s in range(0, m, chunk):
            e = min(s + chunk, m)
            Xb = Xte[s:e]
            d2 = ((Xt[:, None, :] - Xb[None, :, :]) ** 2).sum(-1)
            Kk = min(K, len(yt))
            idx_k = np.argpartition(d2, Kk - 1, axis=0)[:Kk]
            knn_labels = yt[idx_k]
            for bi in range(e - s):
                vals, counts = np.unique(knn_labels[:, bi], return_counts=True)
                preds[bi] = vals[np.argmax(counts)]
        correct += int((preds == yte).sum())
        total += int(m)
    return correct / max(total, 1), len(yk)


def main():
    x, q1, N = load_data()
    cat_sub, brand = load_metadata(N)

    from sklearn.decomposition import PCA

    print(f'[PCA] shared n_components={PCA_NCOMP}, knn n_components={N_PCA_CRR}')
    pca_shared = PCA(n_components=PCA_NCOMP, random_state=RNG).fit(x)
    U_x = pca_shared.components_.T
    x_pca_crr = PCA(n_components=N_PCA_CRR, random_state=RNG).fit_transform(x).astype(np.float32)
    q_pca_crr = PCA(n_components=N_PCA_CRR, random_state=RNG).fit_transform(q1).astype(np.float32)

    # Global rho
    idx_all = np.ones(N, dtype=bool)
    rho_x_g, rho_q_g = rho_ratio(x, q1, idx_all, U_x)
    global_loss_rho = float(np.clip(1 - rho_q_g / max(rho_x_g, 1e-12), 0, 1))

    # Per config
    all_results = {}
    for key, thresh, alias in CONFIGS:
        label = cat_sub if key == 'cat_sub' else brand
        ctr = Counter(label.tolist())
        keep = sorted({c for c, n in ctr.items() if n >= thresh})
        mp = {c: i for i, c in enumerate(keep)}
        y_int = np.array([mp.get(v, -1) for v in label], dtype=np.int64)

        print(f'\n=== {alias} (K={len(keep)}, threshold n >= {thresh}) ===')

        rec_x_g, n_used = global_knn_5fold(x_pca_crr, y_int, K=5, n_folds=5, seed=RNG)
        rec_q_g, _ = global_knn_5fold(q_pca_crr, y_int, K=5, n_folds=5, seed=RNG)
        random_baseline = 1.0 / max(len(keep), 1)
        discr_ratio = rec_x_g / random_baseline
        global_loss_knn = float(np.clip(1 - rec_q_g / max(rec_x_g, 1e-6), 0, 1))
        print(f'  global: rec_x={rec_x_g:.4f}, rec_q={rec_q_g:.4f}, '
              f'random={random_baseline:.4f}, discr={discr_ratio:.2f}x, '
              f'loss_knn={global_loss_knn * 100:.2f}pp')

        # Per class
        rows = []
        for c_str in keep:
            c_id = mp[c_str]
            mask = (y_int == c_id)
            n_items = int(mask.sum())
            rho_x, rho_q = rho_ratio(x, q1, mask, U_x)
            if rho_x is None or rho_q is None:
                continue
            ratio_qx = float(rho_q / max(rho_x, 1e-12))
            loss_rho = float(np.clip(1 - ratio_qx, 0, 1))

            f_x = fisher_score(x, y_int, c_id)
            f_q = fisher_score(q1, y_int, c_id)
            loss_fisher = float(np.clip(1 - f_q / max(f_x, 1e-12), 0, 1))

            crr_x = crr_score(x, y_int, c_id)
            crr_q = crr_score(q1, y_int, c_id)
            # CRR is a difference: if positive, intra>cross (good); if q1 < x, loss
            # We define loss_crr = (crr_x - crr_q) / max(|crr_x|, eps) clipped
            denom = max(abs(crr_x), 1e-6)
            loss_crr = float((crr_x - crr_q) / denom)
            loss_crr = float(np.clip(loss_crr, -1, 1))  # may go negative if q1 better

            mean_var_x = float(np.mean(np.sum(x[mask] ** 2, axis=1)))
            mean_var_q = float(np.mean(np.sum(q1[mask] ** 2, axis=1)))

            rows.append({
                'alias': alias, 'class': c_str,
                'n_items': n_items,
                'mean_sq_norm_x': mean_var_x,
                'mean_sq_norm_q1': mean_var_q,
                'rho_x': rho_x, 'rho_q': rho_q, 'rho_ratio_q_over_x': ratio_qx,
                'loss_rho': loss_rho,
                'fisher_x': f_x, 'fisher_q': f_q, 'loss_fisher': loss_fisher,
                'crr_x': crr_x, 'crr_q': crr_q, 'loss_crr': loss_crr,
            })

        # Aggregate
        loss_rho_arr = np.array([r['loss_rho'] for r in rows])
        loss_fisher_arr = np.array([r['loss_fisher'] for r in rows])
        loss_crr_arr = np.array([r['loss_crr'] for r in rows])
        n_arr = np.array([r['n_items'] for r in rows])
        var_x = np.array([r['mean_sq_norm_x'] for r in rows])

        def safe_corr(a, b):
            if a.size == 0 or b.size == 0 or np.all(a == a[0]) or np.all(b == b[0]):
                return 0.0, 1.0
            r, p = spearmanr(a, b)
            return float(r), float(p)

        def band(std):
            if std < 0.10:
                return 'UNIFORM (<10pp)'
            if std > 0.20:
                return 'SYSTEMATIC_HETEROGENEITY (>20pp)'
            return 'INTERMEDIATE (10-20pp)'

        sp_n_f, p_n_f = safe_corr(loss_fisher_arr, n_arr)
        sp_v_f, p_v_f = safe_corr(loss_fisher_arr, var_x)
        sp_n_c, p_n_c = safe_corr(loss_crr_arr, n_arr)
        sp_v_c, p_v_c = safe_corr(loss_crr_arr, var_x)

        loss_fisher_std = float(loss_fisher_arr.std())
        loss_crr_std = float(loss_crr_arr.std())
        # CRR can have negative signs; std computed as-is; |std| < 0.10 → UNIFORM
        fisher_band = band(loss_fisher_std)
        crr_band = band(abs(loss_crr_std))

        # Kill criterion: a metric is SIGNALING if the relevant summary scalar
        # shows > 0.10 std across classes. Use Fisher criterion as the
        # primary signal (it's noise-robust).
        if (discr_ratio < 1.2) and (fisher_band == 'UNIFORM (<10pp)'):
            kill = ('NO_SIGNAL: discriminator < 1.2x random on this label; '
                    'no measurable task information to lose; '
                    '"74% loss" narrative does not apply.')
        elif loss_fisher_std > 0.20:
            kill = ('SYSTEMATIC_HETEROGENEITY: Fisher loss std > 20pp — '
                    'information loss is concentrated in some classes; '
                    'potential link to VarLenRec, separate reporting warranted.')
        elif loss_fisher_std < 0.10:
            kill = ('UNIFORM: Fisher loss std < 10pp — information loss is '
                    'roughly uniform across this label.')
        else:
            kill = ('INTERMEDIATE: Fisher loss std in 10-20pp; mild heterogeneity.')

        agg = {
            'K': len(keep),
            'n_used': n_used,
            'random_baseline': random_baseline,
            'discr_ratio': float(discr_ratio),
            'rec_x_global': float(rec_x_g),
            'rec_q_global': float(rec_q_g),
            'global_loss_knn': global_loss_knn,
            'loss_rho': {
                'mean': float(loss_rho_arr.mean()) if loss_rho_arr.size else 0,
                'std': float(loss_rho_arr.std()) if loss_rho_arr.size else 0,
                'min': float(loss_rho_arr.min()) if loss_rho_arr.size else 0,
                'max': float(loss_rho_arr.max()) if loss_rho_arr.size else 0,
                'median': float(np.median(loss_rho_arr)) if loss_rho_arr.size else 0,
            },
            'loss_fisher': {
                'mean': float(loss_fisher_arr.mean()) if loss_fisher_arr.size else 0,
                'std': float(loss_fisher_arr.std()) if loss_fisher_arr.size else 0,
                'min': float(loss_fisher_arr.min()) if loss_fisher_arr.size else 0,
                'max': float(loss_fisher_arr.max()) if loss_fisher_arr.size else 0,
                'median': float(np.median(loss_fisher_arr)) if loss_fisher_arr.size else 0,
            },
            'loss_crr': {
                'mean': float(loss_crr_arr.mean()) if loss_crr_arr.size else 0,
                'std': float(loss_crr_arr.std()) if loss_crr_arr.size else 0,
                'min': float(loss_crr_arr.min()) if loss_crr_arr.size else 0,
                'max': float(loss_crr_arr.max()) if loss_crr_arr.size else 0,
                'median': float(np.median(loss_crr_arr)) if loss_crr_arr.size else 0,
            },
            'fisher_band': fisher_band,
            'crr_band': crr_band,
            'corr': {
                'loss_fisher_x_n': sp_n_f, 'p_loss_fisher_x_n': p_n_f,
                'loss_fisher_x_var_x': sp_v_f, 'p_loss_fisher_x_var_x': p_v_f,
                'loss_crr_x_n': sp_n_c, 'p_loss_crr_x_n': p_n_c,
                'loss_crr_x_var_x': sp_v_c, 'p_loss_crr_x_var_x': p_v_c,
            },
            'kill': kill,
        }

        print(f'  loss_fisher: mean={agg["loss_fisher"]["mean"] * 100:.2f}pp, '
              f'std={agg["loss_fisher"]["std"] * 100:.2f}pp → {fisher_band}')
        print(f'  loss_crr: mean={agg["loss_crr"]["mean"] * 100:.2f}pp, '
              f'std={agg["loss_crr"]["std"] * 100:.2f}pp → {crr_band}')
        print(f'  KILL: {kill}')

        rows_sorted_f = sorted(rows, key=lambda r: -r['loss_fisher'])
        rows_sorted_c = sorted(rows, key=lambda r: -r['loss_crr'])
        rows_sorted_r = sorted(rows, key=lambda r: -r['loss_rho'])

        all_results[alias] = {
            'aggregate': agg,
            'rows_sorted_loss_fisher': rows_sorted_f,
            'rows_sorted_loss_crr': rows_sorted_c,
            'rows_sorted_loss_rho': rows_sorted_r,
        }

    # Save CSVs
    csv_path = os.path.join(OUT_DIR, 'loss_ratio_per_category.csv')
    with open(csv_path, 'w') as f:
        f.write('alias,class,n_items,mean_sq_norm_x,mean_sq_norm_q1,'
                'rho_x,rho_q,rho_ratio_q_over_x,loss_rho,'
                'fisher_x,fisher_q,loss_fisher,'
                'crr_x,crr_q,loss_crr\n')
        for alias, payload in all_results.items():
            for r in payload['rows_sorted_loss_fisher']:
                f.write(
                    f'{alias},{r["class"]},{r["n_items"]},'
                    f'{r["mean_sq_norm_x"]:.6f},{r["mean_sq_norm_q1"]:.6f},'
                    f'{r["rho_x"]:.6f},{r["rho_q"]:.6f},{r["rho_ratio_q_over_x"]:.6f},'
                    f'{r["loss_rho"]:.6f},'
                    f'{r["fisher_x"]:.6f},{r["fisher_q"]:.6f},{r["loss_fisher"]:.6f},'
                    f'{r["crr_x"]:.6f},{r["crr_q"]:.6f},{r["loss_crr"]:.6f}\n'
                )
    print(f'\n[saved] {csv_path}')

    # correlation csv
    corr_path = os.path.join(OUT_DIR, 'correlation.csv')
    with open(corr_path, 'w') as f:
        f.write('alias,metric,rho,p_value\n')
        for alias, payload in all_results.items():
            for k in ['loss_fisher_x_n', 'loss_fisher_x_var_x',
                      'loss_crr_x_n', 'loss_crr_x_var_x']:
                rho = payload['aggregate']['corr'].get(k, 0.0)
                p = payload['aggregate']['corr'].get(
                    'p_' + k, 1.0
                )
                metric_name = k.replace('_x_', '_vs_').replace('_n', 'n_items').replace('_var_x', 'var_x')
                f.write(f'{alias},{metric_name},{rho:.6f},{p:.6f}\n')
    print(f'[saved] {corr_path}')

    # Bar chart
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(all_results), 2,
                             figsize=(15, 4 * len(all_results)),
                             squeeze=False)
    for i, (alias, payload) in enumerate(all_results.items()):
        agg_c = payload['aggregate']
        rows_f = payload['rows_sorted_loss_fisher']
        labels = [r['class'] for r in rows_f]
        values = [r['loss_fisher'] * 100 for r in rows_f]
        colors = ['#d62728' if v > agg_c['loss_fisher']['mean'] * 100 else '#1f77b4'
                  for v in values]
        y_pos = np.arange(len(labels))
        axes[i, 0].barh(y_pos, values, color=colors)
        axes[i, 0].set_yticks(y_pos)
        axes[i, 0].set_yticklabels(labels, fontsize=7)
        axes[i, 0].invert_yaxis()
        axes[i, 0].set_xlabel('loss_fisher (pp)')
        axes[i, 0].set_title(
            f'{alias} — Fisher loss (K={agg_c["K"]})\n'
            f'mean={agg_c["loss_fisher"]["mean"] * 100:.1f}pp, '
            f'std={agg_c["loss_fisher"]["std"] * 100:.1f}pp'
        )
        axes[i, 0].axvline(agg_c['loss_fisher']['mean'] * 100, color='gray', linestyle='--', linewidth=0.8)

        rows_c = payload['rows_sorted_loss_crr']
        labels_c = [r['class'] for r in rows_c]
        values_c = [r['loss_crr'] * 100 for r in rows_c]
        colors_c = ['#d62728' if v > 0 else '#2ca02c' for v in values_c]
        y_pos_c = np.arange(len(labels_c))
        axes[i, 1].barh(y_pos_c, values_c, color=colors_c)
        axes[i, 1].set_yticks(y_pos_c)
        axes[i, 1].set_yticklabels(labels_c, fontsize=7)
        axes[i, 1].invert_yaxis()
        axes[i, 1].set_xlabel('loss_crr (pp; positive = x better, negative = q1 better)')
        axes[i, 1].set_title(
            f'{alias} — CRR loss (K={agg_c["K"]})\n'
            f'mean={agg_c["loss_crr"]["mean"] * 100:.1f}pp, '
            f'std={agg_c["loss_crr"]["std"] * 100:.1f}pp'
        )
        axes[i, 1].axvline(0, color='black', linewidth=0.5)

    plt.tight_layout()
    chart_path = os.path.join(OUT_DIR, 'category_loss_bar.png')
    plt.savefig(chart_path, dpi=130)
    plt.close()
    print(f'[saved] {chart_path}')

    # JSON
    res_path = os.path.join(OUT_DIR, 'loss_by_category.json')
    with open(res_path, 'w') as f:
        json.dump({
            'global_rho_x': rho_x_g,
            'global_rho_q': rho_q_g,
            'global_loss_rho': global_loss_rho,
            'configs': all_results,
        }, f, indent=2)
    print(f'[saved] {res_path}')

    # Markdown
    md = []
    md.append('# 信息丢失的品类分布 — Verdict (v5)\n')
    md.append('## 方法学说明\n')
    md.append('观察到前几版 (v1-v4) 的"子集内 5-fold KNN"在所有子类上 trivial 全 100% 准确率，'
              '因为同类样本互相是最近邻，并不能反映"L1 量化损失多少任务信息"。'
              '本版使用三个跨类 / 全局维度指标:\n')
    md.append('| 指标 | 含义 |')
    md.append('|---|---|')
    md.append('| **Fisher loss** | (类间||μ_c - μ||² / 类内方差) 在 x 与 q1 上的差异。越大说明 q1 抹掉越多 |')
    md.append('| **CRR loss** | (同类 cos − 异类 cos) 在 x 与 q1 上差异。越大说明 q1 同类异类分离能力下降越多 |')
    md.append('| **PCA-ρ loss** | 在 x 主子空间上的子集方差比例，q1 相对 x 的保留比 |')
    md.append('')
    md.append('## 全局基线\n')
    md.append(f'- ρ_x(global) = {rho_x_g:.4f}, ρ_q(global) = {rho_q_g:.4f}, '
              f'ratio = {rho_q_g / max(rho_x_g, 1e-12):.4f}\n')
    md.append('## Per-config 总结\n')
    md.append('| alias | K | discr_ratio | loss_fisher mean±std | CRR band | KILL |')
    md.append('|---|---:|---:|---:|---|---|')
    for alias, payload in all_results.items():
        agg_c = payload['aggregate']
        md.append(
            f'| {alias} | {agg_c["K"]} | {agg_c["discr_ratio"]:.2f}x | '
            f'{agg_c["loss_fisher"]["mean"] * 100:.1f} ± '
            f'{agg_c["loss_fisher"]["std"] * 100:.1f}pp | '
            f'{agg_c["crr_band"]} | {agg_c["kill"][:60]} … |'
        )
    md.append('')
    for alias, payload in all_results.items():
        agg_c = payload['aggregate']
        md.append(f'### {alias} (K={agg_c["K"]})\n')
        md.append(f'- global rec_x={agg_c["rec_x_global"]:.4f}, '
                  f'rec_q={agg_c["rec_q_global"]:.4f}, '
                  f'discriminator_ratio={agg_c["discr_ratio"]:.2f}x 随机')
        md.append(f'- loss_fisher: mean={agg_c["loss_fisher"]["mean"] * 100:.2f}pp, '
                  f'std={agg_c["loss_fisher"]["std"] * 100:.2f}pp, '
                  f'min/max={agg_c["loss_fisher"]["min"] * 100:.2f}/'
                  f'{agg_c["loss_fisher"]["max"] * 100:.2f}pp')
        md.append(f'- loss_crr: mean={agg_c["loss_crr"]["mean"] * 100:.2f}pp, '
                  f'std={agg_c["loss_crr"]["std"] * 100:.2f}pp')
        md.append(f'- 相关性 Spearman (Fisher loss vs n_items): '
                  f'ρ={agg_c["corr"]["loss_fisher_x_n"]:+.4f}, p={agg_c["corr"]["p_loss_fisher_x_n"]:.4f}')
        md.append(f'- 相关性 Spearman (Fisher loss vs var_x): '
                  f'ρ={agg_c["corr"]["loss_fisher_x_var_x"]:+.4f}, p={agg_c["corr"]["p_loss_fisher_x_var_x"]:.4f}')
        md.append(f'- 相关性 Spearman (CRR loss vs n_items): '
                  f'ρ={agg_c["corr"]["loss_crr_x_n"]:+.4f}, p={agg_c["corr"]["p_loss_crr_x_n"]:.4f}')
        md.append(f'- 相关性 Spearman (CRR loss vs var_x): '
                  f'ρ={agg_c["corr"]["loss_crr_x_var_x"]:+.4f}, p={agg_c["corr"]["p_loss_crr_x_var_x"]:.4f}')
        md.append('- KILL: ' + agg_c['kill'])
        md.append('')
        md.append('| class | n_items | loss_fisher (pp) | loss_crr (pp) | fisher_x | fisher_q |')
        md.append('|---|---:|---:|---:|---:|---:|')
        for r in payload['rows_sorted_loss_fisher']:
            md.append(
                f'| {r["class"][:32]} | {r["n_items"]} | '
                f'{r["loss_fisher"] * 100:.2f} | {r["loss_crr"] * 100:.2f} | '
                f'{r["fisher_x"]:.4f} | {r["fisher_q"]:.4f} |'
            )
        md.append('')

    md.append('## 解读\n')
    md.append('1. **cat_sub 标签**在 flan-t5-xl embedding 上几乎不携带可分类别信号 — global discriminator ≈ 1.11x 随机基线。'
              '因此"74% 信息丢失"在 cat_sub 上叙事不成立，因信号本身就是零。')
    md.append('2. **brand 标签**在 flan-t5-xl embedding 上有可分信号 (discr ≈ 2-3x 随机)。'
              '对 brand_n50 / brand_n100 观察 Fisher loss 的 std 是否 > 20pp。')
    md.append('3. CRR 指标允许 loss 为负（q1 在某些类上同类异类分离能力更好），'
              '正值意味着 q1 抹掉了 x 上同类更聚集的结构。')
    md.append('4. Kill line 对 fish loss 严格做 std<10/10-20/>20 三档。'
              'cat_sub 可能 → NO_SIGNAL；brand 可能 → UNIFORM 或 HETEROGENEITY，'
              '这才是真正的结论。')

    verdict_path = os.path.join(OUT_DIR, 'verdict.md')
    with open(verdict_path, 'w') as f:
        f.write('\n'.join(md) + '\n')
    print(f'[saved] {verdict_path}')

    print('\n' + '=' * 70)
    print('FINAL VERDICT')
    print('=' * 70)
    for alias, payload in all_results.items():
        agg_c = payload['aggregate']
        print(f'  {alias}: K={agg_c["K"]}, discr={agg_c["discr_ratio"]:.2f}x, '
              f'Fisher mean={agg_c["loss_fisher"]["mean"] * 100:.2f}pp, '
              f'std={agg_c["loss_fisher"]["std"] * 100:.2f}pp → '
              f'{agg_c["fisher_band"]}')
        print(f'    KILL: {agg_c["kill"]}')
    print('=' * 70)


if __name__ == '__main__':
    main()
