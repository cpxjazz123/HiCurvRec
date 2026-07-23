#!/usr/bin/env python3
"""第二步: 验证 NMI(c_1, c_3) 高 和 causal_gain +0.082 是否数学矛盾

核心问题:
- C_gsrq NMI(c_1, c_3) = 0.628 → c_3 几乎是 c_1 的确定性函数
- 如果 c_3 是 c_1 的 deterministic refinement, 那么 r_l^⊥ (去掉 q_1 线性预测的部分)
  理论上应该接近噪声, 不应该测出显著为正的信息量
- 但 causal_gain = +0.082 → 矛盾?

新设计:
1. 算 r_l^⊥ 的有效维度 (effective rank / participation ratio)
   - erank(r_l^⊥) ≈ 1 表示完全坍缩到一维 (符合 "c_l 几乎由 c_1 决定" 的预期)
   - erank(r_l^⊥) 大 → 残差空间仍有多维结构
2. 算 r_l^⊥ 各维度与 Y 的原始相关性 (不经过 M)
   - 看 Y 信号是不是真的在 r_l^⊥ 里
3. 比较: r_l (原始) vs r_l^⊥ (去 q_1) vs r_l^{⊥,M} (causal 正交)
   三者在 erank / 与 Y 关联 / 每个维度熵 上有何差异

判定:
- 如果 r_l^⊥ erank ≈ 1, 且与 Y 原始关联弱, 但 r_l^{⊥,M} 与 Y 关联强 →
  M 在 "制造" 信息, circular reasoning
- 如果 r_l^⊥ erank 不低, 与 Y 有原始关联, 只是这种关联在欧氏下被稀释 →
  causal_gain 可能是真实信号
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
N_PER_DIM_MI = 500  # Per-dim MI 抽样数 (2048 dim, 全算太慢)


def effective_rank(X):
    """Effective rank via singular value ratio: exp(H(sv/sv_sum)).
    X: (N, D). Returns scalar erank.
    """
    s = np.linalg.svd(X, compute_uv=False)
    s = s / (s.sum() + 1e-12)
    s_nonzero = s[s > 1e-10]
    H = -np.sum(s_nonzero * np.log(s_nonzero + 1e-12))
    return float(np.exp(H))


def per_dim_mi_with_y(X, y, n_sample=N_PER_DIM_MI, seed=42):
    """For each dim of X, MI(X[:,d]; y) in bits.
    Returns (D,) array."""
    N = X.shape[0]
    rng = np.random.default_rng(seed)
    if N > n_sample:
        idx = rng.choice(N, size=n_sample, replace=False)
        X_s = X[idx]
        y_s = y[idx]
    else:
        X_s, y_s = X, y
    mi = mutual_info_classif(X_s, y_s, n_neighbors=5, random_state=seed,
                              discrete_features=False, n_jobs=4)
    return mi / np.log(2)


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


def load_M(ckpt_path, layer=0):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ck['state_dict']
    key = f'quantization_layer_list.{layer}.centroids'
    t = sd[key].float().numpy()
    if t.shape[0] == 256:
        t = t.T
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
    print('Step 2 — NMI vs causal_gain 数学矛盾检验')
    print('=' * 70)
    y = build_y(EMB_PATH, K=20)

    bundles = {n: torch.load(p, map_location='cpu', weights_only=False)
               for n, p in RQIDX_PATHS.items()}

    results = {}
    for algo in ['A_baseline', 'B_mmq', 'C_gsrq']:
        print(f'\n{"="*70}\n{algo}\n{"="*70}')
        bundle = bundles[algo]
        r_lst = bundle['r_lst']
        q_lst = bundle['q_lst']
        M = load_M(RQ_CKPT_PATHS[algo], layer=0)
        algo_res = {}

        for l in range(1, len(r_lst)):
            r_l = r_lst[l].numpy().astype(np.float32)
            q_1 = q_lst[0].numpy().astype(np.float32)
            r_perp_e = decouple_eucl(r_l, q_1)
            r_perp_m = decouple_causal(r_l, q_1, M)

            # === Effective rank ===
            erank_r = effective_rank(r_l)
            erank_perp_e = effective_rank(r_perp_e)
            erank_perp_m = effective_rank(r_perp_m)
            print(f'  L{l}: erank(r_l)={erank_r:.1f}, '
                  f'erank(r_perp_E)={erank_perp_e:.1f}, '
                  f'erank(r_perp_M)={erank_perp_m:.1f}')

            # === Norms (proxy for "how much is left") ===
            r_norm = float(np.linalg.norm(r_l, axis=-1).mean())
            perp_e_norm = float(np.linalg.norm(r_perp_e, axis=-1).mean())
            perp_m_norm = float(np.linalg.norm(r_perp_m, axis=-1).mean())
            ratio_e = perp_e_norm / (r_norm + 1e-12)
            ratio_m = perp_m_norm / (r_norm + 1e-12)
            print(f'  L{l}: ‖r_l‖={r_norm:.3f}, ‖r_perp_E‖={perp_e_norm:.3f} ({ratio_e:.1%}), '
                  f'‖r_perp_M‖={perp_m_norm:.3f} ({ratio_m:.1%})')

            # === Per-dim MI with Y (原始信号, 不经过 M) ===
            print(f'  L{l}: per-dim MI(r_l → Y)...')
            mi_per_dim_r = per_dim_mi_with_y(r_l, y)
            mi_per_dim_e = per_dim_mi_with_y(r_perp_e, y)
            mi_per_dim_m = per_dim_mi_with_y(r_perp_m, y)
            # Stats: top-k mean, median, max, frac > 0.05
            def stats(arr, name):
                top1_mean = float(np.sort(arr)[-10:].mean())  # top-10 dims
                median = float(np.median(arr))
                max_v = float(arr.max())
                frac_above_005 = float((arr > 0.005).mean())
                print(f'    {name}: median={median:.4f}, top10 mean={top1_mean:.4f}, '
                      f'max={max_v:.4f}, frac>0.005={frac_above_005:.3f}')
                return {'median': median, 'top10_mean': top1_mean, 'max': max_v,
                        'frac_above_0p005': frac_above_005}
            sd_r = stats(mi_per_dim_r, 'r_l         ')
            sd_e = stats(mi_per_dim_e, 'r_perp_E    ')
            sd_m = stats(mi_per_dim_m, 'r_perp_M    ')

            # === Fast joint V-info (PCA reduced) ===
            mi_base = fast_mi_bits(r_l, y)
            mi_perp_e = fast_mi_bits(r_perp_e, y)
            mi_perp_m = fast_mi_bits(r_perp_m, y)
            print(f'  L{l}: I_V(r_l)={mi_base:.4f}, I_V(perp_E)={mi_perp_e:.4f}, '
                  f'I_V(perp_M)={mi_perp_m:.4f}')

            algo_res[f'l{l}'] = {
                'erank_r': erank_r,
                'erank_perp_e': erank_perp_e,
                'erank_perp_m': erank_perp_m,
                'r_norm': r_norm,
                'perp_e_norm': perp_e_norm,
                'perp_m_norm': perp_m_norm,
                'ratio_perp_e': ratio_e,
                'ratio_perp_m': ratio_m,
                'mi_per_dim_r': sd_r,
                'mi_per_dim_perp_e': sd_e,
                'mi_per_dim_perp_m': sd_m,
                'v_info_r': mi_base,
                'v_info_perp_e': mi_perp_e,
                'v_info_perp_m': mi_perp_m,
                'delta_causal': mi_perp_m - mi_base,
                'delta_eucl':   mi_perp_e - mi_base,
                'causal_gain':  (mi_perp_m - mi_base) - (mi_perp_e - mi_base),
            }
        results[algo] = algo_res

    # Verdict
    print('\n' + '=' * 70)
    print('VERDICT — Step 2 数学矛盾检验')
    print('=' * 70)
    for algo in ['A_baseline', 'B_mmq', 'C_gsrq']:
        print(f'\n{algo}:')
        for lk, lr in results[algo].items():
            print(f'  {lk}:')
            print(f'    erank(r_l)={lr["erank_r"]:.1f}, erank(perp_E)={lr["erank_perp_e"]:.1f}, '
                  f'erank(perp_M)={lr["erank_perp_m"]:.1f}')
            print(f'    per-dim MI r_l top10={lr["mi_per_dim_r"]["top10_mean"]:.4f}, '
                  f'perp_E top10={lr["mi_per_dim_perp_e"]["top10_mean"]:.4f}, '
                  f'perp_M top10={lr["mi_per_dim_perp_m"]["top10_mean"]:.4f}')
            print(f'    V-info I_V(r_l)={lr["v_info_r"]:.4f}, '
                  f'I_V(perp_E)={lr["v_info_perp_e"]:.4f}, '
                  f'I_V(perp_M)={lr["v_info_perp_m"]:.4f}, '
                  f'causal_gain={lr["causal_gain"]:+.4f}')

    # Key check for C_gsrq:
    c = results['C_gsrq']
    print('\n' + '=' * 70)
    print('KEY CHECK — C_gsrq (高 NMI 是否与高 causal_gain 矛盾)')
    print('=' * 70)
    for lk in c:
        erank_e = c[lk]['erank_perp_e']
        erank_m = c[lk]['erank_perp_m']
        mi_r_top10 = c[lk]['mi_per_dim_r']['top10_mean']
        mi_e_top10 = c[lk]['mi_per_dim_perp_e']['top10_mean']
        mi_m_top10 = c[lk]['mi_per_dim_perp_m']['top10_mean']
        # If erank_e ≈ 1 and mi_e_top10 ≈ 0 → 去 q_1 之后接近噪声
        # If erank_e large and mi_e_top10 > 0 → 残差有原始信息
        erank_collapsed = erank_e < 10
        has_raw_signal = mi_e_top10 > 0.001
        m_amplifies = mi_m_top10 > mi_e_top10 * 1.5
        print(f'  {lk}:')
        print(f'    erank(perp_E) = {erank_e:.1f}  ({erank_collapsed})')
        print(f'    per-dim MI top10 (perp_E) = {mi_e_top10:.5f}  ({"has_raw" if has_raw_signal else "near_noise"})')
        print(f'    per-dim MI top10 (perp_M) = {mi_m_top10:.5f}  '
              f'({"M amplifies" if m_amplifies else "M not amp"})')
        if erank_collapsed and not has_raw_signal and m_amplifies:
            print(f'    ⚠️ r_perp_E 接近噪声, 但 r_perp_M 仍有信号 → M 可能在 "制造" 信息')
        elif has_raw_signal and m_amplifies:
            print(f'    ✓ r_perp_E 有原始信号, M 放大而非制造 → 真实')

    out_path = os.path.join(OUT_DIR, 'ideaB_step2_erank.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()