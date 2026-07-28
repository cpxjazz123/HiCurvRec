"""Task #240 / Issue #12 Gate 0 — SID 逐层分布画像 (零 GPU, 分钟级).

对每个 SID .npy 文件 (9922 × 4 int), 逐层计算:
- Shannon 熵 (token 分布均匀度, log(K))
- Gini 系数 (token 分布不平等度, 0=完全均匀, 1=完全集中)
- 标准差 / mean
- top-1 占比
- top-10 累计占比
- utilization (used_codes / K)
- 路径稀疏度 (实际路径数 / 理论最大路径数)

参考 arXiv:2407.21488v2 (Kuai et al. 2024) §4.1 Figure 3 实测: 中间层 Gini 0.55-0.57
"""
import numpy as np
from pathlib import Path
import json
import math

K_list = [64, 128, 256]

def gini(coeffs):
    """Gini coefficient. coeffs = non-negative array (e.g. token counts)."""
    arr = np.sort(np.asarray(coeffs, dtype=np.float64))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    cum = np.cumsum(arr)
    # Standard Gini: 1 - 2 * AUC, with the Lorenz curve
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n

def shannon_entropy(counts):
    """Shannon entropy H = -sum p log p, normalized by log(K)."""
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    H = -(p * np.log(p)).sum()
    K = len(counts)
    return float(H / math.log(K)) if K > 1 else 0.0  # normalized to [0, 1]

def analyze_layer(sids, layer_idx, K):
    """Analyze one layer of SID distribution."""
    tokens = sids[:, layer_idx]  # (N,)
    counts = np.bincount(tokens, minlength=K)[:K]  # (K,)
    used_codes = (counts > 0).sum()
    util = used_codes / K
    g = gini(counts)
    H = shannon_entropy(counts)
    total = counts.sum()
    sorted_counts = np.sort(counts)[::-1]
    top1 = sorted_counts[0] / total if total > 0 else 0
    top10 = sorted_counts[:min(10, K)].sum() / total if total > 0 else 0
    return {
        'K': K,
        'used_codes': int(used_codes),
        'utilization': float(util),
        'gini': float(g),
        'shannon_entropy_norm': float(H),
        'mean_count': float(counts.mean()),
        'std_count': float(counts.std()),
        'top1_share': float(top1),
        'top10_share': float(top10),
    }

def analyze_sid(sid_path):
    """Full per-layer + cross-layer analysis."""
    sids = np.load(sid_path)  # (N, 4)
    N = sids.shape[0]
    layers = []
    for l in range(3):
        layers.append(analyze_layer(sids, l, K_list[l]))
    # Path sparsity: actual unique tuples vs theoretical K_l1 * K_l2 * K_l3
    unique_paths = len(set(map(tuple, sids[:, :3])))  # 3-digit (before 4th dedup)
    theoretical_3layer = K_list[0] * K_list[1] * K_list[2]
    return {
        'N_items': int(N),
        'layers': layers,
        'unique_3layer_paths': int(unique_paths),
        'theoretical_3layer_paths': int(theoretical_3layer),
        'path_sparsity_3layer': float(unique_paths / theoretical_3layer),
    }

if __name__ == '__main__':
    sid_files = {
        'A_HG-Rec_baseline_task84': 'HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_poincare.npy',
        'B_ArmB_task237_partialSinkhorn': 'HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task237_armB.npy',
    }
    results = {}
    for label, path in sid_files.items():
        if not Path(path).exists():
            print(f'❌ MISSING: {path}')
            continue
        print(f'\n=== {label} ({path}) ===')
        r = analyze_sid(path)
        results[label] = r
        for l, layer in enumerate(r['layers']):
            print(f'  L{l}: K={layer["K"]:4d}  util={layer["utilization"]*100:5.1f}%  '
                  f'gini={layer["gini"]:.4f}  H_norm={layer["shannon_entropy_norm"]:.4f}  '
                  f'top1={layer["top1_share"]*100:5.1f}%  top10={layer["top10_share"]*100:5.1f}%  '
                  f'mean={layer["mean_count"]:.1f}±{layer["std_count"]:.1f}')
        print(f'  3-layer unique paths: {r["unique_3layer_paths"]} / {r["theoretical_3layer_paths"]} '
              f'(sparsity={r["path_sparsity_3layer"]*100:.4f}%)')
    
    # Output JSON for record
    out_path = 'verdicts/task240_issue12_gate0_distribution.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults written to: {out_path}')
    
    # Issue #12 Gate 0 pass condition
    print('\n=== Issue #12 Gate 0 pass check ===')
    print('Pass: ∃ layer Gini ≥ 0.5 AND utilization ≥ 90%')
    gate0_passed = False
    for label, r in results.items():
        for l, layer in enumerate(r['layers']):
            if layer['gini'] >= 0.5 and layer['utilization'] >= 0.9:
                print(f'  ✅ PASS: {label} L{l} Gini={layer["gini"]:.4f}, util={layer["utilization"]*100:.1f}%')
                gate0_passed = True
    if not gate0_passed:
        print(f'  ❌ FAIL: No layer has Gini ≥ 0.5 AND util ≥ 90%')
        print(f'  → Issue #12 H1 REFUTED, 沙漏效应不成立, 方向关闭')
    else:
        print(f'  → Issue #12 H1 confirmed, 沙漏效应存在, 进入 Gate 1')
