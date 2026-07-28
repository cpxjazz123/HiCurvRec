#!/usr/bin/env python3
"""
Task #244 — Issue #12 Gate 0: SID 分布画像 (零 GPU 纯分析)

输入: SID .npy (9922 × 4 int array)
输出: 逐层 (L0/L1/L2/L3) Shannon 熵 / Gini / std / top-1 / top-10 累计占比 / 路径稀疏度 / utilization

Author: Task #244 / Issue #12 (2026-07-29)
参考: Kuai et al. arXiv:2407.21488v2 §4.1
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from collections import Counter


def shannon_entropy(counts):
    """Shannon 熵 H = -Σ p_i log p_i, p_i = count_i / total."""
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]  # 移除零概率
    return float(-(p * np.log2(p)).sum())


def gini_coefficient(counts):
    """Gini 系数 G = Σ_i Σ_j |x_i - x_j| / (2 n Σ x_i).

    简化公式: G = (2 Σ_i i·x_i) / (n Σ_i x_i) - (n+1)/n, x_i 已升序排列
    """
    x = np.sort(counts.astype(np.float64))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * x) / (n * x.sum())) - (n + 1) / n)


def utilization(counts, total_items):
    """utilization = (counts > 0).sum() / total_possible_codes.

    total_possible_codes = counts.size (码字数 K)
    """
    return float((counts > 0).sum()) / counts.size


def path_sparsity(sid_array):
    """路径稀疏度 = unique_paths / (K0 × K1 × K2 × K3).

    返回实际 unique 路径数与理论最大路径数之比 (1.0 = 完全覆盖所有路径).
    """
    unique_paths = len(set(map(tuple, sid_array)))
    K0, K1, K2, K3 = (int(sid_array[:, i].max() + 1) for i in range(4))
    theoretical_max = K0 * K1 * K2 * K3
    return float(unique_paths / theoretical_max), unique_paths, theoretical_max


def profile_sid(sid_path, name):
    """对单个 SID .npy 文件计算逐层指标 + 整体路径稀疏度."""
    sid = np.load(sid_path)
    assert sid.ndim == 2 and sid.shape[1] == 4, f"Unexpected SID shape: {sid.shape}"

    n_items = sid.shape[0]
    K = [64, 128, 256, 1]  # task84 baseline codebook_size = [64,128,256,1]

    print(f"\n{'='*70}")
    print(f"  SID: {name}")
    print(f"  Path: {sid_path}")
    print(f"  shape: {sid.shape}, N items: {n_items}")
    print(f"{'='*70}")

    layer_results = []
    for layer_idx in range(4):
        tokens = sid[:, layer_idx]
        counts = np.bincount(tokens, minlength=K[layer_idx])
        nonzero_counts = counts[counts > 0]
        N_unique = len(nonzero_counts)

        entropy = shannon_entropy(counts)
        gini = gini_coefficient(counts)
        std = float(counts.std())
        mean_freq = float(counts.mean())
        # ideal uniform Gini: 9922 items 均匀分到 K 个码字 → 频次 ≈ N_items/K 每码字 → Gini ≈ 0
        # 但 item 数 > K → 必然有码字频次更高 → Gini > 0 (取决于 K/N 比例)
        top1_pct = float(counts.max() / counts.sum()) if counts.sum() > 0 else 0.0
        top10_pct = float(np.sort(counts)[-10:].sum() / counts.sum()) if counts.sum() > 0 else 0.0
        util = utilization(counts, n_items)

        # Ideal uniform: 9922 个 item 均匀分到 K 个码字, 每个码字频次 = 9922/K
        # Gini of uniform distribution = (K-1)/(K+1) (理论)
        # 但因为 total_items/K 不一定整数, 这里实际 ideal 几乎 = 0 (counts 都是 9922/K)
        ideal_gini = 0.0  # 严格均匀 → Gini = 0
        # 实际 Gini - ideal Gini > 0.3 算显著偏离均匀

        result = {
            'layer': f'L{layer_idx}',
            'K': K[layer_idx],
            'n_unique_codes': N_unique,
            'utilization': util,
            'shannon_entropy': entropy,
            'gini': gini,
            'std': std,
            'top1_pct': top1_pct,
            'top10_pct': top10_pct,
        }
        layer_results.append(result)

        print(f"  L{layer_idx} (K={K[layer_idx]}):")
        print(f"    unique={N_unique}/{K[layer_idx]} ({util*100:.2f}%)")
        print(f"    Shannon entropy = {entropy:.4f} (max ≈ log2({K[layer_idx]}) = {np.log2(K[layer_idx]):.4f})")
        print(f"    Gini = {gini:.4f} (ideal uniform = {ideal_gini})")
        print(f"    std = {std:.4f}, mean = {mean_freq:.4f}")
        print(f"    top-1 占比 = {top1_pct*100:.2f}%, top-10 累计 = {top10_pct*100:.2f}%")

    sparsity, n_unique_paths, theoretical_max = path_sparsity(sid)
    print(f"\n  路径稀疏度: {n_unique_paths} / {theoretical_max} = {sparsity*100:.4f}%")
    print(f"  (theoretical max = K0*K1*K2*K3 = 64*128*256*1 = {64*128*256*1})")

    return {
        'name': name,
        'path': str(sid_path),
        'n_items': int(n_items),
        'layers': layer_results,
        'path_sparsity': sparsity,
        'unique_paths': n_unique_paths,
        'theoretical_max_paths': theoretical_max,
    }


def gate0_decision(results):
    """Issue #12 Gate 0 通过条件: 至少一层 Gini ≥ 0.5 且该层 utilization ≥ 90%."""
    print(f"\n{'='*70}")
    print(f"  Issue #12 Gate 0 决策")
    print(f"{'='*70}")

    gini_fail = []
    util_pass_with_high_gini = []
    for res in results:
        for layer in res['layers']:
            if layer['gini'] >= 0.5 and layer['utilization'] >= 0.90:
                util_pass_with_high_gini.append((res['name'], layer))
            if layer['gini'] < 0.5:
                gini_fail.append((res['name'], layer))

    print(f"\n  Arm-层 Gini < 0.5 (未达 H1 标准) 的有 {len(gini_fail)} 个: ")
    for name, layer in gini_fail:
        print(f"    {name} {layer['layer']}: Gini={layer['gini']:.4f}, util={layer['utilization']:.4f}")

    print(f"\n  Arm-层 Gini ≥ 0.5 且 utilization ≥ 0.90 (H1 通过): {len(util_pass_with_high_gini)} 个")
    for name, layer in util_pass_with_high_gini:
        print(f"    ✅ {name} {layer['layer']}: Gini={layer['gini']:.4f}, util={layer['utilization']:.4f}")

    if len(util_pass_with_high_gini) > 0:
        print(f"\n  🚦 **Gate 0 PASS** — H1 成立, 存在沙漏式集中. 允许进入 Gate 1.")
        return True, "PASS"
    else:
        print(f"\n  🚦 **Gate 0 FAIL** — H1 证伪. 三层 Gini 均 < 0.5. STOP. 关闭方向.")
        print(f"     → 沙漏效应在本数据集不成立, 码本健康↔召回解耦另有原因.")
        return False, "FAIL"


def main():
    print(f"\nTask #244 — Issue #12 Gate 0: SID 分布画像")
    print(f"Reference: Kuai et al. arXiv:2407.21488v2 §4.1 (2024)")
    print(f"Dataset: Musical_Instruments 5-core (9922 items)")

    INSTRUMENTS = '/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments'

    sids_to_analyze = [
        ('Arm A (T84 baseline, no Sinkhorn)', f'{INSTRUMENTS}/Instruments_t5_rqvae_code_default.npy'),
        ('Arm B (T237 partial Sinkhorn max_iters=10)', f'{INSTRUMENTS}/Instruments_t5_rqvae_task237_armB.npy'),
        # Arm C (phonism) 待找
        ('Arm C (T237 strict Sinkhorn max_iters=30)', f'{INSTRUMENTS}/Instruments_t5_rqvae_task237_armC_strict.npy'),
        # Task #200 dual_v5 (Issue #8 confounded reference)
        ('T200 dual_v5 (Issue #8 confounded)', f'{INSTRUMENTS}/Instruments_t5_rqvae_dual_v5.npy'),
        # task178 / task180 200 epoch variants
        ('T178 200 epoch (Mode collapse test)', f'{INSTRUMENTS}/Instruments_t5_rqvae_hyp_e14.npy'),
    ]

    results = []
    for name, path in sids_to_analyze:
        if not Path(path).exists():
            print(f"\n⚠️ {name}: 文件不存在 {path}")
            continue
        result = profile_sid(path, name)
        results.append(result)

    if not results:
        print("\n❌ 没有可分析的 SID 文件")
        sys.exit(1)

    # 输出 JSON
    output_json = '/fs04/ar57/wenyu/GeneRec/verdicts/task244_sid_distribution_profile.json'
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ JSON 输出: {output_json}")

    # Gate 0 决策
    pass_, status = gate0_decision(results)

    # 输出决策标记 (机器可读)
    decision_path = '/fs04/ar57/wenyu/GeneRec/verdicts/task244_gate0_decision.json'
    with open(decision_path, 'w') as f:
        json.dump({'gate0': status, 'h1_pass': pass_, 'n_sids': len(results)}, f, indent=2)
    print(f"✅ 决策: {decision_path}")


if __name__ == '__main__':
    main()
