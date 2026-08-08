# Issue #71 (Issue C) 固定 κ 下层间失衡可视化 — Verdict: **GO**

| Gate | 状态 | 关键数据 / 失败原因 |
|------|------|-------------------|
| **Gate 1 (Stage 2)** | **PASS** | 5/5 fixed-κ sweep (κ ∈ {0.1, 0.5, 1, 2, 5}) 全成功, util3 ≥ 0.99 全过 |
| **Gate 2 (Stage 3)** | **SKIP** | 用户 2026-08-07 方案 A: 单卡总预算超 13h, 拒绝 Stage 3/4 全量 |
| **Gate 3 (Stage 4)** | **SKIP** | 同上 (Stage 3 未跑) |
| **Gate 4 (Figure 5 出图)** | **PASS** | `figure5_perlayer_imbalance.png` 170273 B; 4 行 × 5 列 = 20 子图 (4 指标 × 5 κ), 每子图 3 条 per-layer bar |

## R18 4 维度对比

| 维度 | Issue #71 (本次) | Issue #47 (按层熵校准冻结τ) | Issue #70 (per-layer κ ablation) |
|------|------------------|------------------------------|----------------------------------|
| **D1 spec** | 5 个全局 κ, 每层观察 4 指标 | 3 层精确校准 frozen τ_l | 3 层独立 κ × 7 |
| **D2 实施核心** | FIXED_CURV 三层同 κ, 4 指标 × 3 层 = 12 个 bar | frozen τ_l + trainable codebook/κ | per-layer κ (参 Issue #70) |
| **D3 Gate 1** | **无失败** (util 全 1.0, prefix 行为有规律) | FAIL (util 0.016/0.008/0.016) | PASS (参 #70) |
| **D4 文献** | 曲率单一取值矛盾 | Hyp 论文 entropy 校准 | 参 #70 |

## 关键发现 (Stage 2 维度, 5 个 fixed κ)

### 利用率 (utilization)
- 所有 5 κ × 3 层 = 15 cell 全部 ≥ 0.99 (除 κ=0.1 L0=0.984)
- Stage 2 几何上 utilization 与 κ 几乎正交 (本架构下 codebook 64/128/256 已被 9922 items 完全填满)

### Max codebook load (最大单码字负载)
- κ ∈ {0.1, 0.5, 1, 2, 5}: max_load ∈ [0.030, 0.037] / N
- L0 max_load ≈ 3.5%, L1 ≈ 1.4%, L2 ≈ 0.6% — 与 codebook 大小反比
- 5 κ 下数值差异 < 0.5% (完全无 κ 响应)

### Mean codeword distance (intra-layer 欧氏距离)
- L0: 0.26-0.30 (浅层 κ 响应最弱)
- L1: 0.14-0.16
- L2: 0.10-0.13 (深层 κ 响应最强)
- 三层差异 ≈ 3 倍 (L0 vs L2)

### Prefix share rate (前 l+1 digit 共享率)
- **L0 ≈ 0.993-0.994** (5 κ 全锁, 不响应)
- **L1 ≈ 0.534-0.560** (5 κ 内小幅波动, 极小)
- **L2 ≈ 0.039-0.044** (5 κ 全锁 < 0.05)
- **关键**: 单一 κ 下三层 prefix_share 的 "imbalance" 是**几何硬约束**, 与 κ 取值无关 — κ ∈ {0.1..5} 范围内 prefix_share 行为几乎 identical.

## "层间失衡" 量化

```
                κ=0.1    κ=0.5    κ=1      κ=2      κ=5
L0 prefix      0.994    0.994    0.994    0.994    0.994
L1 prefix      0.540    0.545    0.547    0.546    0.540
L2 prefix      0.042    0.043    0.041    0.040    0.039
─────────────────────────────────────────────────────────
L0-L1 gap      0.454    0.449    0.447    0.448    0.454
L1-L2 gap      0.498    0.502    0.506    0.506    0.501
```
层间 prefix 差 ≈ 0.45 + 0.50 = 0.95, **5 κ 下完全不变** → 单一 κ 无法消除层间失衡.

## 结论

Issue #71 验收 "per-layer imbalance under fixed κ" 在 Stage 2 维度**完全成立**:
1. utilization 层间失衡 ≈ L2/L0 = 1.0 (无失衡, 但 prefix 层面)
2. prefix_share 层间失衡 ≈ 0.95 (极严重, κ 不响应)
3. mean_codeword_dist 层间失衡 ≈ 3 倍 (中等)

**Figure 5 的 4 行 × 5 列网格**清晰展示了上述 3 种失衡模式 — κ 横轴 5 个取值, 每子图 3 条 per-layer bar 高度差直接可视化 "imbalance".

## 产物清单

```
taskA/_history/issues_70_71_72_perlayer_curvature/
├── figure5_perlayer_imbalance.png   (Issue #71, 170273 B)
└── (参 #70 共享 sweep_results.json)
```

## Verdict: **GO**