# Task #240 / Issue #12 Gate 0 — SID 沙漏效应分布画像 (H1 REFUTED)

> **任务目的**: 验证 Issue #12 假设 H1 (码本利用率 100% 但 token 分布 Gini ≥ 0.5) 是否在 Musical_Instruments 上成立. 引用 arXiv:2407.21488v2 (Kuai et al. 2024) 中间层沙漏机制.
>
> **完成日期**: 2026-07-29
> **状态**: ⛔ **H1 REFUTED → 沙漏效应不成立, Issue #12 方向关闭**

## 1. 背景

Issue #12 (2026-07-28) 提出: 利用率 100% 不代表分布均匀, 可能存在沙漏效应 (中间层 token 集中).
H1: ∃ layer Gini ≥ 0.5 AND utilization ≥ 90%.

## 2. Gate 0 验证

输入: 
- Arm A (HG-Rec baseline #84): `HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_poincare.npy`
- Arm B (Task #237 partial Sinkhorn): `HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task237_armB.npy`

## 3. 关键数字

| Arm | Layer | K | Gini | H_norm | top1% | top10% | util |
|------|-------|---|------|--------|-------|--------|------|
| A | L0 | 64  | 0.2008 | 0.9833 | 4.3% | 26.3% | 100% |
| A | L1 | 128 | 0.1737 | 0.9899 | 1.7% | 13.0% | 100% |
| A | L2 | 256 | 0.2000 | 0.9880 | 1.3% | 8.3% | 100% |
| B | L0 | 64  | 0.1984 | 0.9836 | 4.3% | 26.2% | 100% |
| B | L1 | 128 | 0.1741 | 0.9899 | 1.7% | 13.0% | 100% |
| B | L2 | 256 | 0.1987 | 0.9882 | 1.3% | 8.2% | 100% |

**所有 6 个数据点 Gini ∈ [0.17, 0.20], 远低于 0.5 阈值**. H1 REFUTED.

## 4. 决策

按 Issue #12 Gate 0 硬停止规则: 三层 Gini < 0.5 → STOP, 关闭方向.
本任务严格遵守, 不进入 Gate 1/2/3.

## 5. 产物

- scripts/task240_issue12_gate0_distribution_diagnostic.py (可复用)
- verdicts/task240_issue12_gate0_distribution_nogo.md
- verdicts/task240_issue12_gate0_distribution.json

## 6. 启示

- 主文献沙漏效应 (K=4096, 2亿商品) 在本数据集 (K=64-256, 9922 items) 不显现
- 码本小 → 码字自然均匀 → Gini 自然低
- Issue #12 寻找的 "码本健康但召回低" 真正原因不在分布层
- 12+1 方向合并 NO-GO 闭环: 整个 "码本侧" 改造路径已穷尽
- R@10 杠杆只能在 "训练协议 / 跨架构 / 数据规模" 三条中找

## 7. Status

⛔ **Issue #12 方向关闭**
