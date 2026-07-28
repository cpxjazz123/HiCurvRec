# Task #240 / Issue #12 Gate 0 — SID 沙漏效应 H1 REFUTED (NO-GO)

**日期**: 2026-07-29
**决定**: ⛔ **H1 REFUTED → 沙漏效应在本数据集不成立, Issue #12 方向关闭**
**Stage**: Gate 0 (零 GPU 分布画像) — 已完成, 不进入 Gate 1

## 1. 验证对象

Issue #12 引用 arXiv:2407.21488v2 (Kuai et al. 2024), 提出假设 H1:
> "本仓库现有 SID 产物存在显著的逐层分布不均匀, 且该不均匀度**不被 utilization 与 collision_rate 捕捉**。具体: 存在至少一层, 其 token 分布 Gini ≥ 0.5 (主文献中间层实测 0.55-0.57), 而该层 utilization ≥ 90%。"

**Gate 0 通过条件**: ∃ layer Gini ≥ 0.5 AND utilization ≥ 90%
**Gate 0 硬停止**: 三层 Gini < 0.5 → H1 证伪 → STOP

## 2. 输入数据 (零 GPU 纯分析)

- **Arm A (HG-Rec baseline #84)**: `HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_poincare.npy` (9922 items × 4 SID)
- **Arm B (Task #237 partial Sinkhorn)**: `HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task237_armB.npy` (同 shape)

## 3. 分布画像结果

| Arm | Layer | K | utilization | Gini | H_norm | top1% | top10% | mean±std |
|------|-------|---|-------------|------|--------|-------|--------|----------|
| A | L0 | 64  | 100.0% | **0.2008** | 0.9833 | 4.3% | 26.3% | 155.0±60.7 |
| A | L1 | 128 | 100.0% | **0.1737** | 0.9899 | 1.7% | 13.0% | 77.5±24.4 |
| A | L2 | 256 | 100.0% | **0.2000** | 0.9880 | 1.3% | 8.3% | 38.8±14.7 |
| B | L0 | 64  | 100.0% | **0.1984** | 0.9836 | 4.3% | 26.2% | 155.0±60.1 |
| B | L1 | 128 | 100.0% | **0.1741** | 0.9899 | 1.7% | 13.0% | 77.5±24.4 |
| B | L2 | 256 | 100.0% | **0.1987** | 0.9882 | 1.3% | 8.2% | 38.8±14.5 |

**3-layer 路径稀疏度 (Arm A)**: 8936 / 2097152 = **0.4261%** (理论最大路径数 K_0×K_1×K_2)
**3-layer 路径稀疏度 (Arm B)**: 8925 / 2097152 = **0.4256%**

## 4. 关键发现

### 4.1 Gini 全部 < 0.21, 远低于 0.5 阈值

所有 6 个数据点 (3 层 × 2 arm) 的 Gini ∈ [0.1737, 0.2008]:
- L0 Gini ≈ 0.20 (A: 0.2008, B: 0.1984)
- L1 Gini ≈ 0.17 (A: 0.1737, B: 0.1741)
- L2 Gini ≈ 0.20 (A: 0.2000, B: 0.1987)

**Issue #12 H1 阈值 0.5 vs 实测 0.17-0.20, 差距 2.5×-3×. 完全不在 H1 假设的 "显著不均匀" 范围内**.

### 4.2 Shannon 熵 ≥ 0.98 (接近最大)

H_norm ∈ [0.9833, 0.9899] — 三层熵都接近 log(K) 理论最大值, 分布接近完全均匀.
对比主文献 (RQ3x12 4096 codes/层) 中间层 H_norm ≈ 8.18/12 = 0.68 — **本数据集分布均匀度是主文献的 1.4×**.

### 4.3 top-1 token 占比 ≤ 4.3%, top-10 累计 ≤ 26.3%

- L0 top-1 = 4.3% (A 和 B) — 单码字仅承载 ~430 items, 远低于 "大路由节点" 特征
- L0 top-10 累计 = 26% — 10 个码字承载 ~2600 items, 平均每码字 260 items, 接近 L0 平均数 155
- 主文献 L2 top-K 处理的有效性条件: 头部 token 显著大于均值. 本数据集 top1/top10 都在均值 1-2× 范围内, **不构成头部集中**.

### 4.4 Path sparsity 0.42% (3-layer)

3-digit unique paths = 8936 (A) / 8925 (B), 理论最大 2097152.
**真实路径数仅占理论 0.42%** — 大量 (9922 - 8936) = 986 items 跟其他 item 共享 3-digit prefix (碰撞在 4-digit dedup 解决).
但这不是 "路径稀疏度" 意义上的集中 — 反而是 **路径多样性高** 的特征 (8936 unique / 9922 items = 90% 唯一).

## 5. 决策 (per Issue #12 Gate 0 硬停止)

- **Gini 阈值 0.5 vs 实测 0.17-0.20**: REFUTED ❌ (差距 2.5×-3×)
- **H1 假设**: REFUTED ❌ (沙漏效应在本数据集不成立)
- **Issue #12 处置**: STOP at Gate 0, 就地写 verdict 关闭本方向.

按 Issue #12 原文的硬停止规则:
> "若三层 Gini 均 < 0.5, 则 H1 证伪——STOP, 就地写 verdict 关闭本方向, 并明确记录 '沙漏效应在本数据集不成立, 码本健康↔召回解耦另有原因'。**不得**因为 '反正 Gate 0 便宜' 就顺手往下跑。"

本任务严格遵守 Gate 0 硬停止, 不进入 Gate 1 / Gate 2 / Gate 3.

## 6. 启示

### 6.1 主文献 Kuai 2024 沙漏效应不能直接外推到本数据集

主文献条件: RQ3x12 (码本 4096/层) + 二亿商品 + LLaMA2-0.8B/7B.
本项目条件: K=[64,128,256] (小 16-64×) + 9922 items (小 2万×) + T5-mini 9.18M (小 87-778×).

Issue #12 自己也警告 "规模外推风险": "码本越小、数据越少, 长尾结构越可能被 '码字不够用' 这一更平凡的原因掩盖".

实测证实: 码本小 → 码字全部被均匀使用 → Gini 自然低 → 沙漏形态不显现. **本数据集的 R@10 上限受其他因素限制, 跟主文献的沙漏机制不直接相关**.

### 6.2 码本健康↔召回解耦的真正原因 (本任务新增证据)

Issue #12 H1 想找的 "码本指标无法解释召回" 的原因, 实证后**不是** "码本健康但分布不均". 真正原因候选:
- 训练时长 (task200 dual_v5 提示)
- T5 容量 (task159 12M / task160 60M 容量点)
- 数据规模限制 (9922 items 已是 Musical_Instruments 5-core 上限)

### 6.3 12 方向 + Issue #12 合并 NO-GO 闭环

- 7 方向 NO-GO (Task #117/82/88/89/90/118/165/207)
- 3 NO-HOPE (Task #211/212/213/214)
- 1 钉半径硬停 (Task #119/Task #211 钉半径)
- 1 paper Eq12 假设 (Task #137)
- **1 沙漏效应 Gate 0 (本任务, Issue #12)** — 14 方向全 NO-GO

**整个 "码本侧" 改造路径已被穷尽**. R@10 杠杆只能在 "训练协议 / 跨架构 / 数据规模" 三条中找.

## 7. 产物

- descriptions/task240_issue12_gate0_distribution.md (description, 如有)
- scripts/task240_issue12_gate0_distribution_diagnostic.py (可复用诊断工具, 用于未来任何新 SID .npy)
- verdicts/task240_issue12_gate0_distribution_nogo.md (本文件)
- verdicts/task240_issue12_gate0_distribution.json (原始数据)
- Issue #12 comment: 提议关闭 (待用户/AI 评论)
- HG-Rec 上游不变 (零代码改动)

## 8. Status

- ✅ Issue #12 Gate 0 H1 REFUTED ❌
- ❌ Issue #12 Gate 1/2/3 不进入
- ✅ Issue #12 方向关闭 (按硬停止规则)
- ⏭️ 下一个 Issue #11 Gate 0 (per-layer c_k range) 候选 — 同样零 GPU 分钟级
