# Task #339 — Issue #52 — per-layer κ 自由度对照实验

**状态**: 待启动 (2026-07-30)
**Issue**: https://github.com/WENYULIANG123/GeneRec/issues/52

## 目的

Issue #49 Arm B 学到的三层 κ 值 [-0.128, -0.110, -0.123] 彼此接近, 接近"整层统一曲率", 跟最初"per-layer 要有真正不同曲率"目标有落差。本 issue 检验: 这是**自由度不够** (训练机制限制) 还是**数据本就不需要 per-layer 差异化** (数据真实偏好)。

## 假设

- **H1**: 训练机制限制 — 给每层独立 lr_theta + 更宽 κ_max 范围 + 更长 Phase B 训练时间, 三层能学出真正差异 (彼此相差 > 0.3), 且优于趋同结果。
- **H2**: 数据真实偏好 — 不管怎么放开自由度, 三层 κ 都收敛到接近值, 数据不需要 per-layer 差异化 (跟 Task #88 历史结论一致)。后续应把"per-layer 可变曲率"主张收窄成"学出共享的好曲率值"。

## 实验设计

复用 Issue #49 已验证的**解耦调度+统一公式**基础设施, 3 臂对照:

| 臂 | 配置 | 跟 #49 Arm B 的差异 |
|----|------|--------------------|
| **Arm A** | 每层独立 lr_theta (按 vq_index 分别设 lr_theta_l) | 共享 lr_theta=1e-5 → 每层独立 (e.g. L0=1e-4, L1=1e-5, L2=1e-6) |
| **Arm B** | 拓宽 κ_max 范围 (从 ±2.0 → ±1.0 区间检查, 允许更大探索) | kappa_max=2.0 → 1.0 (探索范围稍缩, 避免 saturate) |
| **Arm C** | Phase B 训练延长 (受 Stage 3 早停不牵连, 单独跑满 500 epoch Phase B) | phase_b_epochs=200 → 500 |

每臂记录完整 κ_m 训练轨迹 (不只是终值), 逐 epoch 画出三层曲率分化/收敛过程。

## Stage 1 Gate 1 标准 (跟 #49 一致)

- L0/L1/L2 utilization ≥ 90% 全程 (不只是终态)
- three-digit collision_rate ≤ 0.20
- κ_m 学到非饱和非退化值
- **额外**: 至少 Arm A 或 Arm C 出现三层 κ 彼此相差 > 0.3 的证据

## Stage 2-4 (任一臂 Gate 1 PASS)

- Stage 2: Sinkhorn 推断 + 4-digit dedup → (9922, 4) SID
- Stage 3: T5-mini 200 epoch
- Stage 4: test eval beam=20

## 决策阈值 (vs Issue #49 Arm B R@10=0.1005 + Issue #40 正确 baseline 0.1020)

| 条件 | 含义 | 决策 |
|------|------|------|
| 任何臂三层 κ 差异 > 0.3 + R@10 > 0.1005 | 自由度不够, 学出有意义差异 | H1 部分成立 |
| 任何臂 R@10 > 0.1020 | 真正超过 baseline | GO |
| 三臂三层 κ 都趋同 (不管怎么放开) | 数据不需要 per-layer 差异化 | **H2 成立**, 建议收窄主张 |
| 三臂 R@10 ≤ 0.1005 | 跟 #49 持平或更差 | NO-GO, 验证 H2 |

## 风险与不确定性

1. **GPU 占用**: 3 臂并行 Stage 1 (3 卡 × ~5 min Stage 1) + Stage 3 (3 × ~1.5h)
2. **时间预算**: Stage 1 总 ~5 min + Stage 3 总 ~1.5h + Stage 2-4 各 ~5 min = ~2h

## 产物

| 文件 | 路径 |
|------|------|
| Stage 1 训练 | `products/task339/{arm_a,arm_b,arm_c}/{best_collision_model,phase_b_final}.pth` |
| κ 轨迹 | `products/task339/{arm}/kappa_log.json` |
| Stage 2 SID | `products/task339/{arm}/sid.npy` |
| Stage 3 T5 | `products/task339/{arm}/t5_*/` |
| Stage 4 eval | `verdicts/task339_issue52_stage4_beam20.json` |
| Verdict | `verdicts/task339_issue52_result.md` |

## R11.5 自主决策

- θ_init=-0.02 (跟 Issue #49 Arm B 一致, 唯一学到有意义 κ 的起点)
- Recipe 复用 Issue #49 Arm B, 只改自由度参数