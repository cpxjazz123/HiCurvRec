# Task #239 — Task #229 Gate 2: REVERSED per-layer c proposal (c=[L0高, L1/L2低])

## 来源

- Task #229 Gate 1 (verdicts/task229_gate1_crash_geometry_result.md) 发现: **崩溃 cliff 在 c ∈ [30, 100], 但崩溃层是 L1/L2 不是 L0 (跟 verdict #230 预测相反)**
- 用户原提案 c=[13.45, 96.21, 199.78] (L0 低, L1/L2 高) 已证伪 — L1/L2 在 c=100 已坍
- Gate 2 目标: 验证 **REVERSED 提案 c=[L0 高, L1/L2 低]** 是否安全

## Gate 1 关键发现 (驱动 Gate 2 REVERSED 提案)

| c | L0 c·‖e‖²_max | L1 c·‖e‖²_max | L2 c·‖e‖²_max | 状态 |
|---|---|---|---|---|
| 1 | 0.140 ✓ | 0.019 ✓ | 0.010 ✓ | 全 healthy |
| 10 | 0.317 ✓ | 0.180 ✓ | 0.099 ✓ | 全 healthy |
| 30 | 0.594 ✓ | 0.272 ✓ | 0.185 ✓ | 全 healthy |
| 100 | 1.186 ✓ | **53.5** ❌ | **49.8** ❌ | L1/L2 坍缩 |
| 300 | **3.98** ⚠️ | **54.6** ❌ | 17.8 ❌ | L1/L2 坍缩 |

**关键洞察**: L0 范数随 c 增大递减 (Poincaré ball 压缩), 容忍高 c; L1/L2 范数最小但被 c 拉爆, 严格限制 c.

## REVERSED 提案 (Gate 2 验证)

| 方案 | c=[L0, L1, L2] | 假设 | 风险 |
|------|---------------|------|------|
| **A1** | c=[10, 1, 1] | L0 提曲率, L1/L2 维持 baseline | 边际效应 (跟 baseline 接近) |
| **A2** ⭐ | c=[30, 3, 3] | L0 显著提曲率, L1/L2 维持低曲率 | L0 c=30 已被 Gate 1 证 healthy |
| **A3** | c=[30, 1, 1] | L0 显著提曲率, L1/L2 完全 baseline | 同 A2 但 L1/L2 不用 c=3 |
| **B** | c=[10, 10, 10] | 全局均匀 c=10 (Gate 1 best survivor) | 没 per-layer 差异 |

## Gate 2 候选 (R11.3 自主推荐)

- **核心 3 变体**: A1 (c=[10,1,1]) + A2 (c=[30,3,3]) + B (c=[10,10,10] 对照)
- **目的**: 验证 L0 单独提 c 是否能保留 baseline 健康 + 是否有 R@10 边际增益
- **不跑 Stage 3+4** (跟 Gate 1 R11.3 一致: Q1 是"测崩溃点",不是"测 R@10")

## 配置 (跟 Gate 1 完全一致)

| 项 | 值 |
|----|-----|
| ckpt epoch | 200 (Gate 1 用) |
| batch_size | 1024 |
| epochs | 200 |
| Stage 2/3/4 | 跳过 (只跑 Stage 1 几何) |
| GPU | 3 GPU 并行 (R7: 全空闲) |
| seed | 42 |
| 数据集 | Musical_Instruments (24588 → 5-core 9922) |
| CLI flag | `--curvatures 30.0,3.0,3.0` (per-layer list, 跟 Gate 1 launchers 一致) |
| 其他 | 干净 config (r_target=None, w_div=0, w_angular=0, kappa_mode=fixed) |

## 决策阈值

| Gate | 指标 | 阈值 | Pass 含义 | Fail 含义 |
|------|------|------|-----------|-----------|
| Gate 2 | per-arm collision @ best_collision | < 0.10 | REVERSED 提案不会坍缩 | REVERSED 也坍缩, 整条 per-layer 路线收口 |
| Gate 2 | per-arm ‖e‖_max L1/L2 | < 0.30 | L1/L2 范数受控 | L1/L2 再次失控 |
| Gate 2 | per-arm ‖e‖_max L0 | < 0.30 (Gate 1 c=30 L0 = 0.141) | L0 高 c 也受控 | L0 高 c 失控 (跟 Gate 1 预测相反) |

## 资源预算

- 3 变体 × 1 GPU × 200 epoch = ~30min wall clock (R7: 3 GPU 并行)
- Stage 2/3/4 跳过 → 总成本 ≤ 1h

## 步骤

1. ✅ 写 Task #239 description (本文件)
2. 写 3 launchers (A1/A2/B)
3. 启动 3 GPU 并行 Stage 1 训练
4. 等全部完成 (~30min)
5. 提取 best_collision + per-layer ‖e‖_max 跟 Gate 1 对比
6. 写 verdict: Gate 2 PASS/FAIL
7. 决定是否进 Stage 3+4 (测 R@10)

## 产物

- scripts/task239_gate2_c{10_1_1,30_3_3,10_10_10}_stage1.sh
- products/task239/gate2_{A1,A2,B}/best_collision_model.pth (R12)
- descriptions/task239_gate2_reversed.json (per-arm 测量结果)
- verdicts/task239_gate2_reversed_result.md (PASS/FAIL + Gate 3 决策)

## 依赖

- Task #229 Gate 1 verdict (PASS) — c=10-30 健康, c=100 L1/L2 坍
- Task #229 Gate 0 (PASS) — ‖e‖ 分布 + max_c2 预测
- HG-Rec baseline Task #84 (R@10=0.1020) 作为对照

## Status

R11.3 自主决策 (R10 主动推进): Issue #10 redesign 等用户, Issue #9 closed, 利用空闲 GPU 跑 Gate 2 Stage 1 几何验证 (~30min). **不主动跑 Stage 3+4** (Gate 2 PASS 才进, Gate 2 FAIL 收口).
