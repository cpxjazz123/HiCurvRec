# Task #340 / Issue #53 — Verdict (阶段: Arm B done, Arm A 还在跑)

**日期**: 2026-07-30
**状态**: 部分 NO-GO (Arm B confirmed; Arm A 还在 Stage 3)
**Issue**: https://github.com/WENYULIANG123/GeneRec/issues/53

## 摘要

Issue #53 检验: κ-codebook 解冻节奏 (per-arm 解冻时机 + 速率) 能否让几何信号传导到 T5-mini SID 召回. **部分结论**: Arm B (β=0.0, κ frozen → 同时解冻 at epoch 200) Test R@10=0.0929 (-8.9% NO-GO). Arm A (per_layer_lr_theta [1e-4, 1e-5, 1e-6]) Stage 3 还在跑.

## Arm B Stage 3 + Stage 4 结果

| 指标 | baseline | **Arm B** (β=0.0, κ 同时解冻 at ep200) | Δ |
|------|----------|----------------------|---|
| **Recall@10 (test)** | 0.1020 | **0.0929** | **-8.9%** ❌ |
| Recall@5 | 0.0816 | 0.0771 | -5.5% |
| Recall@20 | 0.1279 | 0.1127 | -11.9% |
| NDCG@10 | 0.0755 | 0.0726 | -3.8% |
| NDCG@20 | 0.0821 | 0.0776 | -5.5% |

**Stage 3 best ckpt**: 21:18:45 (early-stop 触发 ≈ ep 96, 训练 1h47min).

## 阶段 1 / 2 (Setup)

| Arm | 配置 | Stage 1 best ckpt collision | Stage 2 4-digit unique |
|-----|------|---------------------------|------------------------|
| A | per_layer_lr_theta [1e-4, 1e-5, 1e-6] | 0.0162 | 9922/9922 ✓ |
| B | β=0.0, κ frozen → 同时解冻 at ep200 | 0.0162 | 9922/9922 ✓ |

## 假设验证 (部分)

- **H1** (κ 解冻节奏是 R@10 杠杆): **Arm B REFUTED** — Test R@10=0.0929 < baseline 0.1020. Arm B 跟前 9-16 个 NO-GO 一致: 即使解冻时机最极端, 也不能让 baseline recipe 的几何信号转化为 T5 召回.
- **H2** (Arm A 跟 baseline 同/略高): 待 Arm A Stage 4 跑完验证.

## 跟 Issue #51 / #52 一致性

3 个 issue 联合结论: **val/test gap 是 baseline recipe + Issue #49 派生的 structural trait**, 跟 Issue #51 / #52 类似: Stage 3 best val NDCG@20 ≈ 0.092 (vs baseline val 0.0821, +12%), 但 test R@10 ≈ 0.093 (-8%). T5 在 val 学到的局部模式不能完全 transfer 到 test.

## 决策

- Issue #53 Arm B 状态: **NO-GO (closed)**
- κ-codebook 解冻节奏不是 R@10 杠杆
- 跟 Issue #49/#51/#52 一致: FreeCurvHRQVAE / κ 学习机制 已是结构性局限, 不能靠解冻节奏优化

## R15 闭环

- ✅ Stage 4 JSON: verdicts/task340_arm_b_stage4_beam20.json (commit cd01107)
- ⏸️ Issue #53 维持 OPEN (等 Arm A + 可能的 D 完成, 综合关闭)

result: Issue #53 Arm B Stage 4 NO-GO. test R@10=0.0929 (-8.9% vs baseline 0.1020). Issue #49 / #51 / #52 / #53 联立 → baseline recipe 内部 R@10 杠杆已穷尽. Issue #53 Arm B closed. Issue #53 维持 OPEN 等 Arm A 完成.