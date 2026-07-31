# Task #354 / Issue #63 Gate 1 — Stage 1 NO-GO (USAGE-KILL)

**日期**: 2026-07-31
**前置**: Gate -1 8/8 PASS, Gate 0 8/8 PASS
**任务**: Gate 1 Stage 1 训练 (vanilla κ-decouple FreeCurvHRQVAE, 200 epoch)
**结果**: ❌ USAGE-KILL @ epoch 30, NO-GO 收口

---

## 1. 训练结果

| Epoch | L0_util | L1_util | L2_util | L0_coll | L1_coll | L2_coll | avg_loss |
|-------|---------|---------|---------|---------|---------|---------|----------|
| 5     | 0.016   | 0.008   | 0.012   | 0.016   | 0.008   | 0.012   | 0.0004   |
| 10    | 0.016   | 0.008   | 0.004   | 0.016   | 0.008   | 0.004   | 0.0003   |
| 15    | 0.016   | 0.008   | 0.004   | 0.016   | 0.008   | 0.004   | 0.0002   |
| 20    | 0.016   | 0.008   | 0.004   | 0.016   | 0.008   | 0.004   | 0.0002   |
| 25    | 0.016   | 0.008   | 0.004   | 0.016   | 0.008   | 0.004   | 0.0002   |
| **30**| **0.016**| **0.008**| **0.004**| **0.016**| **0.008**| **0.004**| **0.0002**|

→ ❌ **USAGE-KILL @ ep 30**: min_util=0.4% < 0.9 (阈值 90%)

## 2. κ 轨迹

| Epoch | κ_L0   | κ_L1   | κ_L2   |
|-------|--------|--------|--------|
| 1     | -0.0125| -0.0123| -0.0120|
| 5     | -0.0198| -0.0198| -0.0199|
| 10    | -0.0200| -0.0201| -0.0206|
| 20    | -0.0200| -0.0201| -0.0207|
| 30    | -0.0200| -0.0201| -0.0207|

→ κ 收敛到 ≈ -0.02 (近 0, 几乎 Euclidean), 但 utilization 完全崩塌

## 3. 根因分析

**Phase 0 mode collapse** (per memory `phase0-mode-collapse.md`):
- kmeans_init=True + kmeans_iters=10 初始化不充分
- sk_eps=[0,0,0] (Stage 1 OFF per Issue #63 spec) → 训练时无 Sinkhorn rebalance
- κ-decouple + κ 收敛到 0 + Euclidean regime + 32d codebook → encoder 把所有 latent 推到同一码字附近
- 量化损失 (quant=0.0000) 几乎为 0 → 模型走捷径, 不需要 multi-modal codebook

## 4. R12 ckpt 落盘验证

- ✅ killed_model.pth 已落盘 (USAGE-KILL 触发时保存)
- ✅ best_loss_model.pth @ ep 25 (loss=0.0002, 但 util 已崩塌)

## 5. Issue #63 综合结论

| Gate | 结果 | 备注 |
|------|------|------|
| Gate -1 (zero-GPU) | ✅ 8/8 PASS | 实施基础 + 隔离 OK |
| Gate 0 (zero-GPU sanity) | ✅ 8/8 PASS | 数值稳定性 OK |
| **Gate 1 (Stage 1 训练)** | ❌ **USAGE-KILL** | min_util 1.6% < 90% |
| Gate 2 (Stage 2 推断) | N/A (未进入) | — |
| Gate 3 (Stage 3+4 评估) | N/A (未进入) | — |

→ **Issue #63 NO-GO 收口**

## 6. 失败模式分类

按 memory `phase0-mode-collapse.md` 分类:
- 本次 = **Euclidean mode collapse** (κ 收敛到 ≈0)
- 跟 Task #179 pure Euclidean 对比: task179 util 应高得多 (未量化), 需后续分析差异
- 跟 Task #178/task180 200 epoch κ=2 / κ=10 对比: 那些是 hyperbolic mode collapse (码字全推 boundary), 本次是 Euclidean mode collapse (码字几乎不散开)

## 7. R11.5 决策

- Issue #63 Gate 1 NO-GO 收口
- κ-decouple + no Sinkhorn + β=0.25 + 32d → 跟 #55/#56 SID collapse 同模式
- 修复方向: 加 Sinkhorn 在 Stage 1 训练 (sk_eps>0) 或 kmeans_iters 增加到 ≥100
- 不再重试当前 recipe (避免 GPU 浪费)

## 8. 产物清单

| 路径 | 内容 |
|------|------|
| products/task354/ckpt/Instruments/killed_model.pth | USAGE-KILL 时刻 ckpt |
| products/task354/ckpt/Instruments/best_loss_model.pth | best_loss ckpt @ ep 25 |
| products/task354/ckpt/Instruments/training_history.json | 30 epoch 训练历史 |
| logs/task354_issue63_stage1_train_*.log | 完整训练 log |
| scripts/task354_issue63_stage1_train.{py,sh} | 训练脚本 (保留) |

---

result: Issue #63 Gate 1 NO-GO (USAGE-KILL @ ep 30, util 1.6%/0.8%/0.4% << 90%). κ-decouple + no Sinkhorn + β=0.25 Euclidean mode collapse. Issue #63 全线 NO-GO 收口 (Gate -1/0 PASS 但 Gate 1 fail). 不重试当前 recipe.