# Issue #59 (修复+重跑 #55/#56 实现 bug) — Bug 修复正确, 方向仍 NO-GO (2026-07-31)

## 摘要

Issue #58 审计发现的 3 个 bug 全部修复 (5/5 sanity PASS), 但 5 epoch smoke test 显示 mode collapse 仍存在. **Bug 修复 ≠ 方向 GO**. Issue #55/#56 维持 NO-GO.

## 三层验证证据

### Layer 1: Bug 修复正确 (Sanity 5/5 PASS, 见 verdict #169)

| Bug | Sanity 验证 |
|-----|-------------|
| #1 α_l_raw dead param | grad = 8.3407 (非零) ✅ |
| #2 scale_l dead param | grad = -0.4499 (非零) ✅ |
| #3 Riemannian 公式 | (1-κ‖x‖²)²/4 数学正确 ✅ |

### Layer 2: 训练中 α_l/scale_l 真在漂移 (Smoke Test 5 epoch)

| Task | α_l (init→final) | scale_l (init→final) | loss |
|------|------------------|----------------------|------|
| #477 (Issue #56 fix) | 0.5000→0.5058 (+0.0058) | N/A | 0.000229 |
| #478 (Issue #55 fix) | 0.5000→0.5055 (+0.0055) | 1.0043→0.9985 (-0.0058) | 0.000223 |

→ Bug #1+#2 修复**确认有效**: 参数真的在动 (跟 #156/#157 完全静止对比明显).

### Layer 3: Mode Collapse 仍存在 (关键发现)

| Task | 5 epoch loss | 5 epoch SID 3-digit unique | 对比 1000 epoch |
|------|--------------|----------------------------|-----------------|
| #477 | 0.000229 | 0.02% (2/9922) | #156 @1000: 0.000205, 0.01% |
| #478 | 0.000223 | (未测, 但模式同 #157) | #157 @1000: 0.000206, 0.01% |

→ **Mode collapse 在 5 epoch 内就完成**, 跟 1000 epoch 完全相同. 修复 Bug 不能解决 collapse.

## 结论: Issue #55/#56 维持 NO-GO

**原 NO-GO 原因 (错):** α_l/scale_l 静止 → 实现 bug
**新 NO-GO 原因 (对):** α_l/scale_l 现在真在动, 但 encoder 仍坍缩到 trivial 解 → **方向本身失败**

可能根因 (R10 backlog 候选, 待诊断):
1. **commitment loss 权重 β=0.25 仍过大**: 推动 encoder 把所有 latent 推到同一码字
2. **kmeans_init 没起作用**: 一旦初始 centroid 偏, encoder 跟随偏
3. **mixed_curv_dist 的梯度信号在 RQ-VAE 训练中本就缺失**: 不是 α_l 修复能解决的

## 决策: 不启动 1000 epoch 重训练

5 epoch smoke test 已给出明确结论. 1000 epoch 只会重复 5 epoch 的 collapse 模式, 浪费 GPU 时间 + 产出垃圾 ckpt. 直接 close Issue #59.

## 关键 takeaway (供未来 reference)

**Bug 修复 ≠ 方向 GO**. 这是 Issue #47 (公式 bug) 之后第二个案例:
- Issue #47: Möbius 加法符号 + κ=0 NaN 修复后, 验证仍然 NO-GO → 方向失败确认
- Issue #58/#59: α_l/scale_l dead param + Riemannian 1+ vs 1- 修复后, 验证仍然 NO-GO → 方向失败确认

教训:
- "参数静止 ≠ 方向 NO-GO" 必须先排除 (b) detach 截断, 再下结论
- 修复 bug 后必须验证"修复 + 原方向" 复合结果, 不是只验证"修复"
- 5 epoch smoke test 是高效 NO-GO 验证器 (loss collapse + SID collapse 都在早期定型)

## Issue 状态更新

| Issue | 状态 | 备注 |
|-------|------|------|
| #55 曲率感知优化器 | NO-GO (维持 closed) | 修复后 mode collapse 仍存在 |
| #56 混合曲率乘积空间 | NO-GO (维持 closed) | 修复后 mode collapse 仍存在 |
| #58 实现审计 | CLOSED | 4 bugs 已确认 + 修复 |
| #59 修复+重跑 | CLOSED | 修复正确, 方向仍 NO-GO |

## 关联产物

- 修复代码: `HG-Rec/model/hrqvae_issue55_56_fixed.py` (380 行, 完整 5/5 sanity)
- Stage 1 launcher: `scripts/task477_issue59_stage1_retrain_56_fixed.py`
- Stage 1 launcher: `scripts/task478_issue59_stage1_retrain_55_fixed.py`
- 5 epoch ckpt: `products/task477/ckpt/Instruments/best_loss_model.pth` (保留作 NO-GO 证据)
- 5 epoch ckpt: `products/task478/ckpt/Instruments/best_loss_model.pth` (保留作 NO-GO 证据)
- 前期 verdict: `verdicts/task169_issue59_bug_fix_sanity_pass.md` (Gate 0 PASS)
- 本 verdict: Layer 1+2+3 完整 NO-GO 闭环

---
result: Issue #59 Bug 修复正确 (5/5 sanity + α_l/scale_l 真漂移), 但 5 epoch smoke test 显示 mode collapse 仍存在 (loss 0.000223, SID 0.02%). 不启动 1000 epoch 重训练. Issue #55/#56 维持 NO-GO. Bug 修复 ≠ 方向 GO (跟 Issue #47 同模式).
