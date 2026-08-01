# Task #458 / Issue #165 方向A Gate 2 元数据解耦残差注入 短训验证

## 背景

Issue #165 owner 2026-08-01 派发: 验证 κ/尺度元数据与T5残差幅值解耦方案. 方向A 上一 issue #162 已 wrap up (a2adffe NO-GO, 根因 alpha clamp 立即饱和导致残差过大). 新 spec 提出"零中心仿射调制 + 逐层 norm 控制"假设替代 alpha clamp 路径.

## R18 4 维度对比 (vs #162 closed NO-GO)

| 维度 | #162 (closed NO-GO, a2adffe) | #165 NEW |
|------|------------------------------|----------|
| **D1 spec** | alpha clamp 残差放大 | **零中心 + 范数受控 + gain/bias init=0** |
| **D2 实施** | `alpha = torch.clamp(alpha_raw, max=1.0)` | 零中心仿射调制 + 逐层 norm 控制 |
| **D3 失败机制** | alpha 立即饱和 → 残差过大 → T5 LN 崩溃 | 元数据条件化替代残差放大, 残差起步 0 |
| **D4 引用** | Task #84 HG-Rec baseline | arXiv:2405.13979 curvature-aware optimization |

→ **D1/D2/D3 显著不一致** (R18 强制实验, 禁用"路径同构"沿用).

## 任务内容 (Gate 2 短训 sanity)

- **Gate 1 预检**: 验证三层独立 κ_0/κ_1/κ_2 + L0 K64/L1 K128/L2 K256 + metadata 导出
- **Gate 2 短训**: 冻结 #157/#159 同一 SID, 跑 5-10 epoch sanity
- **关键报告**: SID 输入 hash, metadata→token 数据流, loss 下降, val_R@10 协议, 有限非零梯度, T5 残差增量有限
- **关键指标**: 连续两次 val_R@10 ≠ 0 (避免 #162 NO-GO 重复)
- **失败立即停**: 单次 val_R@10=0 → 立即 R23 kill + 写 NO-GO verdict + close issue

## 关键决策点 (R11.5 自主决策)

1. **R7 分配 GPU 1** (R7 + R19 并行 + #450 占 GPU 0)
2. **Stage 3 短训 epoch = 5** (R22 立即开工, 不等 200 epoch 完成)
3. **R12 强制 ckpt 落盘** (每 epoch 末 + R11.5 决策)
4. **R23 监控**: 跨 2 checkpoint val_R@10=0 立即 kill

## 验收

- 6 项 Stage 4 指标齐全 + R@10 > 0.1020 → Target reached
- 任一 Gate FAIL → 立即 NO-GO + close issue (R16)
