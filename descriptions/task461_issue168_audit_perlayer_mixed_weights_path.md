# Task #461 / Issue #168 方向B Gate 2 逐层混合权重路径与归一化实现一致性审计

## 背景

Issue #168 owner 2026-08-01 06:50 派发: 审计 task459 (#166) wrapper 实际代码路径 vs Issue spec (逐层归一化 + simplex 权重 + 零中心 residual adapter). 复盘 #166 NO-GO 根因 (commit 53a0404): Gate 3 机制 PASS 但无 val_R@10 验证, alpha 跟 #163 失败模式同类. **不重复 #166 模板长训**.

## R18 4 维度对比 (vs #166 closed NO-GO 53a0404)

| 维度 | #166 (closed) | #168 (NEW 审计) |
|------|---------------|-----------------|
| **D1 spec** | 逐层归一化 + simplex 权重 + 零中心 adapter | **代码路径审计, 实施诊断** |
| **D2 实施** | 复用了 task452 wrapper (跟 #163 失败模式同源) | 三分量范数 + 权重和 + κ/权重梯度 + 实际 forward 参数 |
| **D3 失败机制** | α 持续 1.5× growth/epoch | 实施-执行不一致 + 权重不归一 |
| **D4 引用** | arXiv:2307.04514 | 同一引用 + tangent-space estimation / adaptive normalization |

→ **D1/D2 显著不同**: 从"长训 + 期望表现" 转为 "代码路径审计 + 实证诊断". R18 强制实验.

## 任务内容 (Gate 2 最小复核)

- **审计 6 项**:
  1. 每层三分量 (固定双曲 + 固定欧氏 + learnable-κ) 真实存在
  2. 逐层归一化 (norm-preserving/tangent-space) 真实生效
  3. simplex 权重 (0-1 求和=1) 真实生效
  4. 零中心 residual adapter (gain init=0) 真实生效
  5. κ/权重梯度健康 (≠ 0, ≠ NaN)
  6. 残差增量范数 (≤ 阈值, 不爆炸)
- **短训**: 10 epoch, 固定 #158 SID, 同一数据切分, 单 seed
- **关键监控**: val_R@10 逐 epoch (类似 task450 d54aa68 patch), 连续两次 val_R@10=0 → 立即 NO-GO
- **禁止 fallback**: 禁止用 loss 下降替代 val_R@10, 禁止权重不归一仍继续

## 关键决策点 (R11.5 自主决策)

1. **R7 分配 GPU 2** (#450 占用 GPU 0, #460 占用 GPU 1)
2. **R22 立即开工**: owner 派发审计, 不允许等下一 tick
3. **R11.5 兜底 4. 简单实用**: 复用 task459 模板 + 加 d54aa68 val_R@10 早停 patch + 加三分量范数/权重和/κ梯度/残差范数打印
4. **R19 跨 GPU 并行**: #460 (GPU 1) + #461 (GPU 2) 同时跑
5. **R12 ckpt 落盘**: 每 epoch 末保存
6. **R23 监控**: 跨 2 checkpoint val_R@10=0 或权重不归一 → 立即 §25 kill + NO-GO

## 验收

- Gate 2 PASS: 实际权重路径通过审计 (三层独立 κ + 真实逐层归一化 + simplex 权重 + 零中心 adapter)
- val_R@10 跨 10 epoch ≠ 0 (持续健康)
- 任一 Gate FAIL → 立即 NO-GO + close issue (R16)
