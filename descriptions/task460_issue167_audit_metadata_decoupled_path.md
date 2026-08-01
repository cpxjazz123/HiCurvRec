# Task #460 / Issue #167 方向A Gate 2 κ元数据条件化路径与实际wrapper一致性审计

## 背景

Issue #167 owner 2026-08-01 06:50 派发: 审计 task458 (#165) wrapper 实际代码路径 vs Issue spec (零中心 + gain init=0 + 元数据条件化). 复盘 #165 NO-GO 根因 (commit 53a0404): alpha 跨 8 epoch 持续 1.5× growth 达 12.32, 跟 #162 alpha clamp 失败模式轨迹一致. **不重复 #165 长训**.

## R18 4 维度对比 (vs #165 closed NO-GO 53a0404)

| 维度 | #165 (closed) | #167 (NEW 审计) |
|------|---------------|-----------------|
| **D1 spec** | 零中心 + gain init=0 + 实际工具 | **代码路径审计, 实施诊断** |
| **D2 实施** | 复用了 task452 wrapper (跟 #162 失败模式同源) | 实际 wrapper 类 + forward + gain/alpha + 残差范数 + metadata 打印 |
| **D3 失败机制** | α 持续 1.5× growth/epoch | 实施-执行不一致 |
| **D4 引用** | arXiv:2405.13979 | 同一引用 + adaptive normalization / task-geometry decoupling |

→ **D1/D2 显著不同**: 从"长训 + 期望表现" 转为 "代码路径审计 + 实证诊断". R18 强制实验.

## 任务内容 (Gate 2 最小复核)

- **审计 6 项**:
  1. 实际 wrapper 类名 (应 HG_Rec_with_ZeroCenteredLayerNormAdapter)
  2. forward 调用路径 (是否真的 zero-centered, gain 是 init=0 吗)
  3. gain/alpha 数值 (是否真的从 0 起步, 路径饱和?)
  4. 残差增量范数 (是否真元数据条件化, 不放大 T5 残差)
  5. 传入 metadata (kappa_meta / sid_meta 真实注入)
  6. 与 #162 失败模式路径对比 (执行一致性)
- **短训**: 10 epoch, 固定 #157/#159 SID, 同一数据切分, 单 seed
- **关键监控**: val_R@10 逐 epoch (类似 task450 d54aa68 patch), 连续两次 val_R@10=0 → 立即 NO-GO
- **禁止 fallback**: 禁止用 loss 下降替代 val_R@10

## 关键决策点 (R11.5 自主决策)

1. **R7 分配 GPU 1** (#450 占用 GPU 0, 其他空闲)
2. **R22 立即开工**: owner 派发审计, 不允许等下一 tick
3. **R11.5 兜底 4. 简单实用**: 复用 task458 模板 + 加 d54aa68 val_R@10 早停 patch + 加实际 wrapper 类/gain/alpha/残差范数/metadata 打印
4. **R19 跨 GPU 并行**: #460 + #461 (Issue #168) GPU 1/2 同时跑
5. **R12 ckpt 落盘**: 每 epoch 末保存
6. **R23 监控**: 跨 2 checkpoint val_R@10=0 → 立即 §25 kill + NO-GO

## 验收

- Gate 2 PASS: 实际 wrapper 路径通过审计 (gain init=0 真实生效, 残差范数 ≤ 阈值, 每层 metadata 注入)
- val_R@10 跨 10 epoch ≠ 0 (持续健康)
- 任一 Gate FAIL → 立即 NO-GO + close issue (R16)
