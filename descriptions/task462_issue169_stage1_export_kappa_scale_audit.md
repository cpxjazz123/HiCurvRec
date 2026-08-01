# Task #462 / Issue #169 方向A Gate 1-2 κ元数据条件化wrapper真实 Stage 1 路径审计

## 背景

Issue #169 owner 2026-08-01 07:00 派发: 方向A Gate 1-2 κ元数据条件化wrapper **真实** Stage 1 路径审计. 跟 #167 的差异 = Stage 1 **真实** κ/scale export (从 RQ-VAE 模型 extract), 不是硬编码 metadata. 复用 #160 终点 (任务 #460/#167 已确认 wrapper 实质合规, 但 val_R@10=0), owner 进一步要真实 Stage 1 export + 3 epoch sanity.

## R18 4 维度对比 (vs #167 closed NO-GO 11b0ee5)

| 维度 | #167 (closed) | #169 (NEW) |
|------|---------------|------------|
| **D1 spec** | wrapper 路径审计, 硬编码 metadata | **真实 Stage 1 export + κ/scale 三层独立learnable + 3 epoch sanity** |
| **D2 实施** | task452 模板, alpha=0.117→1.854 1 epoch 16× | **load Stage 1 RQ-VAE model → extract codebook → compute per-item κ/scale** |
| **D3 失败机制** | α 1 epoch 涨 16× | wrapper 用了硬编码 metadata, 缺真实 Stage 1 κ/scale |
| **D4 引用** | arXiv:2405.13979 | 同一引用 + real RQ-VAE integration |

→ **D1/D2 显著不同**: 从"代码审计" 转为"真实 Stage 1 export + 训练验证". R18 强制实验.

## 任务内容 (Gate 1-2 完整, Gate 3-4 ⏸ per spec)

- **Gate 1 Stage 1 真实 export**:
  1. 加载 task84 Stage 1 RQ-VAE best_loss_model.pth
  2. 提取三层独立 codebook (K=64/128/256) + 导出 per-item κ_l (per layer curvature) + scale_l (||c||)
  3. 计算三层 hash (per layer codebook hash + 导出 metadata hash) — 必须非空, 一致
  4. 失败条件: Stage 1 model 不存在 / codebook 不能 extract / hash 缺失 → 立即 NO-GO
- **Gate 2 Stage 2 真实 metadata 注入 + 3 epoch sanity**:
  1. 用真实 per-item κ/scale metadata 喂给 T5 wrapper (替代硬编码 [0, 0.333, 0.333, 0.334])
  2. 3 epoch sanity (per issue spec, 不重复长训)
  3. 报告: SID hash + metadata 流 + 残差范数 + loss + val_R@10 逐 epoch + κ/权重梯度 + ckpt
  4. 失败条件: α 单调增长 / val_R@10=0 跨 ≥2 epoch / metadata 未进入 forward → 立即 NO-GO
- **Gate 3 save/load + forward consistency + 无 NaN/Inf**: ⏸ STOP per spec (Gate 2 必须 PASS)
- **Gate 4 single-seed Task84 评估**: ⏸ STOP per spec (Gate 3 必须 PASS)

## 关键决策点 (R11.5 自主决策)

1. **R7 分配 GPU 1** (#450 GPU 0 在跑, GPU 1 空闲)
2. **R22 立即开工**: owner 派发, 不允许等下一 tick
3. **R11.5 兜底 4. 简单实用**: 复用 task460 模板 + 加 Stage 1 export 函数 + 改 num_epochs=3 (per spec)
4. **R19 跨 GPU 并行**: task462 (GPU 1) + task463 (GPU 2) 同时跑
5. **R12 ckpt 落盘**: 每 epoch 末保存
6. **R23 监控**: 跨 2 checkpoint val_R@10=0 → 立即 §25 kill + NO-GO

## 验收

- Gate 1 Stage 1 export: 三层独立 codebook 真实 extract, hash 一致
- Gate 2 sanity: 真实 metadata 注入, val_R@10 跨 3 epoch 健康 (持续 > 0)
- 任一 Gate FAIL → 立即 NO-GO + close issue (R16)

## 框架合规 (R5 + R2)

- L0 K=64, L1 K=128, L2 K=256 三层独立learnable κ
- 同步 codebook 本 + 距离和 SID 流
- ❌ 禁止 global κ, fixed-only, 纯欧氏绕过, proxy SID, 完整解冻 T5
