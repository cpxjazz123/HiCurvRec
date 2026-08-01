# Task #463 / Issue #170 方向B Gate 1-2 逐层混合曲率wrapper真实 Stage 1 路径审计

## 背景

Issue #170 owner 2026-08-01 07:00 派发: 方向B Gate 1-2 逐层混合曲率wrapper **真实** Stage 1 路径审计. 跟 #168 的差异 = Stage 1 **真实** 三分量 (固定双曲+固定欧氏+learnable-κ) extract, 不是硬编码 metadata. 复用 #160 终点 (任务 #461/#168 已确认 wrapper 实质合规 + 三组件真实存在, 但 val_R@10=0), owner 进一步要真实 Stage 1 export + 3 epoch sanity.

## R18 4 维度对比 (vs #168 closed NO-GO 11b0ee5)

| 维度 | #168 (closed) | #170 (NEW) |
|------|---------------|------------|
| **D1 spec** | wrapper 路径审计, 硬编码 metadata 三组件 | **真实 Stage 1 export + 每层三分量 (固定双曲+固定欧氏+learnable-κ) + 3 epoch sanity** |
| **D2 实施** | task452 模板, α 0.117→1.854 1 epoch 16× | **load Stage 1 RQ-VAE → extract codebook → compute per-item 每层三分量** |
| **D3 失败机制** | α 1 epoch 涨 16× | wrapper 用了硬编码 metadata, 缺真实 Stage 1 三分量 |
| **D4 引用** | arXiv:2307.04514 | 同一引用 + real RQ-VAE integration + per-component |

→ **D1/D2 显著不同**: 从"代码审计" 转为"真实 Stage 1 三分量 export + 训练验证". R18 强制实验.

## 任务内容 (Gate 1-2 完整, Gate 3-4 ⏸ per spec)

- **Gate 1 Stage 1 真实三分量 export**:
  1. 加载 task84 Stage 1 RQ-VAE best_loss_model.pth
  2. 提取三层独立 codebook (K=64/128/256)
  3. 对每层 codebook 每个 centroid, 计算三分量:
     - 固定双曲分量: expmap0(centroid, c=1) (Poincaré ball)
     - 固定欧氏分量: centroid (直接)
     - learnable-κ 分量: 通过 learnable_kappa_l (per layer)
  4. 导出 per-item 三分量范数 + κ_l (per layer)
  5. 计算三层 hash (per layer codebook hash + 三分量 hash) — 必须非空, 一致
  6. 失败条件: Stage 1 model 不存在 / codebook 不能 extract / 三分量缺失 / hash 缺失 → 立即 NO-GO
- **Gate 2 Stage 2 真实三分量 metadata 注入 + 3 epoch sanity**:
  1. 用真实 per-item 三分量 metadata 喂给 T5 wrapper (替代硬编码 [0, 0.333, 0.333, 0.334])
  2. 3 epoch sanity (per issue spec, 不重复长训)
  3. 报告: SID hash + 三分量范数 + 权重和 + κ/权重梯度 + loss + val_R@10 逐 epoch + ckpt
  4. 失败条件: 任一层权重不归一 / 三分量被静态欧氏化 / α 增长失控 / val_R@10=0 跨 ≥2 epoch → 立即 NO-GO
- **Gate 3 save/load + forward consistency + 无 NaN/Inf**: ⏸ STOP per spec
- **Gate 4 single-seed Task84 评估**: ⏸ STOP per spec

## 关键决策点 (R11.5 自主决策)

1. **R7 分配 GPU 2** (#450 GPU 0, task462 GPU 1, GPU 2 空闲)
2. **R22 立即开工**: owner 派发, 不允许等下一 tick
3. **R11.5 兜底 4. 简单实用**: 复用 task461 模板 + 加 Stage 1 三分量 export 函数 + 改 num_epochs=3
4. **R19 跨 GPU 并行**: task462 (GPU 1) + task463 (GPU 2) 同时跑
5. **R12 ckpt 落盘**: 每 epoch 末保存
6. **R23 监控**: 跨 2 checkpoint val_R@10=0 或权重不归一 → 立即 §25 kill + NO-GO

## 验收

- Gate 1 Stage 1 export: 三层三分量真实 extract, hash 一致
- Gate 2 sanity: 真实三分量 metadata 注入, val_R@10 跨 3 epoch 健康 (持续 > 0)
- 任一 Gate FAIL → 立即 NO-GO + close issue (R16)

## 框架合规 (R5 + R2)

- L0 K=64, L1 K=128, L2 K=256 三层独立learnable κ
- 每层三分量 + 独立混合权重
- ❌ 禁止 global κ, fixed-only, 纯欧氏绕过, proxy SID, 完整解冻 T5
