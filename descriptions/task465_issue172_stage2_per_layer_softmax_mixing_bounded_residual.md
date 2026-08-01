# Task #465 / Issue #172 [方向B Gate2] 逐层 softmax 混合权重 + 零中心范数受限残差 Stage1-2 sanity

## 任务来源
- Issue #172 owner 2026-08-01 07:46 派发
- 引用前序证据: #170 已闭环 (commit 4197640), Gate1 通过, Gate2 失败 (α 4.5e-5→1.95, 三层 κ 同步漂移)
- 失败根因: 混合分量仍被 loss 共同推动, 未形成可审计的逐层混合比例学习

## 任务目标
- 修复 B 的 Stage 2, 保留三层独立 learnable κ (L0 K64 / L1 K128 / L2 K256)
- 实现 (1) 每层固定双曲 + 欧氏分量, 用逐层 softmax 混合权重
- 实现 (2) 零中心 + 范数受限 residual 连接 (LN + clamp ±bound)
- 实现 (3) 禁止共同 α 放大 κ (per-layer mixing_logit_l 独立)
- 用真实 Stage1 三分量 metadata 做短程单 seed sanity
- 3 epoch sanity + 记录每层混合权重/κ/梯度/分量范数 + SID 流一致性
- 不得复用 #170 共同漂移路径

## 实施
- `scripts/task465_issue172_stage2_per_layer_softmax_mixing_bounded_residual.py` (674 行)
- `PerLayerSoftmaxMixingBoundedResidualAdapter`: 
  - `mixing_logit_l` (per-layer, 3 logits) → softmax over [hyp, eucl, learnable_κ]
  - 起点均匀 (mixing_init_logits=[0,0,0])
  - `kappa_logit_l` per-layer 独立 (跟 #464 一致)
  - 零中心范数受限: clamp(LN(x), -bound, bound)
- GPU 2 (R7: #450 GPU 0, #464 GPU 1, #465 GPU 2)
- batch_size=32, lr_conditioner=1e-3, lr_mixing=1e-3, lr_kappa=1e-4, lr_layernorm=1e-4

## 验收 Gate 标准
- precheck: L0 K64 / L1 K128 / L2 K256 三层独立 learnable κ
- Gate 1: Stage 1 三层三分量 metadata 导出可审计 (含 shape/hash)
- Gate 2: Stage 2 sanity 通过
  - 逐层混合权重可学习且不塌缩/爆炸
  - κ 保持逐层独立 (max-min > 1e-4)
  - mixing 权重也 per-layer 独立 (std > 1e-4)
  - SID 流可复现
  - 否则 NO-GO
- Gate 3: 仅在 Gate 2 PASS 后接入 Stage 3 T5
- Gate 4: 仅在 Gate 3 PASS 后做单 seed Stage 4 正式评估

## 强制
- R23: val_R@10=0 跨 ≥2 epoch → kill + NO-GO
- R12: ckpt 落盘
- R15 + R17 + R18 + R20 + R21 + R24: commit + push + 4-Gate 详细 + commit hash 强制
