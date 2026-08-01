# Task #464 / Issue #171 [方向A Gate2] 零中心有界残差 + κ 同步重校准 Stage1-2 sanity

## 任务来源
- Issue #171 owner 2026-08-01 07:46 派发
- 引用前序证据: #169 已闭环 (commit 4197640), Gate1 通过, Gate2 失败 (α 4.5e-5→1.74, val_R@10=0)
- 失败根因: loss 直接推动 alpha 放大, 没有有效的零中心残差/更新后重校准闭环

## 任务目标
- 修复 A 的 Stage 2 路径, 在保留三层独立 learnable κ (L0 K64 / L1 K128 / L2 K256) 前提下
- 实现 (1) 每层独立的零中心有界残差门控 (LN + tanh*bound)
- 实现 (2) κ 更新后同步重算代码本尺度 + 距离计算 + 统一公式 (#47 复用)
- 用真实 Stage1 κ/scale metadata 做短程单 seed sanity
- 3 epoch sanity + 记录 α/κ/梯度轨迹 + forward save/load 一致性
- 不得复用 #169 已失败的无界 alpha-growth 路径

## 实施
- `scripts/task464_issue171_stage2_zero_centered_bounded_residual_kappa_recalibration.py` (634 行)
- `ZeroCenteredBoundedResidualAdapter`: tanh(0) * bound = 0 起点, |residual| <= 0.5 饱和
- `kappa_logit_l` (per-layer independent): softplus(0) ≈ 0.693 起点
- `synchronize_kappa()`: κ 更新后重校准 codebook 尺度 + 距离
- GPU 1 (R7: #450 GPU 0, #464 GPU 1, #465 GPU 2)
- batch_size=32, lr_conditioner=1e-3, lr_layernorm=1e-4, lr_kappa=1e-4

## 验收 Gate 标准
- precheck: L0 K64 / L1 K128 / L2 K256 三层独立 learnable κ
- Gate 1: Stage 1 真实 metadata 导出可审计 (含三层 shape/hash)
- Gate 2: Stage 2 短程 sanity 通过
  - alpha/κ 有界且非爆炸
  - 代码本/距离随 κ 同步重校准
  - SID 流可复现
  - 否则 NO-GO
- Gate 3: 仅在 Gate 2 PASS 后接入 Stage 3 T5
- Gate 4: 仅在 Gate 3 PASS 后做单 seed Stage 4 正式评估

## 强制
- R23: val_R@10=0 跨 ≥2 epoch → kill + NO-GO
- R12: ckpt 落盘 (delete old + save new)
- R15 + R17 + R18 + R20 + R21 + R24: commit + push + 4-Gate 详细 + commit hash 强制
