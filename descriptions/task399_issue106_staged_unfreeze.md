# Task #399 / Issue #106 [方向B Gate1] staged product unfreeze 避免 L1L2 坍缩

**日期**: 2026-07-31
**触发**: Issue #106 [方向B Gate1] staged product unfreeze 避免L1L2坍缩
**前置**:
- task397 #101 per-component std norm + mean aggregation Gate 1 NO-GO (agree3 L1/L2 ≈ 0)
- task391 #98 audit FAIL: aggregate-sum argmin bias toward component 0
- 跨 19 方向 × 25 verdict 累积 NO-GO: per-component scale fix 不解决 product 路径

**任务**: 实施 staged product unfreeze (L0 → L0+L1 → L0+L1+L2) + Stage 1 50 epoch 实证
**结果**: ⏳ TRAINING (GPU 2, parallel to task396b GPU 0)

---

## Issue #106 Gate 1 spec (R17 强制 4 Gate 顺序)

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE product path): ⏳ TRAINING
- **目标**: L1/L2 util ≥ 90% (vs task397 14.84%/12.11%)
- **修复路径**: staged unfreeze — 阶段性解冻防止 component scale bias 主导
  - Epoch 0-15: L0 trainable, L1/L2 frozen (inherited kmeans init from L0 encoded residual)
  - Epoch 15-30: L0+L1 trainable, L2 frozen
  - Epoch 30-50: all 3 layers trainable
- **预期**: L0/L1/L2 util 都 ≥ 90%, per-component 不会全坍缩到 component 0
- **实施**: scripts/task399_issue106_staged_unfreeze.py
- **GPU**: 2 (R7 空闲)

### Gate 2/3/4: ⏸ STOP per Issue #106 spec
- **原因**: Issue #106 spec 仅 Gate 1, Gate 2/3/4 不在本 issue 范围

---

## 训练配置

- dataset: Instruments (Musical_Instruments)
- architecture: FreeCurvVectorQuantization M=3 block_dims=[11,11,10] e_dim=32
- num_epochs: 50 (staged: 0-15 L0 only, 15-30 L0+L1, 30-50 all)
- batch_size: 256, lr: 1e-4
- seed: 42
- GPU: 2 (CUDA_VISIBLE_DEVICES=2)
- ckpt path: products/task399_issue106_staged_unfreeze/ckpt/

---

## 关键产物

- **ckpt**: products/task399_issue106_staged_unfreeze/ckpt/issue106_ckpt.pt (R12 强制)
- **evidence**: products/task399_issue106_staged_unfreeze/evidence.json
- **log**: products/task399_issue106_staged_unfreeze/nohup.out
- **verdict**: verdicts/task399_issue106_staged_unfreeze_v2.md

---

## 后续 (per R22 + R19)

1. **task398: Stage 4 R@K eval** — 等 task396b Stage 3 训练完成后启动 (GPU 1)
2. **Issue #106 闭环** — 写 verdict + close issue
3. **跨方向联立**: task397 + task399 联立判定 product path 是否还有杠杆 (per R18 实证)