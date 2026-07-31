# Task #401 / Issue #108 [方向B Gate1] anchored residual best-epoch 窗口复验

**日期**: 2026-07-31
**触发**: Issue #108 [方向B Gate1] anchored residual best-epoch 窗口复验
**前置**:
- task400 #107 anchored residual product NO-GO final (L1/L2 31.25%/15.62% @ ep49)
- **task400 epoch 35 关键信号**: L0/L1/L2 = 100/100/98.4% 真 PASS (mid-training)
- task397/399 都没测 mid-training → 训练时长是 product path 隐性变量

**任务**: 用 task400 同款 anchored residual patch + 35 epoch 截断复验
**结果**: ⏳ TRAINING (GPU 1, parallel to task396b GPU 0)

---

## Issue #108 Gate 1 spec (R17 强制 4 Gate 顺序)

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE product path): ⏳ TRAINING
- **目标**: L0/L1/L2 util ≥ 90% @ 35 epoch 截断
- **修复路径**: 跟 task400 一样 anchored residual product, 但训练 epoch ≤ 35 (避免 task400 ep40-49 退化)
- **预期**: 复现 task400 epoch 35 L0/L1/L2 = 100/100/98.4% → 验证这是真稳定点
- **实施**: scripts/task401_issue108_best_epoch_verify.py (跟 task400 同 patch, num_epochs=35 + early_stop=30)
- **GPU**: 1 (R7 空闲)

### Gate 2/3/4: ⏸ STOP per Issue #108 spec
- **原因**: Issue #108 spec 仅 Gate 1 ("best-epoch 窗口复验"), Gate 2/3/4 不在本 issue 范围

---

## 训练配置

- dataset: Instruments (Musical_Instruments)
- architecture: FreeCurvVectorQuantization M=3 block_dims=[11,11,10] e_dim=32
- fix: anchored residual product (跟 task400 同)
- num_epochs: 35 + early_stop=30 (若 epoch 30 util 已 ≥ 0.95 提前停)
- batch_size: 256, lr: 1e-4
- seed: 42
- GPU: 1 (CUDA_VISIBLE_DEVICES=1)
- ckpt path: products/task401_issue108_best_epoch_verify/ckpt/

---

## 关键产物

- **ckpt**: products/task401_issue108_best_epoch_verify/ckpt/issue108_ckpt.pt (R12 强制)
- **evidence**: products/task401_issue108_best_epoch_verify/evidence.json
- **log**: products/task401_issue108_best_epoch_verify/nohup.out
- **verdict**: verdicts/task401_issue108_best_epoch_verify_v2.md

---

## 后续 (per R22 + R19)

1. **task398 Stage 4 R@K eval** — 等 task396b Stage 3 训练完成后 launch (GPU 2 或 3, 因 GPU 1 在跑 task401)
2. **Issue #108 闭环** — 写 verdict + close issue
3. **跨方向联立**:
   - 若 task401 ep35 真 PASS (L1/L2 ≥ 90%) → 训练时长 (≤ 35 epoch) 是 product path 的真杠杆
   - 若 task401 ep35 仍 FAIL → 确认 product path 整体 NO-GO, 后续只走 HG-Rec baseline