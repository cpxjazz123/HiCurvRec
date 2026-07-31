# Task #400 / Issue #107 [方向B Gate1] anchored residual product 避免 component 主导坍缩

**日期**: 2026-07-31
**触发**: Issue #107 [方向B Gate1] anchored residual product 避免component主导坍缩
**前置**:
- task397 #101 per-comp std norm + mean agg NO-GO (L1/L2 util 14.84%/12.11%)
- task399 #106 staged unfreeze NO-GO (L1/L2 util 10.16%/5.47%)
- task391 #98 audit FAIL: aggregate-sum argmin bias toward component 0

**任务**: 实施 anchored residual product (L0 codebook 作为 anchor, L1/L2 在 L0 anchored residual 上做 product) + Stage 1 50 epoch 实证
**结果**: ⏳ TRAINING (GPU 1, parallel to task396b GPU 0)

---

## Issue #107 Gate 1 spec (R17 强制 4 Gate 顺序)

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE product path): ⏳ TRAINING
- **目标**: L1/L2 util ≥ 50% (vs task397 14.84%/12.11%, task399 10.16%/5.47%)
- **修复路径**: anchored residual product
  - L0 codebook 作为 anchor base (跟 task397 #101 一样 per-comp std norm + mean agg)
  - L1 codebook 基于 L0 selected centroid + residual variance 自适应分配 (避免 component 0 主导)
  - L2 codebook 基于 L1 selected centroid + residual variance 自适应分配
  - 关键: 残差编码按 layer 顺序级联, 每层 residual 独立 norm, 避免 component-0 在 residual space 主导
- **预期**: L0/L1/L2 util 都 ≥ 50% (vs task397/399 失败)
- **实施**: scripts/task400_issue107_anchored_residual.py
- **GPU**: 1 (R7 空闲, parallel to task396b GPU 0)

### Gate 2/3/4: ⏸ STOP per Issue #107 spec
- **原因**: Issue #107 spec 仅 Gate 1 ("anchored residual product 避免component主导坍缩"), Gate 2/3/4 不在本 issue 范围

---

## 训练配置

- dataset: Instruments (Musical_Instruments)
- architecture: FreeCurvVectorQuantization M=3 block_dims=[11,11,10] e_dim=32
- fix: anchored residual product (L0 anchor, L1/L2 on L0-anchored residual)
- num_epochs: 50
- batch_size: 256, lr: 1e-4
- seed: 42
- GPU: 1 (CUDA_VISIBLE_DEVICES=1)
- ckpt path: products/task400_issue107_anchored_residual/ckpt/

---

## 关键产物

- **ckpt**: products/task400_issue107_anchored_residual/ckpt/issue107_ckpt.pt (R12 强制)
- **evidence**: products/task400_issue107_anchored_residual/evidence.json
- **log**: products/task400_issue107_anchored_residual/nohup.out
- **verdict**: verdicts/task400_issue107_anchored_residual_v2.md

---

## 后续 (per R22 + R19)

1. **task398: Stage 4 R@K eval** — 等 task396b Stage 3 训练完成后 launch (GPU 1, 但 task400 在用 GPU 1, 调整: 等 task400 完释放 GPU 1, 再 launch Stage 4 — 或用 GPU 2/3)
2. **Issue #107 闭环** — 写 verdict + close issue
3. **跨方向联立**: task397 + task399 + task400 联立判定 product path 是否还有杠杆 (per R18 实证)
4. **3-way NO-GO 收口条件**: 若 task400 也 NO-GO, product_manifold 路径彻底锁死, 后续只走 HG-Rec baseline (#97 patch path)