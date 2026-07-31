# Task #397 / Issue #101 [方向B Gate1] per-component scale 修复 + Stage 1 实证

**日期**: 2026-07-31
**触发**: task391 #98 audit FAIL (per-component scale bias) + Issue #101 [方向B Gate1] 修复product聚合尺度并复验训练后agree3
**前置**:
- task391 #98 audit FAIL: synthetic agree3=0% (per-comp 完全不重合), init agree3=4.20% (<95% PASS 阈值)
- 根因诊断: `FreeCurvVectorQuantization._per_component_dist_sq` (line 346-389) 没 per-component std normalization + aggregate-sum argmin bias toward component 0
- task394 #97 patch 已修 Stage 1 距离公式, 但 product path 的 scale 仍未修

**任务**: 实施 per-component std normalization + Stage 1 50 epoch 实证复跑 (per Issue #101 spec "复验训练后agree3")
**结果**: ⏳ TRAINING (GPU 1, parallel to task396 GPU 0)

---

## Issue #101 Gate 1 spec (R17 强制 4 Gate 顺序)

### Gate 1 (= Stage 1 RQ-VAE/HRQVAE product path): ⏳ TRAINING
- **目标**: agree3 ≥ 95% (per-component 一致性)
- **根因**: per-component scale 没归一化 → aggregate-sum argmin 单 component 0 → per-component 选错
- **修复**:
  1. `FreeCurvVectorQuantization._per_component_dist_sq`: 加 per-component std normalization
  2. `init_emb`: per-component kmeans init (替换 full-space kmeans)
- **实施**: scripts/task397_issue101_per_component_stage1_train.py
- **GPU**: 1 (R7 空闲, parallel to task396 GPU 0)

### Gate 2/3/4: ⏸ STOP per Issue #101 spec
- **原因**: Issue #101 spec 仅 Gate 1 ("复验训练后agree3"), Gate 2/3/4 不在本 issue 范围
- **后续**: Gate 2 已由 Issue #102 [方向C Gate2] 处理 (task395 PASS), Gate 3 已由 task396b 启动 (PID 660598), Gate 4 由 task398 下一轮跑

---

## 训练配置

- dataset: Instruments (Musical_Instruments)
- architecture: FreeCurvVectorQuantization (M=3, block_dims=[11,11,10], e_dim=32)
- fix: per-component std normalization (in _per_component_dist_sq)
- num_epochs: 50
- batch_size: 256, lr: 1e-4
- seed: 42
- GPU: 1 (CUDA_VISIBLE_DEVICES=1)
- ckpt path: products/task397_issue101_per_component_scale_stage1/ckpt/

---

## 关键产物

- **ckpt**: products/task397_issue101_per_component_scale_stage1/ckpt/issue101_ckpt.pt (R12 强制, 训练完落盘)
- **evidence**: products/task397_issue101_per_component_scale_stage1/evidence.json (agree3 metrics)
- **log**: logs/task397_issue101_*.log
- **verdict**: verdicts/task397_issue101_per_component_scale_stage1_v2.md (训练完成后)

---

## 后续 (per R22 + R19)

1. **task398: Stage 4 R@K eval** — 用 task396 ckpt + Instruments test set, 验证 R@10 > 0.1020 (HG-Rec baseline threshold)
2. **Issue #101 实证结果** — 跟 task391 audit 联立: 若 agree3 ≥ 95% → per-comp scale fix 修复 product 聚合; 若 agree3 < 95% → 仍需探索其他 product 聚合方法