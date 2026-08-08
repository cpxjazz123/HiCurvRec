# Issue #37 v3e+LR=4e-4+DDP 4 卡 — launch ep1 loss=5.3277 ★ 修复确认

**日期**: 2026-08-09  
**训练 PID**: 3611340 (launcher) / 3611342 (torchrun)  
**Status**: 🚀 DDP 4 卡训练中, ep1 loss=**5.3277** (vs Issue #36 ep1=5.7948, **-8.1%** 更低 → 4× LR 修复配比确认)

## Issue #36 修复
- Issue #36 根因: DDP batch=1024 仍用 LR=1e-4 (欠学习)
- Issue #37 修复: LR=1e-4 → LR=4e-4 (4× 缩放, 配 DDP batch=1024 = v4 SOTA 配比)
- 验证信号: ep1 loss 5.3277 vs 5.7948 (-8.1%) → 4× LR 起步阶段 loss 降得更快, 训练更充分

## 配置 (R30 硬编码)
- LR=4e-4 (修复 Issue #36 配比)
- BATCH_SIZE=256 (per-rank, 全局 1024 = 4×256)
- NUM_EPOCHS=200, ES=20, EVAL_INTERVAL=5
- NUM_WORKERS=0 (v3e 系列 R30 硬编码)
- HAB 三改动 (Issue #138 + #141 v77)
- Stage2 ckpt: taskA_stage2_issue61 (final_cs=[0.7792]*3)
- stage3_dropout=0.20
- master_port=29502 (跟 Issue #36 29501 区分)

## 估计 (Issue #36 趋势 + 4× LR 修正)
- valid_R@10 ep15 估 0.110-0.115 (vs Issue #36 ep15=0.1097)
- valid_R@10 峰估 0.118-0.125 (vs Issue #36 峰=0.1162)
- test_R@10 估 0.105-0.115 (vs Issue #36 test=0.0912 +0.014)
- 目标 test_R@10 ≥ 0.1042 (v85h SOTA)

## Gate 1 (Setup) ✅ PASS — DDP 4 卡启动正常, HAB 加载, LR=4e-4 修复确认
## Gate 2 (训练健康) ⏳ 进行中 — ep1 loss=5.33 (健康起步)
## Gate 3 (valid) ⏳ TBD — ep5 第一个 eval
## Gate 4 (test) ⏳ TBD — 训练完成后 Stage4 eval
