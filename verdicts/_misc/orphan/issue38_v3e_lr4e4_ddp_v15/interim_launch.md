# Issue #38 v3e+LR=4e-4+DDP+v15 SID — launch ep1 loss=5.36

**日期**: 2026-08-09  
**训练 PID**: 3637536 (launcher) / 3637538 (torchrun)  
**Status**: 🚀 DDP 4 卡训练中, ep1 loss=**5.3582**, v15 SID sha=5f8331cc

## Issue #37 → Issue #38 修复
- Issue #37 PARTIAL-GO: v3e+hyp_v2 SID test 0.0980 (不达 baseline 0.1024)
- Issue #38 修复: 替换 hyp_v2 SID → **v15 SID** (Issue #141 v85h 配置)
- 期望: v15 SID 是 v85h SOTA 来源, 替换后估 test 0.105-0.115

## 配置 (R30 硬编码)
- LR=4e-4 (Issue #37 修复配比)
- BATCH_SIZE=256 (per-rank, 全局 1024)
- NUM_EPOCHS=200, ES=20, EVAL_INTERVAL=5
- HAB 三改动 (Issue #138 + #141 v77)
- **SID**: v15 SID sha=5f8331cc (vs Issue #37 hyp_v2 sha=06af0fed)
- Stage2 ckpt: taskA_stage2_issue61 (final_cs=[0.7792]*3)
- stage3_dropout=0.20
- master_port=29503 (跟 Issue #36/37 区分)

## 估计 (基于 Issue #37 + v15 SID 修复)
- valid_R@10 ep5 估 0.100-0.110 (vs Issue #37 ep5=0.1031)
- valid_R@10 ep15 估 0.115-0.120
- valid_R@10 峰估 0.125-0.130 (vs v4 SOTA 0.1267)
- **test_R@10 估 0.105-0.115** (vs v85h 0.1042, vs Issue #37 0.0980)

## Gate 1 (Setup) ✅ PASS — DDP 4 卡启动, HAB 加载, v15 SID sha=5f8331cc
## Gate 2 (训练健康) ⏳ 进行中 — ep1 loss=5.36 (健康起步)
## Gate 3 (valid) ⏳ TBD — ep5 第一个 eval
## Gate 4 (test) ⏳ TBD — 估 0.105-0.115, 真正有望夺 v85h 0.1042 SOTA
