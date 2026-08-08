# Issue #33 v3e pureT5 复现 valid_R@10=0.1083 — interim ep60 PASS 0.1124

**日期**: 2026-08-09  
**训练 PID**: 3436130 (单卡 Stage3 主脚本)  
**Status**: 训练中 (ep60/200, ES=20), 持续上升趋势

## 完整轨迹 (ep5→ep60)

| epoch | valid_R@10 | ΔR@10 | train_loss | NDCG@20 |
|-------|-----------|-------|-----------|---------|
| 5 | 0.0637 | — | 4.6628 | 0.0360 |
| 10 | 0.0726 | +0.0089 | 3.8945 | 0.0532 |
| 15 | 0.0857 | +0.0131 | 3.6083 | 0.0685 |
| 20 | 0.0959 | +0.0102 | 3.4575 | 0.0755 |
| 25 | 0.1006 | +0.0047 | 3.3536 | 0.0792 |
| 30 | 0.1021 | +0.0015 | 3.2770 | 0.0807 |
| 35 | 0.1040 | +0.0019 | 3.2135 | 0.0820 |
| 40 | 0.1062 | +0.0022 | 3.1608 | 0.0834 |
| **45** | **0.1087** | +0.0025 | 3.1167 | 0.0850 |
| 50 | 0.1095 | +0.0008 | 3.0815 | 0.0859 |
| 55 | 0.1112 | +0.0017 | 3.0505 | 0.0869 |
| **60** | **0.1124** | +0.0012 | 3.0212 | 0.0882 |

## 任务达成

✅ **PASS** valid_R@10=0.1124 ≫ 0.1083 用户目标 (+0.0041)

## 4 Gate 验收

### Gate 1 (训练 Setup)
✅ PASS — 同前

### Gate 2 (训练健康)
✅ PASS — 同前 + loss 持续下降 (ep1=6.42 → ep60=3.02), R@10 持续上升 (0.0637 → 0.1124), 无任何异常信号

### Gate 3 (用户目标达成)
✅ **PASS at ep45 (首次)** → 持续上升至 ep60=0.1124
- ep45: 0.1087 > 0.1083 (+0.0004) ✓
- ep50: 0.1095 (+0.0012 vs ep45)
- ep55: 0.1112 (+0.0017) ✓ 突破 0.11
- ep60: 0.1124 (+0.0012) ✓ 突破 0.112

### Gate 4 (test_R@10)
⏳ **待 Stage4 eval** — 训练完成后跑. 估算基于 hyp_v2 路径 test_R@10=0.1031 + 单卡 small-batch regularization, 估 test_R@10 ≈0.090-0.095 (valid/test ratio ~1.18-1.20).

## 改动文件 (commit 47ce823)

1. `common/stage3/stage3_train_pureT5_v3e.py` (新增, fork 自 stage3_train_pure_t5.py)
   - 顶部 CONSTANTS: NUM_EPOCHS=200, EARLY_STOP=20, BATCH_SIZE=256, INFER_SIZE=96, LR=1e-4, NUM_WORKERS=0
2. `taskA/stage3/taskA_stage3_pureT5_v3e.py` (新增)
   - V3E_CONFIG 硬编码: SID=hyp_v2 (sha=06af0fed), 单卡 cuda:0, HAB=False
3. `verdicts/_misc/orphan/issue33_v3e_repro/interim_ep45_pass.md` (新增)

## 后续

1. 让训练继续 (ep70-200), 看能否突破 0.115 / 0.120
2. 训练完成后跑 Stage4 eval 出 test_R@10, 写最终 verdict
3. commit + push 最终 verdict + 关闭 Issue #33

## commit hash

`47ce823` (Issue #33 v3e pureT5 单卡重训 PASS ep50 valid_R@10=0.1095 ≥ 0.1083)