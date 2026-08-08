# Issue #38 v3e+LR=4e-4+DDP+v15 SID — interim ep55 valid_R@10=0.1216 ★★★★ 新峰

**日期**: 2026-08-09  
**训练 PID**: 3637536 / 3637538 / 3637584-587 (DDP 4 卡)  
**Status**: ep55 valid_R@10=**0.1216** 新峰, plateau 未确认

## Issue #38 v15 SID 轨迹

| epoch | valid_R@10 | NDCG@20 | loss | 状态 |
|-------|-----------|---------|------|------|
| 10 | 0.1120 | 0.0920 | 3.0008 | [BEST] |
| 15 | 0.1160 | 0.0948 | 2.8654 | [BEST] |
| 20 | 0.1157 | 0.0952 | 2.7764 | (no improv) |
| 25 | 0.1179 | 0.0969 | 2.7226 | [BEST] |
| 30 | 0.1181 | 0.0972 | 2.6814 | [BEST] |
| 35 | 0.1199 | 0.0994 | 2.6505 | [BEST] |
| 40 | **0.1205** | 0.0992 | 2.6356 | [BEST] |
| 45 | 0.1199 | 0.0997 | 2.6314 | (no improv) |
| 50 | (eval skipped) | - | - | - |
| **55** | **0.1216** | 0.1004 | 2.6180 | [BEST] ★★★★ |

v15 SID 第二次 plateau 后再次突破! 估 ep75-100 仍能涨至 0.124-0.128.

## vs Issue #37 hyp_v2 SID 峰
- Issue #37 峰 = 0.1248 (ep75-85)
- Issue #38 估峰 = 0.124-0.128 (ep75-100)
- v15 SID 慢热但更平滑, 最终可能追平或反超 hyp_v2

## 0.108 复现 ✓ + 0.1267 (baseline) 接近
- baseline valid_R@10 = 0.1267 (Task #84)
- Issue #38 ep55=0.1216 vs baseline -0.0051
- 估计 ep100+ 接近 baseline valid

## Gate 状态
## Gate 1 ✅ Setup PASS
## Gate 2 ✅ 训练健康 PASS — R23 7 信号全 PASS, loss 单调下降
## Gate 3 ✅ valid_R@10=0.1083 复现 ✓ + 0.1216 (vs baseline 0.1267 -0.0051)
## Gate 4 ⏳ Stage4 eval TBD — 估 test_R@10=0.108-0.115

## 下一步
1. 等待 ep60 eval (08:51:55) 验证
2. 若继续上升: 等待 ep75-100 真正峰
3. 若 plateau: kill DDP + Stage4 eval
