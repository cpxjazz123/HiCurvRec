# Issue #36 v3e+HAB+DDP 4 卡 — interim ep75/ep80 饱和

**日期**: 2026-08-09  
**训练 PID**: 3569049 / 3569051 (torchrun)  
**Status**: ep60-80 区间 valid 饱和 (0.1152-0.1157), ckpt 仍 ep60 版本, ES 触发还需 ~16 epoch

## 训练曲线 (ep60-82)

| epoch | R@10 | Δ vs 峰 |
|-------|------|---------|
| 60 | **0.1157** ★ | (峰) |
| 65 | 0.1152 | -0.0005 |
| 75 | 0.1157 | = 峰 |
| 80 | 0.1153 | -0.0004 |
| 82 | — | loss 2.96 仍微降 |

**DDP 路径在 ep60-80 区间饱和**: valid 波动 ±0.0005, 不再单调上升. loss 2.95-2.96 稳定.

## ES 状态
- 最新峰: ep60=0.1157
- 不破峰计数: 估 4-6 epoch (ep65/70/75/80)
- ES 触发还需 ~14-16 epoch (ep96-100+)
- 当前 ckpt (HG_Rec_best.pth) = ep60 版本

## 决策
让训练继续 ep100+ 等 ES 自然触发 (~7 min). ES 触发后立即跑 Stage4 eval 验证 test_R@10 ≥ 0.1031 v4 SOTA.

## 估计
- 训练峰值: 0.1157-0.1170 (DDP 路径上限)
- 估计最终 valid_R@10: 0.1157 (ep60 ckpt)
- 估计 test_R@10: 0.105-0.120 (DDP 大 batch 估降 ratio 1.231 → 1.05-1.10)

## Gate 状态
- Gate 1 ✅ Setup
- Gate 2 ✅ 训练健康 (loss 2.95-2.96 稳定)
- Gate 3 ✅ valid (ep60=0.1157 > 0.1083)
- Gate 4 ⏳ 待 ES 触发 + Stage4 eval
