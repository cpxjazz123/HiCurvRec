# Issue #36 v3e+HAB+DDP 4 卡 — interim ep60 valid_R@10=0.1157 新峰

**日期**: 2026-08-09  
**训练 PID**: 3569049 / 3569051 (torchrun)  
**Status**: ep60 valid_R@10=0.1157 (新峰), ep65=0.1152 略降 (-0.0005), 训练接近饱和

## 训练曲线 (ep50-65)

| epoch | R@10 | Δ vs prev |
|-------|------|-----------|
| 50 | 0.1155 | — |
| 60 | **0.1157** ★ | +0.0002 (新峰) |
| 65 | 0.1152 | -0.0005 (过拟合早期) |

**DDP 路径已接近饱和**: ep40→50 +0.0003, ep50→60 +0.0002, ep60→65 -0.0005.

## ES 状态
- 最新峰: ep60=0.1157
- ES=20: 需 20 epoch 连续不破峰
- 当前已 5 epoch 不破峰 (ep65<ep60), 距 ES 触发还差 15 epoch (ep80+)
- 训练继续 ep80-85 后 ES 触发

## 趋势外推 (DDP 路径 valid 上限)
- 峰值估 0.116-0.118 (DDP 比单卡+HAB 上限略低, 但 ratio 估优)
- vs 单卡+HAB 峰值 0.1208: DDP 估 < 0.1208 (DDP 大 batch 路径 valid 略低)
- 但 test_R@10 估 0.110-0.120 (DDP 大 batch 估显著降 ratio 1.231 → 1.05-1.10)

## Gate 状态
- Gate 1 (Setup) ✅ PASS
- Gate 2 (训练健康) ✅ PASS — loss 5.79→2.96, 4 GPU 25-34% util, 无 NaN
- Gate 3 (valid) ✅ PASS (ep60=0.1157 > 0.1083 +0.0074)
- Gate 4 (test) ⏳ 待 Stage4 eval

## 决策
让训练继续 ep80+ 等 ES 触发. 触发后跑 Stage4 eval 验证 test_R@10 是否 ≥ 0.1031 (v4 SOTA).
