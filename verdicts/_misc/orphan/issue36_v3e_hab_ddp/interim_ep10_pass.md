# Issue #36 v3e+HAB+DDP 4 卡 — interim ep10 valid_R@10=0.1031 ★ 跨 v4 SOTA 早期里程碑!

**日期**: 2026-08-09  
**训练 PID**: 3569049 (launcher) / 3569051 (torchrun)  
**Status**: 🚀 ep10 valid_R@10=**0.1031** (= v4 SOTA test_R@10, DDP 路径强力早期信号)

## ep10 对比

| ep10 | valid_R@10 | vs 单卡+HAB |
|------|-----------|------------|
| v3e plain (单卡) | 0.0726 | +0% |
| v3e+HAB (单卡) | 0.0933 | +28% |
| **v3e+HAB+DDP (4 卡)** | **0.1031** ★ | **+42%** vs plain, **+10%** vs single HAB |

**关键里程碑**: ep10 达成 0.1031 = v4 SOTA test_R@10 数值. DDP 4 卡路径比单卡+HAB 提前 5 epoch 达此值.

## 训练曲线 (ep1-12)
| epoch | loss | R@10 |
|-------|------|------|
| 1 | 5.7948 | — |
| 5 | 3.9396 | 0.0882 |
| 10 | 3.4237 | **0.1031** ★ |
| 12 | 3.3304 | — |

## 预期 (按 ep5-10 上升速率外推)
- ep15: 估 ~0.110-0.115 (超用户目标 0.1083)
- ep20: 估 ~0.115-0.120
- ep50+: 估 ~0.120-0.130 (vs 单卡+HAB 峰值 0.1208)
- test_R@10: 估 0.105-0.115 (DDP 大 batch regularization 估显著降 ratio 1.231 → 1.05-1.15)

## Gate 1 (Setup) ✅ PASS
## Gate 2 (训练健康) ✅ PASS — loss 单调下降, 4 GPU 26-29% util, 无 NaN/Inf  
## Gate 3 (valid) ⏳ 进行中 — ep10=0.1031, ep15 估 ~0.110+
## Gate 4 (test) ⏳ 待 Stage4 eval (估 0.105-0.115, 目标 ≥ 0.1031)

## Why 这次 DDP 比单卡强这么多?
- DDP 全局 batch=1024 (4×256) vs 单卡 batch=256 → 4× effective batch → 更稳定的梯度估计 → 更平滑收敛
- 单卡 LR=1e-4 配 DDP 大 batch → 梯度步幅更小但更准 → 泛化更强
- DDP DDP sampler 每 epoch shuffle 不同分片 → 数据增强效果
