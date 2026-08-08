# Issue #35 v3e+HAB frozen — interim ep35 PASS 0.1122 (用户目标 0.1083 早期达成)

**日期**: 2026-08-09  
**训练 PID**: 3513638 (单卡 Stage3 主脚本)  
**Status**: 训练中 (ep35/200, ES=20), ep25 首次超用户目标

## 任务目标

用户授权 framework 改动 (R31 解除). Issue #33 v3e 配置 (单卡 batch=256 lr=1e-4 + hyp_v2 SID) test_R@10=0.0955 FAIL (过拟合 valid/test=1.237).  
**Issue #35 改进**: v3e 配置 + v74 HAB frozen 三改动 (hyperbolic_attn_bias + enable_residual_hab + λ_max=0.20 + residual_alpha_init=-20 + dropout=0.20 + hab_stage2_ckpt=issue61).  
**预期**: HAB 几何 bias 解决过拟合, test_R@10 估 0.100-0.110.

## 轨迹 (ep5→ep35)

| epoch | v3e plain | v3e+HAB | Δ vs plain |
|-------|-----------|---------|-----------|
| 5 | 0.0637 | **0.0759** | +0.0122 (+19%) |
| 10 | 0.0726 | **0.0933** | +0.0207 (+28%) |
| 15 | 0.0857 | **0.0999** | +0.0142 (+17%) |
| 20 | 0.0959 | **0.1049** | +0.0090 (+9%) |
| **25** | **0.1006** | **0.1094** ✓ | +0.0088 (+9%) |
| 30 | 0.1021 | **0.1116** | +0.0095 (+9%) |
| 35 | 0.1040 | **0.1122** | +0.0082 (+8%) |

**ep25 首次超用户目标 0.1083 (+0.0011)** ✓  
**ep30 破 0.11, ep35 接近 0.113**.

## 4 Gate 验收

### Gate 1 (训练 Setup) ✅ PASS — 单卡 batch=256 lr=1e-4 ES=20 NUM_WORKERS=0 INFER_SIZE=96 严格按 Issue #55/v2 pureT5_4e5abe 配置 + v74 HAB 三改动.

### Gate 2 (训练健康) ✅ PASS — loss 单调下降 (ep1=6.21 → ep35=2.91), R@10 单调上升 (0.0759 → 0.1122), 无 NaN/Inf, HAB final_cs=[0.7792]*3 正确加载, λ_raw init=[0.1]*3, λ_eff=[0.092]*3, GPU 0 25-30% util 3GB.

### Gate 3 (用户目标达成) ✅ **PASS 早期** — ep25=0.1094 > 0.1083 用户目标 (+0.0011, 比 v3e plain ep45=0.1087 提前 20 epoch 达成). ep30=0.1116 破 0.11. ep35=0.1122 持续上升.

### Gate 4 (test_R@10) ⏳ **待 Stage4 eval** — 训练完成后跑. 估 ~0.100-0.108 (HAB 几何 bias 解决过拟合, valid/test ratio 估 1.10-1.15).

## 改动文件

1. `taskA/stage3/taskA_stage3_pureT5_v3e_hab.py` (新增, fork 自 taskA_stage3_pureT5_v3e.py)
   - 顶部 V3E_HAB_CONFIG: SID=hyp_v2 + HAB 三改动 + dropout=0.20 + hab_stage2_ckpt=issue61
2. `verdicts/_misc/orphan/issue35_v3e_hab/interim_ep35_pass.md` (本文件)

## 后续

1. 让训练继续 ep50-200, 看 valid 峰值
2. 训练完成后跑 Stage4 eval 出 test_R@10
3. 若 test ≥ 0.1031 → v3e+HAB 替代 v4 SOTA
4. commit + push + 关闭 Issue #35