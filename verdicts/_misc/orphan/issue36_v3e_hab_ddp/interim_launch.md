# Issue #36 v3e+HAB+DDP 4 卡 (v3e 单卡 LR + DDP 大 batch regularization)

**日期**: 2026-08-09  
**训练 PID**: 3569049 (launcher) / 3569051 (torchrun) / 4×rank (PIDs 3569097-3569100)  
**Status**: 🚀 DDP 4 卡训练中 (ep1 完成 loss=5.7948, 23s/epoch)

## 任务目标
用户授权 framework 改动 (R31 解除). v3e+HAB 单卡 test_R@10=0.0982 (-0.0049 vs v4 SOTA 0.1031).
Issue #35 观察: HAB 帮助 test +0.0027 (vs plain 0.0955), 但 valid/test ratio 仅 1.237→1.231 (微降).
**Issue #36 改进**: v3e 配置 + HAB 三改动 + DDP 4 卡 (全局 batch=1024) — 验证 "DDP 大 batch regularization 是否帮助泛化".

## 关键配置 (R30 硬编码)
- BATCH_SIZE=256 (per-rank, 全局 1024)
- LR=1e-4 (单卡值, **不缩放**, 与 v77 DDP 4e-4 路径不同)
- NUM_EPOCHS=200, ES=20, EVAL_INTERVAL=5
- NUM_WORKERS=0 (R30 硬编码)
- HAB 三改动: hyperbolic_attn_bias=True, enable_residual_hab=True, λ_max=0.20, residual_alpha_init=-20.0
- Stage2 ckpt: taskA_stage2_issue61 (final_cs=[0.7792]*3)
- stage3_dropout=0.20

## 启动时间线
- 07:00: 创建 product_dir + _TRAINING_PID
- 07:02: 第一次 launch → torchrun FileNotFoundError (PATH 不含 genrec_env/bin) → 失败
- 07:02: 修正 wrapper 加 TORCHRUN 绝对路径 → 第二次 launch → "unrecognized arguments: --batch_size" → 失败
- 07:02: 移除 wrapper 重复传参 (BATCH_SIZE/LR/NUM_EPOCHS/ES/NUM_WORKERS 在 v3e 脚本 R30 硬编码) → 第三次 launch ✅
- 07:03: epoch 1 完成 loss=5.7948, 23s/epoch

## 预期 (R26)
- valid_R@10 估 0.118-0.125 (DDP 大 batch + 单卡 LR 应略低于 v3e 单卡 0.1181, 但更稳定)
- test_R@10 估 0.105-0.115 (DDP 大 batch regularization 估显著降 ratio 至 1.05-1.15)
- 训练时长: 200 epoch × 23s = ~80 min

## Gate 1 (Setup) ⏳ 进行中
## Gate 2 (训练健康) ⏳ 进行中
## Gate 3 (valid) ⏳ TBD
## Gate 4 (test) ⏳ TBD
