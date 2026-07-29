# Task #278 — Batch Stage 4 R@10 eval for untested ckpts

## 背景

仓库有 78 个 `HG_Rec_best.pth` ckpts (R12 强制保存), 但只有 8 个有 test 评估 (R@5/10/20 + NDCG):
- task84 (baseline)
- task85 (FDSA)
- task149 (heterokappa)
- task200 (dual_v5)
- task209 (A3)
- task211 (C1)
- task233 (dual_v5 RERUN)
- task237 (Arm B Sinkhorn=10)
- task243 (epoch 200 + 400)

**未测试的 ckpt 候选 (跟 issue 10 Gate 1 闭环相关)**:
| Task | ckpt | code_path | Stage 3 训练时间 | 状态 |
|------|------|-----------|-----------------|------|
| #144 arm_A | κ-decouple Arm A | `_t5_hrqvae_kappa_decouple_arm_A.npy` | 2026-07-24 | ❌ NO-GO Gate 1 (Issue #11) |
| #144 arm_B | κ-decouple Arm B | `_t5_hrqvae_kappa_decouple_arm_B.npy` | 2026-07-24 | ❌ NO-GO Gate 1 (Issue #11) |
| #156 | HG-Rec variant | `_t5_rqvae_code_default.npy` | 2026-07-24 | ? |
| #160 | HG-Rec variant | `_t5_rqvae_code_default.npy` | 2026-07-24 | ? |
| #161 | HG-Rec variant | `_t5_rqvae_code_default.npy` | 2026-07-24 | ? |
| #185 | T5-small resume | (待查) | 2026-07-25 | ? |
| #194 K=32 | T5-mini K=32 | `_t5_rqvae_k032.npy` | 2026-07-25 | ? |
| #194 K=64 | T5-mini K=64 | `_t5_rqvae_k064.npy` | 2026-07-25 | ? |
| #194 K=128 | T5-mini K=128 | `_t5_rqvae_k0128.npy` | 2026-07-25 | ? |
| #194 K=256 | T5-mini K=256 | `_t5_rqvae_k0256.npy` | 2026-07-25 | ? |
| #206 hyp_e19 | T5-small hyp e19 | `_t5_rqvae_hyp_e19.npy` | 2026-07-26 | ? |
| #206 euc_2000ep | T5-small euc 2000ep | `_t5_rqvae_euclidean_2000ep.npy` | 2026-07-26 | ? |
| #218 pck_spread | HG-Rec pck_spread | `_t5_hrqvae_pck_spread.npy` | 2026-07-27 | ? |
| #89 curv_free_M1 | HG-Rec curv free M=1 | `_t5_hrqvae_curv_free_M1.npy` | 2026-07-24 | ? |

## 任务范围

1. **批量 Stage 4 eval**: 写通用 driver `scripts/task278_batch_stage4_eval.py` (跟 task276_stage4_eval.py v6 / task243_stage4_eval.sh v2 兼容), 接受 ckpt_path + code_path 作为参数
2. **批量执行**: 对每个未测试 ckpt 跑一次 test eval (~2 min/ckpt, ~28 min 总)
3. **写 verdict**: 综合所有未测试 ckpt 的 test 数字, 找出超越 HG-Rec baseline 0.1020 的候选

## 关键决策点 (R11.5 + 用户 override)

- **自主执行**: loop.md §16 等用户决策, 但用户 override "不允许等用户拍板, 必须自行决定" + R10 主动推进 → 直接 launch
- **不加新架构**: 纯 Stage 4 eval, 不训练任何模型. 不消耗 GPU 训练时间
- **可中断**: 任何 ckpt 跑完即记录 metrics, 单 ckpt 失败不影响整体
- **Stage 4 driver**: 复用 task243_stage4_eval.sh v2 修过的 exclude-start-token 逻辑

## 物理产物

```
descriptions/task278_batch_stage4_eval.md  (本文件)
scripts/task278_batch_stage4_eval.py  (通用 driver)
scripts/task278_batch_stage4_eval.sh  (launcher, 遍历 ckpts)
verdicts/task278_*_test_metrics.json  (每个 ckpt 1 个)
verdicts/task278_batch_stage4_eval_result.md  (综合 verdict)
logs/task278/stage4_eval_*.log
```

result: Task #278 — 批量 Stage 4 R@10 eval for 14 个未测试 ckpts. 目的: 找出哪些变体在 test set 上超越 HG-Rec baseline 0.1020. GPU 总耗时 ~28 min. 复用 task243 v2 exclude-start-token 修过的 evaluate() 逻辑. 不训练任何新模型, 纯 eval.