# Issue #33 v3e pureT5 复现 valid_R@10=0.1083 — FINAL test_R@10=0.0955 (过拟合)

**日期**: 2026-08-09  
**Stage4 PID**: 3506420 (单卡 Stage4 eval)  
**Status**: 端到端完成, **valid 达成 / test FAIL**

## 最终端到端结果

| 维度 | v3e | v4 (我们 SOTA) | baseline Task #84 | pureT5_4e5abe 历史 (Issue #55/v2) |
|------|-----|---------------|------------------|---------------------------------|
| **valid_R@10** | **0.1181** ★ | 0.1267 (最佳 valid) | 0.1267 | 0.1197 (ep31) |
| **test_R@5** | 0.0749 | 0.0820 | — | — |
| **test_R@10** | **0.0955** ❌ | **0.1031** ★ | 0.1024 | 0.0956 |
| **test_R@20** | 0.1211 | 0.1314 | — | — |
| **test_NDCG@10** | 0.0691 | 0.0744 | — | — |
| **valid/test ratio** | **1.237** ❌ | 1.229 | — | 1.181 |
| 配置 | 单卡 batch=256 lr=1e-4 + hyp_v2 | DDP batch=1024 lr=4e-4 + hyp_v2 | 原 baseline | 单卡 batch=256 lr=1e-4 + v3e poincare SID (lost) |

## 用户目标达成情况

- **核心目标 valid_R@10=0.1083** ✅ **PASS 大幅** (0.1181, +0.0098, +9.0%)
- **隐含目标 test_R@10 ≥ 0.1024 baseline** ❌ FAIL (-0.0069)

## 4 Gate 验收 (FINAL)

### Gate 1 (训练 Setup) ✅ PASS — 同前

### Gate 2 (训练健康) ✅ PASS — 同前

### Gate 3 (用户目标达成 valid_R@10) ✅ **PASS 大幅** (0.1181 ep115 峰值)

### Gate 4 (test_R@10) ❌ **FAIL** — test_R@10=0.0955 < baseline 0.1024 (-0.0069), < v4 SOTA 0.1031 (-0.0076)

**根因分析** (过拟合):
- valid/test ratio = 1.237 (vs v4 1.229, vs pureT5_4e5abe 历史 1.181)
- 单卡 batch=256 lr=1e-4 + cosine decay LR_min=1e-6 → 模型在 valid set 上特化
- hyp_v2 SID 本身 fine-grained 提升 valid 但泛化弱
- 缺乏 v4 路径的 DDP 4 卡 regularization + HAB 几何 bias 泛化增强

## 改动文件 (5 commits)

1. `common/stage3/stage3_train_pureT5_v3e.py` (新增 fork)
2. `taskA/stage3/taskA_stage3_pureT5_v3e.py` (新增 wrapper)
3. `taskA/stage4/taskA_stage4_pureT5_v3e.py` (新增 wrapper)
4. `verdicts/_misc/orphan/issue33_v3e_repro/interim_ep45_pass.md`
5. `verdicts/_misc/orphan/issue33_v3e_repro/interim_ep60_pass.md`
6. `verdicts/_misc/orphan/issue33_v3e_repro/interim_ep100_pass.md`
7. `verdicts/_misc/orphan/issue33_v3e_repro/final_result.md`
8. `verdicts/_misc/orphan/issue33_v3e_repro/final_test_eval.md` (本文件)

## Commit chain

- `47ce823` Issue #33 v3e pureT5 单卡重训 PASS ep50 valid_R@10=0.1095
- `9c6b62f` Issue #33 v3e interim ep60 valid_R@10=0.1124
- `e6f505e` Issue #33 v3e interim ep100 valid_R@10=0.1170 (peak ep95)
- `7a48560` Issue #33 v3e FINAL valid_R@10=0.1181 (peak ep115)

## 关键产物

- Stage3 ckpt: `/home/wlia0047/ar57_scratch/wenyu/full/stage3_pureT5_v3e_hyp_v2/HG_Rec_best.pth` (ep115 最佳)
- Stage4 verdict: `/home/wlia0047/ar57_scratch/wenyu/full/stage4_pureT5_v3e_hyp_v2/eval_test.json`

## 结论与后续

**v3e 不是替代 v4 的方案**. v3e valid 高但 test 低, 是单卡 small-batch + cosine LR 路径典型过拟合模式.

**本环境 SOTA 仍是 v4** (test_R@10=0.1031, Issue #141 v85g 路径), 曲率路线尚未超 0.11.

**建议下一步** (不立即执行, 待用户决策):
1. (推荐) v3e 配置 + HAB 泛化增强 → 估可同时提升 valid + test
2. (可选) 用 v3e 路径作 v85g/v85h 的对照, 验证 valid/test ratio 改善
3. (存档) v3e 配置 (单卡 batch=256 lr=1e-4 + hyp_v2) 作为纯 valid baseline 工具, 不用作 test submission

## Why & How to apply

**Why**: 用户持续触发"复现 0.108" — 实际 = valid_R@10=0.1083 from Issue #55/v2 pureT5_4e5abe ep10 2026-08-03 (ckpt 已丢失). 用户授权改 framework (R31 解除), 立即 fork Stage3 + Stage4 wrapper + 启动训练, ep45 首次超目标 0.1087, ep115 峰值 0.1181 (+9.0%). 但 Stage4 eval 显示 test_R@10=0.0955 FAIL (过拟合 valid/test ratio=1.237).

**How to apply**: 用户再说"复现 0.108"时:
- 核心 valid 目标已达成并大幅超越 (0.1181 vs 0.1083, +9.0%)
- test 目标 FAIL (0.0955 < 0.1024 baseline), 不可作为生产基线
- 真实本环境 SOTA test_R@10=0.1031 (我们 v4 ckpt), 是当前唯一超 baseline 路径
- v3e 配置作 valid-only baseline 工具, 不作 test submission