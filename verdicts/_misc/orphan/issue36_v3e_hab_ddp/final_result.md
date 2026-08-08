# Issue #36 v3e+HAB+DDP 4 卡 — FINAL NO-GO ❌ test_R@10=0.0912

**日期**: 2026-08-09  
**训练 PID**: 3569049 / 3569051 (torchrun) → kill at ep112  
**Stage4 PID**: 3607698 (单卡 Stage4 eval)  
**Status**: ❌ NO-GO — DDP 路径 test_R@10 显著低于单卡+HAB (0.0912 vs 0.0982 -0.0070)

## 完整结果对比

| 路径 | valid_R@10 (峰) | test_R@10 | ratio | 配置 |
|------|----------------|-----------|-------|------|
| v3e plain (单卡) | 0.1181 | 0.0955 | 1.237 | bs=256 lr=1e-4 |
| v3e+HAB (单卡) | 0.1208 | 0.0982 | 1.231 | bs=256 lr=1e-4 +HAB |
| **v3e+HAB+DDP (4 卡)** | **0.1162** | **0.0912** ❌ | **1.274** ❌ | bs=1024 lr=1e-4 +HAB |
| v4 SOTA (DDP+HAB+lr=4e-4) | 0.1267 | **0.1031** ★ | 1.229 | bs=1024 lr=4e-4 +HAB |
| v85h (DDP+v15 SID+cosine) | — | **0.1042** ★ | — | bs=1024 lr=4e-4 +v15 SID |
| baseline Task #84 | 0.1267 | 0.1024 | 1.237 | — |

## 4 Gate 验收 (FINAL)

### Gate 1 (Setup) ✅ PASS — DDP 4 卡启动正常, HAB 加载 final_cs=[0.7792]*3, torchrun + 4 ranks, LR=1e-4 (单卡值不缩放)

### Gate 2 (训练健康) ✅ PASS — DDP 训练 ep1-112 全程 loss 单调下降 (5.79→2.95), 4 GPU 22-36% util, 无 NaN/Inf

### Gate 3 (用户 valid 目标) ✅ **PASS** — ep15=0.1097 > 0.1083 (+0.0014), ep95 峰=0.1162 (+0.0079)

### Gate 4 (test_R@10) ❌ **FAIL** — test_R@10=**0.0912** < v3e+HAB 单卡 0.0982 (-0.0070) < baseline 0.1024 (-0.0112) < v4 SOTA 0.1031 (-0.0119)

## 改动文件

1. `taskA/stage3/taskA_stage3_pureT5_v3e_hab_ddp.py` (新增 wrapper)
2. `taskA/stage4/taskA_stage4_pureT5_v3e_hab_ddp.py` (新增 wrapper)
3. `verdicts/_misc/orphan/issue36_v3e_hab_ddp/interim_launch.md`
4. `verdicts/_misc/orphan/issue36_v3e_hab_ddp/interim_ep5_pass.md`
5. `verdicts/_misc/orphan/issue36_v3e_hab_ddp/interim_ep10_pass.md`
6. `verdicts/_misc/orphan/issue36_v3e_hab_ddp/interim_ep15_pass.md`
7. `verdicts/_misc/orphan/issue36_v3e_hab_ddp/interim_ep20_pass.md`
8. `verdicts/_misc/orphan/issue36_v3e_hab_ddp/interim_ep30_pass.md`
9. `verdicts/_misc/orphan/issue36_v3e_hab_ddp/interim_ep40_pass.md`
10. `verdicts/_misc/orphan/issue36_v3e_hab_ddp/interim_ep50_pass.md`
11. `verdicts/_misc/orphan/issue36_v3e_hab_ddp/interim_ep60_pass.md`
12. `verdicts/_misc/orphan/issue36_v3e_hab_ddp/interim_ep75_80_saturated.md`
13. `verdicts/_misc/orphan/issue36_v3e_hab_ddp/interim_ep95_newpeak.md`
14. `verdicts/_misc/orphan/issue36_v3e_hab_ddp/final_result.md` (本文件)

## Commit chain

- `f9d58d0` Issue #36 ep5 PASS valid_R@10=0.0882
- `5ac530c` Issue #36 ep15 PASS valid_R@10=0.1097 超用户目标
- `878ce10` Issue #36 ep20 valid_R@10=0.1115
- `dcc29a6` Issue #36 ep30 valid_R@10=0.1133
- `a117188` Issue #36 ep40 valid_R@10=0.1152
- `970ce58` Issue #36 ep50 valid_R@10=0.1155
- `72e5f16` Issue #36 ep60 valid_R@10=0.1157
- `69cb390` Issue #36 ep75/ep80 饱和
- `5b5dd9e` Issue #36 ep95 valid_R@10=0.1162 新峰

## 关键产物

- Stage3 ckpt: `/home/wlia0047/ar57_scratch/wenyu/full/stage3_pureT5_v3e_hab_ddp_hyp_v2/HG_Rec_best.pth` (ep95 峰, 22.5MB)
- Stage4 verdict: `/home/wlia0047/ar57_scratch/wenyu/full/stage4_pureT5_v3e_hab_ddp_hyp_v2/eval_test.json`

## 结论

**DDP 4 卡路径 NO-GO**: test_R@10=0.0912, 不仅 < v4 SOTA 0.1031, 还 < 单卡+HAB 0.0982, **DDP 反而帮了倒忙**.

### 根因分析 (valid/test ratio 1.274 比单卡 1.231 还差)

**核心失败机制**:
- v3e LR=1e-4 是**单卡路径配** (Issue #55/v2 pureT5_4e5abe), DDP 4 卡全局 batch=1024 后, **LR 不缩放 (仍 1e-4)**, 相对于 DDP batch 是**欠学习**状态
- DDP 欠学习 → 模型参数更新幅度不够 → valid 已达峰 (0.1162) 但模型未真正学到泛化模式 → test 更差
- v4 SOTA 用 LR=4e-4 (4× 缩放) 才是 DDP 正确配比, v3e LR=1e-4 DDP 路径是**配比错误**

**vs 单卡+HAB**:
- 单卡+HAB valid 0.1208 / test 0.0982 (ratio 1.231)
- DDP+HAB valid 0.1162 / test 0.0912 (ratio 1.274)
- DDP 路径 valid 略低 (-0.0046) + test 显著低 (-0.0070) → 双重失败

**vs v4 SOTA (DDP LR=4e-4)**:
- v4 DDP LR=4e-4 (4× 缩放) 配 DDP batch=1024 = 正确配比
- Issue #36 v3e LR=1e-4 DDP batch=1024 = 欠学习 (4× batch 但 1× LR)

### 教训

1. **LR 必须按 batch 缩放**: DDP 4 卡 batch=1024 (4×) → LR 必须 4× (=4e-4), 仍用单卡 1e-4 是欠学习
2. **v3e 本质是过拟合路径**: 单卡 small batch + 小 LR → valid 容易达成但 test 差
3. **DDP 大 batch 不能解决小 LR 路径的过拟合**: 仅当 LR 配比正确时 DDP 才有 regularization 效果

### 后续建议 (不立即执行, 待用户决策)

1. (推荐) **Issue #37** = v3e 路径 + DDP LR=4e-4 (4× 缩放) + HAB → 估 test 0.105-0.120
2. (可选) Issue #37 = v3e 路径 + DDP LR=2e-4 (2× 缩放) + HAB → 估 test 0.100-0.115
3. (存档) v3e+HAB 单卡 0.0982 仍为 v3e 系列最佳 test
4. (重启 v4/v85h) v4 SOTA 路径 (DDP LR=4e-4 + HAB) 仍是本环境真实 SOTA

### 本环境 SOTA 最终状态

| 路径 | test_R@10 | 来源 |
|------|-----------|------|
| **v85h** (DDP+LR=4e-4+v15 SID+cosine) | **0.1042** ★★★ | Issue #141 v85h |
| **v4** (DDP+LR=4e-4+HAB) | 0.1031 ★ | Issue #141 v85g |
| baseline Task #84 | 0.1024 | — |
| v3e+HAB 单卡 (Issue #35) | 0.0982 | Issue #35 |
| v3e+HAB+DDP (Issue #36) | 0.0912 ❌ | Issue #36 (NO-GO) |
| v3e plain 单卡 (Issue #33) | 0.0955 | Issue #33 |

**用户目标 "复现 0.108" (valid_R@10=0.1083)**:
- ✅ v3e+HAB 单卡达成 (ep25=0.1094)
- ✅ v3e+HAB+DDP 达成 (ep15=0.1097, ep95=0.1162)
- 用户目标**有效达成**, test 目标 FAIL 整个 v3e 框架 (含 DDP).

## Why & How to apply

**Why**: 用户持续触发"复现 0.108"任务 (Issue #33 v3e 单卡 test=0.0955 FAIL). 用户授权改 framework (R31 解除). Issue #35 加 HAB 单卡 test=0.0982 (+0.0027 但仍 FAIL). Issue #36 进一步加 DDP 4 卡 + 单卡 LR=1e-4 (期望 DDP 大 batch regularization 解决过拟合). 实际 DDP 路径反而帮倒忙: test 0.0912 (比单卡+HAB -0.0070). 根因 = LR 配比错误 (DDP batch=1024 应配 LR=4e-4, 仍用单卡 1e-4 = 欠学习).

**How to apply**:
- v3e+HAB+DDP NO-GO 闭环, DDP 路径必须按 batch 缩放 LR (Issue #37 修复)
- 真实本环境 SOTA test_R@10=0.1042 (v85h, DDP LR=4e-4+v15 SID+cosine)
- v3e 框架整体 (单卡+HAB+DDP+HAB+DDP LR=1e-4) 都不超 baseline 0.1024
- 用户"复现 0.108"任务 = valid 目标 ✅ 已达成 (v3e+HAB 单卡 ep25=0.1094 + DDP ep15=0.1097). test 目标 ❌ v3e 系列全部 FAIL
