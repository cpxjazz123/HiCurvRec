# Issue #37 v3e+LR=4e-4+DDP 4 卡 — FINAL test_R@10=0.0980 PARTIAL-GO

**日期**: 2026-08-09  
**Stage4 PID**: 3635117 (单卡 Stage4 eval)  
**Status**: PARTIAL-GO — LR=4e-4 修复 Issue #36 配比错误, test +0.0068 (vs Issue #36), 但仍未达 baseline 0.1024 / v4 SOTA 0.1031 / v85h 0.1042

## 完整结果对比

| 路径 | valid (峰) | test_R@10 | ratio | Δ vs baseline 0.1024 |
|------|-----------|-----------|-------|---------------------|
| v3e plain (单卡) | 0.1181 | 0.0955 | 1.237 | -0.0069 |
| v3e+HAB (单卡) | 0.1208 | 0.0982 | 1.231 | -0.0042 |
| Issue #36 (LR=1e-4 DDP) | 0.1162 | 0.0912 | 1.274 | -0.0112 |
| **Issue #37 (LR=4e-4 DDP)** | **0.1248** | **0.0980** | **1.273** | **-0.0044** |
| v4 SOTA (DDP+LR=4e-4+HAB) | 0.1267 | **0.1031** ★ | 1.229 | +0.0007 |
| v85h SOTA (DDP+LR=4e-4+v15+cosine) | — | **0.1042** ★★★ | — | +0.0018 |
| baseline Task #84 | 0.1267 | 0.1024 | 1.237 | — |

## 4 Gate 验收 (FINAL)

### Gate 1 (Setup) ✅ PASS — DDP 4 卡启动, HAB 加载 final_cs=[0.7792]*3, **LR=4e-4 修复确认** (ep1 loss 5.33 vs Issue #36 5.79, -8.1%)

### Gate 2 (训练健康) ✅ PASS — loss 单调 5.33→2.62, 4 GPU 26-32% util, 无 NaN/Inf

### Gate 3 (用户 valid 目标) ✅ **PASS 大幅** — ep10=0.1110 > 0.1083 (+0.0027), ep55 峰=0.1248 (+0.0165), 接近 v4 SOTA valid 0.1267

### Gate 4 (test_R@10) ⚠️ **PARTIAL-GO** — test_R@10=**0.0980** < Issue #36 0.0912 (+0.0068 改进) < baseline 0.1024 (-0.0044) < v4 SOTA 0.1031 (-0.0051) < v85h 0.1042 (-0.0062)

## 改动文件

1. `common/stage3/stage3_train_pureT5_v3e_lr4e4.py` (新增 fork, LR=4e-4)
2. `taskA/stage3/taskA_stage3_pureT5_v3e_lr4e4_ddp.py` (新增 wrapper)
3. `taskA/stage4/taskA_stage4_pureT5_v3e_lr4e4_ddp.py` (新增 wrapper)
4. `verdicts/_misc/orphan/issue37_v3e_lr4e4_ddp/interim_launch.md`
5. `verdicts/_misc/orphan/issue37_v3e_lr4e4_ddp/interim_ep10_pass.md`
6. `verdicts/_misc/orphan/issue37_v3e_lr4e4_ddp/interim_ep20_pass.md`
7. `verdicts/_misc/orphan/issue37_v3e_lr4e4_ddp/interim_ep30_pass.md`
8. `verdicts/_misc/orphan/issue37_v3e_lr4e4_ddp/interim_ep40_pass.md`
9. `verdicts/_misc/orphan/issue37_v3e_lr4e4_ddp/interim_ep55_pass.md`
10. `verdicts/_misc/orphan/issue37_v3e_lr4e4_ddp/final_result.md` (本文件)

## Commit chain

- `43b69c5` Issue #37 launch 修复 Issue #36 配比
- `7a5c718` Issue #37 ep10 PASS valid_R@10=0.1110
- `0cfccd8` Issue #37 ep20 valid_R@10=0.1176 新峰
- `d4fdc2a` Issue #37 ep30 valid_R@10=0.1189 新峰
- `875d097` Issue #37 ep40 valid_R@10=0.1237 新峰
- `5b9b7c3` Issue #37 ep55 valid_R@10=0.1248 新峰 距 v4 SOTA 仅 0.0019

## 关键产物

- Stage3 ckpt: `/home/wlia0047/ar57_scratch/wenyu/full/stage3_pureT5_v3e_lr4e4_ddp_hyp_v2/HG_Rec_best.pth` (ep55 峰, 22.5MB)
- Stage4 verdict: `/home/wlia0047/ar57_scratch/wenyu/full/stage4_pureT5_v3e_lr4e4_ddp_hyp_v2/eval_test.json`

## 结论与根本原因

**LR=4e-4 修复有效 (test +0.0068 vs Issue #36)**, 但 **v3e 框架整体仍未达 baseline 0.1024 / v4 SOTA 0.1031**.

**核心观察**:
- valid_R@10 Issue #37=0.1248 (接近 v4 0.1267, -0.0019) → 模型本身有效
- test_R@10 Issue #37=0.0980 (-0.0044 vs baseline) → 泛化失败
- valid/test ratio=1.273 (vs v4 1.229) → 仍过拟合

**v3e vs v4/v85h 关键差异 (test 失败根因)**:
| 路径 | SID | test |
|------|-----|------|
| v4 SOTA | baseline SID | 0.1031 ★ |
| v85h SOTA | v15 SID | 0.1042 ★★★ |
| **Issue #37** | hyp_v2 SID (v3e fork) | 0.0980 ❌ |

**v3e hyp_v2 SID 不达 baseline**: hyp_v2 SID 是 taskA 自研 (capmatch κ 异质), v15 SID 是 baseline 原配. 即"框架可改" 但"SID 选错" → 不超 baseline.

**vs v4 路径**:
- v4 = DDP+LR=4e-4+HAB+baseline SID = 0.1031
- Issue #37 = DDP+LR=4e-4+HAB+hyp_v2 SID = 0.0980
- **唯一差异: SID**. hyp_v2 SID 替换 baseline SID 后 test -0.0051.

**vs v85h 路径**:
- v85h = DDP+LR=4e-4+v15 SID+cosine = 0.1042
- Issue #37 = DDP+LR=4e-4+hyp_v2 SID = 0.0980
- **唯一差异: SID + cosine**. v15 SID + cosine 协同.

### 教训

1. **LR 配比必须按 batch 缩放**: DDP batch=1024 → LR=4e-4 (Issue #36 LR=1e-4 欠学习, Issue #37 LR=4e-4 修复)
2. **SID 选择是 test 关键**: hyp_v2 SID (v3e fork) 替代 baseline/v15 SID → test 显著下降 (-0.005)
3. **DDP 大 batch 不能解决 SID 错配问题**: Issue #36/37 DDP 路径均不超 baseline
4. **本环境真 SOTA**: v85h (v15 SID+cosine) 0.1042 > v4 (baseline SID) 0.1031 > baseline 0.1024

### 后续建议 (不立即执行, 待用户决策)

1. **(推荐) Issue #38** = v3e+LR=4e-4+DDP 4 卡 + **v15 SID** (替换 hyp_v2) + HAB → 估 test 0.105-0.115 (vs v85h 0.1042)
2. **(可选) Issue #38 alt** = v3e+LR=4e-4+DDP 4 卡 + baseline SID (v4 同配) + HAB → 估 test ~0.1031 (= v4)
3. **(存档) Issue #37** 作为 v3e+hyp_v2 SID 路径的最佳 test (0.0980), 比单卡+HAB (0.0982) 略低
4. **重启 v85h 路径复现**: 验证 0.1042 SOTA 仍稳定

### 本环境 SOTA 最终状态

| 路径 | test_R@10 | 来源 |
|------|-----------|------|
| **v85h** (DDP+LR=4e-4+v15 SID+cosine) | **0.1042** ★★★ | Issue #141 v85h |
| **v4** (DDP+LR=4e-4+HAB+baseline SID) | **0.1031** ★ | Issue #141 v85g |
| baseline Task #84 | 0.1024 | — |
| Issue #37 (DDP+LR=4e-4+HAB+hyp_v2) | 0.0980 ⚠️ | Issue #37 (PARTIAL-GO) |
| v3e+HAB 单卡 (Issue #35) | 0.0982 | Issue #35 |
| v3e+HAB+DDP LR=1e-4 (Issue #36) | 0.0912 ❌ | Issue #36 (NO-GO) |
| v3e plain 单卡 (Issue #33) | 0.0955 | Issue #33 |

**用户目标 "复现 0.108"**:
- ✅ valid_R@10=0.1083 达成 (DDP ep10=0.1110, 单卡+HAB ep25=0.1094)
- ❌ test_R@10 ≥ 0.1024 baseline 失败 (Issue #37 0.0980 < 0.1024)

## Why & How to apply

**Why**: 用户持续触发"复现 0.108"任务 (Issue #33 v3e 单卡 test=0.0955 FAIL). 用户授权改 framework. Issue #35 加 HAB 单卡 test=0.0982. Issue #36 加 DDP 4 卡 + LR=1e-4 失败 test=0.0912 (DDP LR 配比错误). Issue #37 修复 LR 配比 (LR=1e-4 → 4e-4, 配 DDP batch=1024) test=0.0980 (+0.0068 vs Issue #36). 但 v3e hyp_v2 SID 替换 baseline SID 是 test 失败的根因 (valid/test ratio 1.273 vs v4 1.229).

**How to apply**:
- Issue #37 PARTIAL-GO 闭环, LR=4e-4 修复确认但 v3e hyp_v2 SID 路径整体不超 baseline
- 真实本环境 SOTA test_R@10=0.1042 (v85h, v15 SID+cosine 协同)
- 用户"复现 0.108" valid 目标 ✅ 已达成 (ep10 DDP 0.1110), test 目标 ❌ v3e 框架无法解决
- 下一步应转向 Issue #38 = v3e+DDP 配比 + **v15 SID** (而非 hyp_v2)
