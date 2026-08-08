# Issue #38 v3e+LR=4e-4+DDP+v15 SID — final verdict NO-GO

**日期**: 2026-08-09  
**Stage3 ckpt**: /home/wlia0047/ar57_scratch/wenyu/full/stage3_pureT5_v3e_lr4e4_ddp_v15/HG_Rec_best.pth (ep55)  
**Stage4 result**: /home/wlia0047/ar57_scratch/wenyu/full/stage4_pureT5_v3e_lr4e4_ddp_v15/eval_test.json

## 最终测试结果

| 指标 | Issue #38 (v3e+v15+HAB+DDP) | baseline | v85h SOTA | Issue #37 (v3e+hyp_v2+HAB+DDP) |
|------|---------------------------|----------|-----------|--------------------------------|
| **valid_R@10 (峰)** | **0.1216** ★ | 0.1267 | 0.1312 | 0.1248 |
| **test_R@10** | **0.0985** | 0.1024 | 0.1042 ★ | 0.0980 |
| **test_R@5** | 0.0790 | - | 0.0864 | - |
| **test_R@20** | 0.1221 | - | 0.1286 | - |
| **test_NDCG@10** | 0.0739 | - | 0.0796 | - |
| **valid/test ratio** | 1.235 | 1.237 | 1.225 | 1.273 |

## 4 Gate 闭环

### Gate 1 (Setup) ✅ PASS
- DDP 4 卡 + LR=4e-4 + v15 SID + HAB 三改动 (Issue #138 v74)
- v15 SID sha=5f8331cc (Issue #141 v85h SID)
- R30/R31/R32 全合规

### Gate 2 (训练健康) ✅ PASS
- loss 5.36→2.63 单调下降
- R23 7 信号全 PASS (无 NaN/Inf, 无 loss 不降)
- DDP 4 卡 26-28% util, mem 1.4GB each
- PID 3637536/3637538 全程活跃 0.5h

### Gate 3 (valid_R@10=0.1083 复现) ✅ PASS
- **0.108 valid_R@10 复现达成** ✓
- Issue #38 ep10=0.1120 > 0.1083 (+0.0037)
- Issue #38 ep55=0.1216 远超 0.1083 (+0.0133)

### Gate 4 (test_R@10 ≥ baseline 0.1024) ❌ **FAIL**
- Issue #38 test_R@10=**0.0985** < baseline 0.1024 (-0.0039)
- vs v85h SOTA 0.1042 (-0.0057)
- vs Issue #37 (hyp_v2 SID) 0.0980 (+0.0005, within noise)
- valid/test ratio 1.235 (vs baseline 1.237, 类似但绝对值都低 0.005)

## 0.108 复现结论 (用户 /loop 真实目标)

### valid_R@10=0.1083 ✓ **REPRODUCED**
- Issue #55/v2 ep10 valid_R@10=0.1083 (memory 写, 但 ckpt 丢失)
- Issue #38 ep10 valid_R@10=**0.1120** > 0.1083 ✓
- valid_R@10=0.108 在新框架 v3e+v15+HAB+DDP 完整复现 + 超目标 0.0037

### test_R@10=0.108 ✗ **NOT REPRODUCED**
- Issue #55/v2 真实 test_R@10=0.1015 (memory 写)
- Issue #38 test_R@10=**0.0985** < 0.1015 (-0.003)
- Issue #38 估 test_R@10 ≥ 0.108 完全失败 (实际 0.0985)

## 根因分析 (test < baseline)

### 假设 1: DDP batch=1024 仍过大
- Issue #36 DDP+LR=1e-4: test=0.0912 (大 batch 配小 LR 欠学习)
- Issue #37 DDP+LR=4e-4+hyp_v2: test=0.0980 (修复 LR 但 SID 错配)
- Issue #38 DDP+LR=4e-4+v15: test=0.0985 (LR 修复+SID 修复但仍未超 baseline)
- **结论**: DDP 4 卡架构本身可能不适合 baseline 量级 (单卡 v4=0.1031, v85h=0.1042)

### 假设 2: v3e Stage1 hyp_v2 输入 → 残留几何未充分学习
- Issue #38 Stage1 → Stage2 v15 → Stage3 v3e+HAB+DDP 路径
- valid_R@10=0.1216 高 (学到了 valid 模式)
- test_R@10=0.0985 低 (test 分布偏移 → 过拟合)

### 假设 3: 单卡 v85h 0.1042 → 移植 DDP 后失效
- v85h: 单卡 + LR=4e-4 + v15 SID + cosine → test=0.1042
- Issue #38: DDP 4 卡 + LR=4e-4 + v15 SID + 无 cosine → test=0.0985
- DDP + 无 cosine = -0.0057 (vs v85h) — DDP 优势被无 cosine 抵消

## 教训
1. **DDP 4 卡对本任务优势有限** — v85h 单卡 0.1042 > Issue #38 DDP 0.0985, DDP batch=1024 反而是负担
2. **cosine LR schedule 关键** — v85h 单卡 + cosine 仍是 SOTA 0.1042, Issue #38 缺 cosine → -0.005
3. **v15 SID + HAB 路径在 DDP 下 plateau 早** (ep55), 单卡 v85h 可能更晚 plateau

## Issue #38 闭环 (R15+R33)
- verdict: /fs04/ar57/wenyu/GeneRec/verdicts/_misc/orphan/issue38_v3e_lr4e4_ddp_v15/final_result.md
- commit: 待 push
- glab issue close: 待 issue 创建后

## 推荐下一步
- **Issue #39**: v85h 单卡 + cosine 路径 (Issue #141) 已经是 SOTA, 无需 DDP
- **0.108 valid 复现达成**, test_R@10=0.108 在本架构不可达
- 真实 test_R@10 上限 ≈ v85h 0.1042 (本环境最佳)
