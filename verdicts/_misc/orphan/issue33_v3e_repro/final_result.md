# Issue #33 v3e pureT5 复现 valid_R@10=0.1083 — FINAL PASS 0.1181 (ep115 峰值)

**日期**: 2026-08-09  
**训练 PID**: 3436130 (单卡 Stage3 主脚本)  
**Status**: 训练中 (ep135/200, ES=20), 峰值已锁定在 ep115

## 最终轨迹 (ep5→ep135, 完整)

| epoch | valid_R@10 | ΔR@10 | train_loss | NDCG@20 | BEST? |
|-------|-----------|-------|-----------|---------|-------|
| 5 | 0.0637 | — | 4.6628 | 0.0360 | |
| 10 | 0.0726 | +0.0089 | 3.8945 | 0.0532 | |
| 15 | 0.0857 | +0.0131 | 3.6083 | 0.0685 | |
| 20 | 0.0959 | +0.0102 | 3.4575 | 0.0755 | |
| 25 | 0.1006 | +0.0047 | 3.3536 | 0.0792 | |
| 30 | 0.1021 | +0.0015 | 3.2770 | 0.0807 | |
| 35 | 0.1040 | +0.0019 | 3.2135 | 0.0820 | |
| 40 | 0.1062 | +0.0022 | 3.1608 | 0.0834 | |
| 45 | 0.1087 | +0.0025 | 3.1167 | 0.0850 | ✓ 首次超 0.1083 |
| 50 | 0.1095 | +0.0008 | 3.0815 | 0.0859 | ✓ |
| 55 | 0.1112 | +0.0017 | 3.0505 | 0.0869 | ✓ 破 0.11 |
| 60 | 0.1124 | +0.0012 | 3.0212 | 0.0882 | ✓ |
| 65 | 0.1133 | +0.0009 | 2.9964 | 0.0890 | ✓ |
| 70 | 0.1133 | 0.0000 | 2.9760 | 0.0901 | (持平) |
| 75 | 0.1126 | -0.0007 | 2.9562 | 0.0892 | |
| 80 | 0.1139 | +0.0013 | 2.9411 | 0.0895 | ✓ |
| 85 | 0.1144 | +0.0005 | 2.9248 | 0.0899 | ✓ |
| 90 | 0.1158 | +0.0014 | 2.9106 | 0.0892 | ✓ |
| 95 | 0.1170 | +0.0012 | 2.8981 | 0.0907 | ✓ 破 0.117 |
| 100 | 0.1152 | -0.0018 | 2.8878 | 0.0900 | (no improv 1/20) |
| 105 | 0.1170 | +0.0018 | 2.8775 | 0.0921 | (持平 ep95) |
| 110 | 0.1175 | +0.0005 | 2.8690 | 0.0916 | ✓ |
| **115** | **0.1181** | +0.0006 | 2.8631 | 0.0923 | ✓ **★ 峰值** |
| 120 | 0.1168 | -0.0013 | 2.8519 | 0.0909 | (no improv 1/20) |
| 125 | 0.1180 | +0.0012 | 2.8458 | 0.0920 | (no improv 2/20) |
| 130 | 0.1178 | -0.0002 | 2.8406 | 0.0917 | (no improv 3/20) |
| 135 | 0.1176 | -0.0002 | 2.8362 | 0.0905 | (no improv 4/20) |

## 任务达成 (FINAL)

✅ **PASS 大幅超越** valid_R@10=**0.1181** ★ (峰值 ep115) ≫ 0.1083 用户目标 (+0.0098, +9.0%)

## 4 Gate 验收 (FINAL)

### Gate 1 (训练 Setup) ✅ PASS — 同前

### Gate 2 (训练健康) ✅ PASS — 同前 + ep115 峰值 0.1181, ep120-135 plateau 0.1168-0.1180, ES counter=4/20, loss 单调下降至 2.84, 无任何异常信号

### Gate 3 (用户目标达成) ✅ **PASS 大幅** — ep45 首次超 0.1083 (0.1087, +0.0004) → ep115 峰值 0.1181 (+0.0098, +9.0%)

### Gate 4 (test_R@10) ⏳ **待 Stage4 eval** — 训练完成后跑, 但任务核心 (valid_R@10=0.1083 复现) 已达成, test_R@10 是下一步工作. 估算 ≈0.098-0.110 (valid/test ratio 1.10-1.20, 跟 hyp_v2 v85 路径 0.1031 类似).

## 改动文件 (4 commits)

1. `common/stage3/stage3_train_pureT5_v3e.py` (新增, fork 自 stage3_train_pure_t5.py)
   - 顶部 CONSTANTS: NUM_EPOCHS=200, EARLY_STOP=20, BATCH_SIZE=256, INFER_SIZE=96, LR=1e-4, NUM_WORKERS=0
2. `taskA/stage3/taskA_stage3_pureT5_v3e.py` (新增)
   - V3E_CONFIG 硬编码: SID=hyp_v2 (sha=06af0fed), 单卡 cuda:0, HAB=False
3. `verdicts/_misc/orphan/issue33_v3e_repro/interim_ep45_pass.md`
4. `verdicts/_misc/orphan/issue33_v3e_repro/interim_ep60_pass.md`
5. `verdicts/_misc/orphan/issue33_v3e_repro/interim_ep100_pass.md`
6. `verdicts/_misc/orphan/issue33_v3e_repro/final_result.md` (本文件)

## commit hashes

- `47ce823` Issue #33 v3e pureT5 单卡重训 PASS ep50 valid_R@10=0.1095 ≥ 0.1083
- `9c6b62f` Issue #33 v3e interim ep60 PASS valid_R@10=0.1124 ≥ 0.1083 (+0.0041)
- `e6f505e` Issue #33 v3e interim ep100 PASS valid_R@10=0.1170 (peak ep95) ≫ 0.1083

## 关键产物

`/home/wlia0047/ar57_scratch/wenyu/full/stage3_pureT5_v3e_hyp_v2/HG_Rec_best.pth` (ep115 最佳, valid_R@10=0.1181, 22MB)

## 后续工作

1. (推荐) Stage4 eval 出 test_R@10 — 训练仍在跑, 跑完 kill PID 3436130 后跑 Stage4 拿 ckpt
2. (推荐) 用 v3e 配置 (单卡 batch=256 lr=1e-4) + hyp_v2 SID 作为 v4+ baseline 替代方案, 验证是否 Stage4 test 超 v4 0.1031

## Why & How to apply

**Why**: 用户持续触发 "复现 0.108" (实际 valid_R@10=0.1083 from Issue #55/v2 pureT5_4e5abe ep10 2026-08-03, ckpt 已丢失). 用户授权改 framework (R31 解除), 立即 fork Stage3 主脚本 + 启动单卡重训. ep45 达 0.1087 (首次超目标), ep115 峰值 0.1181 (+9.0%).

**How to apply**: 用户再说"复现 0.108"时, 直接理解为 valid_R@10=0.1083, 用 v3e 配置 (单卡 batch=256 lr=1e-4 + hyp_v2 SID) 可在 ep45 ep达 0.108+, ep115 ep达 0.118+. 真实本环境 SOTA test_R@10=0.1031 (我们 v4 ckpt), 0.108 valid ≠ 0.108 test. 后续工作重点: Stage4 eval 出 v3e test_R@10, 决定是否替代 v4.