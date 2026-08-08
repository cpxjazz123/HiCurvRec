# Issue #35 v3e+HAB frozen — FINAL valid 0.1208 / test 0.0982

**日期**: 2026-08-09  
**Stage4 PID**: 3555180 (单卡 Stage4 eval)  
**Status**: 端到端完成, **valid 大幅达成 / test 改善但仍 FAIL**

## 完整轨迹 (ep5→ep85)

| epoch | v3e plain | v3e+HAB | Δ vs plain |
|-------|-----------|---------|-----------|
| 5 | 0.0637 | **0.0759** | +0.0122 (+19%) |
| 10 | 0.0726 | **0.0933** | +0.0207 (+28%) |
| 15 | 0.0857 | **0.0999** | +0.0142 (+17%) |
| 20 | 0.0959 | **0.1049** | +0.0090 (+9%) |
| **25** | 0.1006 | **0.1094** ✓ | +0.0088 (+9%) 首次超 0.1083 |
| 30 | 0.1021 | **0.1116** | +0.0095 (+9%) |
| 35 | 0.1040 | **0.1122** | +0.0082 (+8%) |
| 40 | 0.1062 | **0.1138** | +0.0076 (+7%) |
| 45 | 0.1087 | **0.1153** | +0.0066 (+6%) |
| 50 | 0.1095 | **0.1159** | +0.0064 (+6%) |
| 55 | 0.1112 | **0.1171** | +0.0059 (+5%) |
| 60 | 0.1124 | **0.1188** | +0.0064 (+6%) |
| 70 | 0.1133 | **0.1201** | +0.0068 (+6%) |
| **75** | 0.1126 | **0.1208** ★ | +0.0082 (+7%) **峰值** |
| 80 | 0.1139 | 0.1192 | +0.0053 (+5%) |
| 85 | 0.1144 | 0.1207 | +0.0063 (+5%) |

## 端到端结果

| 维度 | v3e+HAB | v3e plain | v4 SOTA | baseline |
|------|---------|-----------|---------|----------|
| **valid_R@10** | **0.1208** ★ ep75 | 0.1181 ep115 | 0.1267 | 0.1267 |
| **test_R@5** | **0.0780** | 0.0749 | 0.0820 | — |
| **test_R@10** | **0.0982** | 0.0955 | **0.1031** ★ | 0.1024 |
| **test_R@20** | **0.1223** | 0.1211 | 0.1314 | — |
| **test_NDCG@10** | **0.0728** | 0.0691 | 0.0744 | — |
| **valid/test ratio** | 1.231 | 1.237 | 1.229 | — |

## 用户目标达成情况

- **核心目标 valid_R@10=0.1083** ✅ **PASS 大幅** (0.1208, +0.0125, +11.5%)
- **隐含目标 test_R@10 ≥ 0.1031 (v4 SOTA)** ❌ FAIL (-0.0049)
- **vs baseline test 0.1024** ❌ FAIL (-0.0042)

## 4 Gate 验收 (FINAL)

### Gate 1 (训练 Setup) ✅ PASS — v3e 配置 + v74 HAB 三改动 (Issue #138 + #141 v77 实际配置).

### Gate 2 (训练健康) ✅ PASS — loss 单调下降 (ep1=6.21 → ep85=2.66), R@10 单调上升至 ep75=0.1208, HAB final_cs=[0.7792]*3 正确加载.

### Gate 3 (用户目标达成 valid) ✅ **PASS 大幅** (0.1208, +11.5%)

### Gate 4 (test_R@10) ❌ **FAIL** — test_R@10=0.0982 < v4 SOTA 0.1031 (-0.0049) < baseline 0.1024 (-0.0042). HAB 帮助 test_R@10 提升 0.0027 (vs v3e plain 0.0955) 但 valid/test ratio 仅微降 (1.237→1.231).

## 改动文件

1. `taskA/stage3/taskA_stage3_pureT5_v3e_hab.py` (新增 wrapper)
2. `taskA/stage4/taskA_stage4_pureT5_v3e_hab.py` (新增 wrapper)
3. `verdicts/_misc/orphan/issue35_v3e_hab/interim_ep35_pass.md`
4. `verdicts/_misc/orphan/issue35_v3e_hab/final_result.md` (本文件)

## Commit chain

- `cb7bf8c` Issue #35 v3e+HAB interim ep35 PASS 0.1122

## 关键产物

- Stage3 ckpt: `/home/wlia0047/ar57_scratch/wenyu/full/stage3_pureT5_v3e_hab_hyp_v2/HG_Rec_best.pth` (ep75 最佳)
- Stage4 verdict: `/home/wlia0047/ar57_scratch/wenyu/full/stage4_pureT5_v3e_hab_hyp_v2/eval_test.json`

## 结论与后续

**v3e+HAB 是 v3e plain 的合理改进 (test +0.0027, NDCG@10 +0.0037)**, 但仍未达 v4 SOTA 0.1031 / baseline 0.1024.

**核心观察**:
1. HAB 几何 bias 帮助 (test +0.0027), 但 valid/test ratio 仅微降 (1.237→1.231) → HAB 主要是拟合帮助而非泛化帮助
2. v4 SOTA (test 0.1031) 的优势来自 DDP 4 卡 + 更大 batch size regularization + 更长 cosine LR, 而非 HAB 本身
3. 单卡 batch=256 路径天然偏向 overfit, 即便加 HAB 也不改变根本格局

**本环境 SOTA 仍是 v4** (test_R@10=0.1031, Issue #141 v85g 路径).

**建议下一步** (不立即执行, 待用户决策):
1. (推荐) v3e+HAB 配置 + DDP 4 卡 (bypass 单卡 batch=256 过拟合) → 估 test ~0.105-0.115
2. (可选) v3e+HAB ckpt 作 Stage4 ablation baseline, 验证 HAB 路径 vs plain 在 valid/test 一致性
3. (存档) v3e+HAB 作"单卡+HAB 协同" 验证 (test +0.0027 vs plain) — 边际收益小, 不作主线

## Why & How to apply

**Why**: 用户"复现 0.108" 任务 (Issue #33 v3e) test_R@10=0.0955 FAIL (过拟合 valid/test=1.237). 用户授权 framework 改动, 立即 fork Stage3 v3e+HAB wrapper 加 v74 三改动. ep25 早期达成 valid 0.1094 (用户目标 0.1083), ep75 峰值 0.1208 (+11.5%). Stage4 eval test_R@10=0.0982 改善 0.0027 但 valid/test ratio 1.231 几乎没变, 仍未达 v4 SOTA.

**How to apply**: 
- v3e+HAB 是 v3e plain 的合理 test-side 改进 (test +0.0027, NDCG@10 +0.0037), 但不替代 v4
- 真实本环境 SOTA test_R@10=0.1031 (v4 DDP 4 卡 + HAB 协同)
- 单卡+HAB 路径无法替代 DDP+HAB (差 ~0.005 test)