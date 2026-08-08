# Issue #71 v82 Stage1 radius 强化 NO-GO

## 4 Gate 结果 (test 评估用 ep75 best ckpt, valid 0.1259)

- **Gate 1 PASS**: Stage1 sigmoid_temp 3→5 + R_MAX 0.99→0.95 (common/stage1/stage1_hyperbolic.py), argparse + py_compile PASS. Stage2 用 taskA_stage2_v82_capmatch 新 ckpt (SID ba51e84d). Stage3 v77 HAB 配置 (frozen residual + λ_max 0.20 + WD 0.01 + dropout 0.20)
- **Gate 2 PASS**: DDP 4 卡 11s/epoch, 训练健康 (loss 1.79→1.72), ep120 ES=9/10 (旧 v73 valid 规则触发)
- **Gate 3 MARGINAL**: valid R@10=0.1259 (vs v77 0.1312 -0.0053, vs v78 0.1335 -0.0076, vs baseline 0.1267 -0.0008)
- **Gate 4 FAIL NO-GO**: test R@10 = **0.0992**, -0.0052 vs baseline (0.1024), -0.0048 vs v77 (0.1040), -0.0100 vs v78 (0.1092)

## test 评估

| 指标 | baseline | v77 (HAB) | v78 (DECOR) | v80 (ΔD) | **v82 (radius强化)** |
|------|---------:|----------:|------------:|---------:|---------------------:|
| test R@5 | 0.0819 | 0.0841 | 0.0846 | 0.0815 | **0.0806** |
| test R@10 | 0.1024 | 0.1040 | 0.1092 | 0.1020 | **0.0992** |
| test R@20 | 0.1283 | 0.1301 | 0.1372 | 0.1275 | **0.1200** |
| test NDCG@10 | 0.0755 | 0.0781 | 0.0764 | 0.0758 | **0.0750** |
| ratio (valid/test) | 1.237 | 1.262 | 1.222 | 1.245 | **1.269** |
| valid R@10 | 0.1267 | 0.1312 | 0.1335 | 0.1272 | **0.1259** |

**v82 valid 退化 + test 退化**, ratio 1.269 (vs v77 1.262, v78 1.222), 过拟合与 v77 持平.

## 改动

| Stage | 改动 | 结果 |
|-------|------|------|
| Stage1 v82 | R_MAX 0.99→**0.95** + sigmoid_temp 3→**5** | std_norm 0.0405→**0.0641** (+58%), unique_radius 1843→**2665** (+45%) |
| Stage2 v82 capmatch | 沿用 Stage1 v82 输出 | κ=[**-0.239, -0.225, -0.180**] (vs v77 [-0.229, -0.187, -0.093] 更平), util_3digit=[0.92, 0.95, 0.78] |
| Stage3 v82 | HAB v77 全套配置 + ep75 best ckpt | valid 0.1259, test 0.0992 |

## 根因分析 — κ 全面变平导致 HAB 失效

**v82 κ 比 v77 更平** (Layer 2 κ=-0.180 vs -0.093, 反而**绝对值更大**但**层间方差更小**):
- v77: κ=[-0.229, -0.187, -0.093] (层间方差 0.0042)
- v82: κ=[-0.239, -0.225, -0.180] (层间方差 0.0009) ← **层间方差衰减 4.7x**

HAB 的曲率信号来自 κ 层间差异. v82 三层 κ 几乎一致 → HAB 学到的 bias 几乎是常数 → 失去 per-layer 区分能力 → test 退步.

**Why**: R_MAX 0.95 (vs v77 0.99) 把 v 推到球面边界, stage2 expmap0 也把码字推到边界, 不同 κ 投影差异被"边界效应"压缩. sigmoid_temp 5 (vs 3) 让 radius 极端化, 进一步加剧码字聚集.

## 路线全景 (Issue #71 全部 NO-GO + v82 加入 NO-GO)

| Issue | 方案 | test R@10 | vs baseline | ratio |
|-------|------|-----------|-------------|-------|
| baseline | T5 only | 0.1024 | — | 1.237 |
| #138 v74 | HAB frozen + WD + dropout | 0.1063 | +0.0039 | 1.234 |
| #141 v77 | Stage1 + HAB | 0.1080 | +0.0056 | 1.215 |
| #68 v78 | DECOR + 抗 trap | 0.1092 | +0.0068 | 1.225 |
| #71 Phase A | v78 + Delayed HAB | 0.1012 | -0.0012 | 1.257 |
| #71 Phase B | v77 + ΔD HAB | 0.1020 | -0.0004 | 1.245 |
| **v82** | **v77 + Stage1 radius 强化** | **0.0992** | **-0.0032** | **1.269** |

**当前曲率路线最佳仍是 v77 (test 0.1080, ratio 1.215), DECOR 路线最佳 v78 (test 0.1092, ratio 1.225). 0.11 目标仍未达.**

## 教训

**Stage1 radius 强化 (R_MAX 0.99→0.95 + sigmoid_temp 3→5) 在 v77 基础上是负向改动**. 

**Why**: 强化 radius 多样性 → expmap0 把更多码字推到边界 → 不同 κ 投影差异被压缩 → κ 层间方差衰减 → HAB bias 信号弱化 → test 退化.

**How to apply**: 
1. Stage1 radius 不能与 HAB 同时强化, 二者路径冲突
2. κ 层间方差是 HAB 的关键, 强化 Stage1 应保持 κ 多样性, 而非压缩
3. 下一轮 (v83+) 不应继续 Stage1 radius 强化, 应改走 Stage3 HAB 路径 (例如 v83 loss-based 早停 + v74 HAB 配置重跑, 或 v78 DECOR 路线延续)

## 产物

- verdict: verdicts/issue71_v82_stage1_radius_strong_nogo.md (本文档)
- Stage1: taskA/_data/Instruments/Instruments_t5_hyp_v2_v82_r095_t5.parquet
- Stage2: taskA/_history/taskA_stage2_v82_capmatch/ (SID ba51e84d, κ=[-0.239, -0.225, -0.180])
- Stage3: taskA/_history/issue71_v82_stage1_radius_strong/stage3/HG_Rec_best.pth (ep75)
- Stage4: taskA/_history/issue71_v82_stage1_radius_strong/eval_test/eval_test.json (test R@10=0.0992)

## 时间线

- 2026-08-07 13:35 — Issue #71 Phase B precheck PASS, 同时 v82 Stage1 启动 (R_MAX 0.95 + sigmoid_temp 5)
- 2026-08-07 13:42 — Stage1 v82 完成 (std_norm +58%)
- 2026-08-07 13:50 — Stage2 v82 capmatch 启动
- 2026-08-07 13:55 — Stage3 v82 DDP 4 卡启动 (v77 HAB 配置)
- 2026-08-07 14:18 — ep115, valid 仍锁 0.1259 (ES=8/10)
- 2026-08-07 14:19 — ep120 ES=9/10
- 2026-08-07 14:20 — 用户指示 kill 训练 + 跑 Stage4
- 2026-08-07 14:21 — Stage4 eval 启动 (ep75 best ckpt)
- 2026-08-07 14:22 — test R@10 = 0.0992 (NO-GO)
