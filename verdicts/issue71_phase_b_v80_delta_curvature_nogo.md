# Issue #71 Phase B v80 Curvature-Differential HAB NO-GO

## 4 Gate 结果 (test 评估用 ep45 best ckpt, valid 0.1272)

- **Gate 1 PASS**: ΔD 代码改动 (hyperbolic_attention_bias.py + stage3 + stage4), py_compile + argparse OK, precheck 数值审计 PASS, commit 189775b push done
- **Gate 2 PASS**: DDP 4 卡 11s/epoch, ES=6/10 at ep75, best ep45
- **Gate 3 MARGINAL**: valid R@10=0.1272 (vs v77 0.1312 -0.0040, vs v78 0.1335 -0.0063, vs baseline 0.1267 +0.0005). 训练曲线健康 (loss 4.75→1.90, 无 NaN/Inf)
- **Gate 4 FAIL NO-GO**: test R@10 = **0.1020**, -0.0072 vs v78 (0.1092), -0.0020 vs v77 (0.1040), -0.0004 vs baseline (0.1024)

## test 评估

| 指标 | baseline | v77 (HAB) | v78 (DECOR) | v79-delayed | **v80 (ΔD)** |
|------|---------:|----------:|------------:|------------:|-------------:|
| test R@5 | 0.0819 | 0.0841 | 0.0846 | 0.0815 | **0.0815** |
| test R@10 | 0.1024 | 0.1040 | 0.1092 | 0.1012 | **0.1020** |
| test R@20 | 0.1283 | 0.1301 | 0.1372 | 0.1260 | **0.1275** |
| test NDCG@10 | 0.0755 | 0.0781 | 0.0764 | 0.0749 | **0.0758** |
| ratio (valid/test) | 1.237 | 1.262 | 1.222 | 1.257 | **1.245** |

## 关键发现 — ΔD 比 D_hyp 略差

| 方案 | test R@10 | vs baseline | ratio | ΔD vs D_hyp |
|------|-----------|-------------|-------|-------------|
| baseline | 0.1024 | — | 1.237 | — |
| v77 (HAB frozen, D_hyp) | 0.1040 | +0.0016 | 1.262 | (基线) |
| v78 (DECOR) | 0.1092 | +0.0068 | 1.222 | — |
| v79-delayed (DECOR+HAB+warmup) | 0.1012 | -0.0012 | 1.257 | — |
| **v80 (HAB ΔD 曲率差分)** | **0.1020** | **-0.0004** | **1.245** | **-0.0020 vs v77** |

**v80 (ΔD) test 0.1020 vs v77 (D_hyp) test 0.1040: -0.0020**. ΔD 比完整 D_hyp 退步, 假设"ΔD 剥离码字距离尺度更纯粹"被证伪.

## ΔD 为何不奏效 — 三层原因

1. **L2 κ=-0.093 接近 0 → ΔD 几乎为 0**: precheck 显示 L2 ΔD median=4.1e-5, T5 学到的 bias 信号极弱. 等同于 L2 bias 完全失效 → 损失 L2 层曲率贡献
2. **ΔD 与 D_hyp 共享同一坐标系但量级小**: ΔD range=[0.0008, 0.0153] vs D_hyp range=[0.21, 0.80]. ΔD median 标准化后与 D_hyp 量纲接近, 但曲率信号稀疏 (大部分 L2 pair 是零)
3. **T5 学到的不是"曲率贡献"而是"距离残差"**: ΔD 是数学定义的残差, 但 T5 attention 不一定能识别这种高阶信号. 完整 D_hyp 包含码字位置 + 曲率两层信息, T5 可能更容易学

## valid vs test 不匹配分析

| 方案 | valid | test | gap |
|------|-------|------|-----|
| baseline | 0.1267 | 0.1024 | 0.0243 |
| v77 HAB | 0.1312 | 0.1040 | 0.0272 |
| v78 DECOR | 0.1335 | 0.1092 | 0.0243 |
| **v80 ΔD HAB** | **0.1272** | **0.1020** | **0.0252** |

v80 valid/test gap=0.0252 (vs v77 0.0272 略好, vs baseline 0.0243 略差). 过拟合程度中等, 不如 v79-delayed (0.0260) 严重.

## 路线全景 (Issue #71 全部完成, 全 NO-GO)

| Issue | 方案 | test R@10 | vs baseline | ratio |
|-------|------|-----------|-------------|-------|
| baseline | T5 only | 0.1024 | — | 1.237 |
| #138 v74 | HAB frozen + WD + dropout | 0.1063 | +0.0039 | 1.234 |
| #141 v77 | Stage1 + HAB | 0.1080 | +0.0056 | 1.215 |
| #68 v78 | DECOR + 抗 trap | 0.1092 | +0.0068 | 1.225 |
| **#71 Phase A** | **v78 + Delayed HAB** | **0.1012** | **-0.0012** | **1.257 ← NO-GO** |
| **#71 Phase B** | **v77 + ΔD HAB** | **0.1020** | **-0.0004** | **1.245 ← NO-GO** |

**Issue #71 三个 Phase (含 Phase C 待做) 全部 NO-GO 风险高**. 当前曲率路线最佳仍是 v77 (0.1080, ratio 1.215), DECOR 路线最佳是 v78 (0.1092, ratio 1.225).

## 教训

**ΔD 剥离码字距离尺度 ≠ 更纯粹信号**. 实际上 L2 κ 接近 0 时 ΔD 信号几乎为零, T5 学到的 bias 变得稀疏; 而完整 D_hyp 包含码字位置 + 曲率两层信息, T5 反而能学.

**Why:** ΔD 是数学定义的曲率残差, 但 T5 attention 学习的不是"曲率贡献"而是"什么 token 关注什么 token". 完整 D_hyp 提供了更丰富的码字位置信号, ΔD 反而割裂了这些信息.

**How to apply:** 后续 HAB 改进不应走 ΔD 路线. 应改走以下方向:
1. **λ_max 调度**: 训练前期 λ_max 小 (HAB 弱信号, T5 先学 embedding), 后期 λ_max 大 (HAB 强信号), 类似 v79 warmup 思想但应用到 λ_max 而非 residual
2. **Stage1 radius 强化**: 沿 v77 路线继续强化 per-item radius (R_MAX 0.99→0.95, sigmoid_temp 3→5)
3. **曲率可学习 λ_l**: 让每层 λ_l 独立可学习, T5 自动选最优曲率强度

## Phase C 状态

**Issue #71 Phase C (行为监督曲率) NO-GO 风险极高, 不建议继续**.

Phase C 设计: Stage2 训练加 L_beh-curv InfoNCE 损失从真实用户序列监督曲率. 但:
1. Phase A 已证 DECOR + HAB 不可叠加 (issue69 v79 NO-GO)
2. Phase B 已证 ΔD 路线无 ROI
3. Phase C 是另一类改造 (Stage2 训练目标), 但与现有 Stage2 SID + Stage3 HAB 集成有未知冲突
4. 用户目标 test R@10 ≥ 0.11, 当前最佳 v78 0.1092 仍差 0.0008; 继续实验 ROI 边际

**建议**: 接受 v78 (test 0.1092) 作为本轮最佳曲率路线, 或重启新 issue 单独设计 Phase C (Stage2 行为监督) + 新架构 (DECOR-bin 集成)。

## 产物

- verdict: verdicts/issue71_phase_b_v80_delta_curvature_nogo.md (本文档)
- 代码: commit 189775b (push done)
- 训练: taskA/_history/issue71_v80_delta_curvature_hab/HG_Rec_best.pth (ep45)
- 评估: taskA/_history/issue71_v80_delta_curvature_hab/eval_test/eval_test.json
- precheck: verdicts/issue71_phase_b_precheck_pass.md
- SID: 06af0fed (Stage2 hyp_v2_capmatch_1000ep)
- 实际 Stage2 ckpt: taskA_stage2_issue61/hrqvae_kappa_sync.ckpt (κ=[-0.229, -0.187, -0.093], 历史 v77/v78/v79 一致)

## 时间线

- 2026-08-07 13:06 — DDP 4 卡 v80 启动 (ep 0)
- 2026-08-07 13:08 — ep5 first eval valid 0.1038
- 2026-08-07 13:13 — ep30 best valid 0.1253
- 2026-08-07 13:16 — ep45 best valid 0.1272 (BG 触发)
- 2026-08-07 13:22 — ep75 ES 6/10, best 仍 0.1272
- 2026-08-07 ~13:30 — 用户指示用 best ckpt 跑 test (训练继续)
- 2026-08-07 ~13:32 — test eval 启动 (ep45 best ckpt)
- 2026-08-07 ~13:35 — test R@10 = 0.1020 (NO-GO, ratio 1.245)