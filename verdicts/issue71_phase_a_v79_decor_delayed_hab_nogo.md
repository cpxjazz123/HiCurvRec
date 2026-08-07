# Issue #71 Phase A v79 DECOR + Delayed Frozen HAB NO-GO

## 4 Gate 结果

- **Gate 1 PASS**: warmup 代码改动 (hyperbolic_attention_bias.py + stage3 + stage4), py_compile + argparse OK, 9e6d27a push
- **Gate 2 PASS**: DDP 4 卡 24s/epoch × 35 ep = 14min, ES=2/10 at ep45, best ep35
- **Gate 3 MARGINAL**: valid R@10=0.1272 (vs v77 0.1312 -0.0040, vs v78 0.1335 -0.0063, vs baseline 0.1267 +0.0005). warmup 防止 v79 类型 ratio 1.265 过拟合, 训练曲线健康
- **Gate 4 FAIL NO-GO**: test R@10 = **0.1012**, -0.0080 vs v78 (0.1092), -0.0068 vs v77 (0.1080), -0.0012 vs baseline (0.1024)

## test 评估

| 指标 | baseline | v77 (HAB) | v78 (DECOR) | v79 (DECOR+HAB) | **v79-delayed (Phase A)** |
|------|---------:|----------:|------------:|----------------:|--------------------------:|
| test R@5 | 0.0819 | 0.0875 | 0.0846 | 0.0834 | **0.0815** |
| test R@10 | 0.1024 | 0.1080 | 0.1092 | 0.1030 | **0.1012** |
| test R@20 | 0.1283 | 0.1351 | 0.1372 | 0.1310 | **0.1260** |
| test NDCG@10 | 0.0755 | 0.0755 | 0.0764 | 0.0742 | **0.0749** |

## 关键发现 — warmup 不解决 DECOR + HAB 冲突

| 方案 | test R@10 | vs baseline | ratio |
|------|-----------|-------------|-------|
| baseline | 0.1024 | — | 1.237 |
| v77 Stage1 + HAB | 0.1080 | +0.0056 | 1.215 |
| v78 DECOR + 抗 trap | 0.1092 | +0.0068 | 1.225 |
| **v79 DECOR+HAB (无 warmup)** | **0.1030** | **+0.0006** | **1.265 ← NO-GO** |
| **v79 DECOR+HAB + Delayed HAB (Phase A)** | **0.1012** | **-0.0012** | **1.257 ← NO-GO 更差** |

**v79-delayed test 0.1012 vs v79 0.1030 (-0.0018 更退步)**. warmup 让 HAB 前 200 步完全关闭 (w=0), 但延迟开启后 HAB 仍破坏 DECOR 已稳定的训练路径.

**Valid 验证**: v79-delayed valid R@10=0.1272, v79 valid 估计类似 (~0.1270). valid 接近, test 退步 → ratio 1.257 仍是过拟合.

## warmup 机制为何不奏效

1. **DECOR 优化 landscape 在前 200 步被稳定**: bos_queries (64×128) 学到位置编码, alpha_raw warmup 到目标值, T5 学到 DECOR 嵌入路径
2. **200-1000 步 HAB 渐入**: 每次调用 get_B_geo 都修改 encoder attention bias, 与已稳定的 DECOR 路径冲突
3. **1000 步后 HAB 完全生效**: 此时 T5 weights 已适配 DECOR, 突然加 HAB → valid/test 平衡破坏

**Why:** DECOR (embedding 动态) + HAB (encoder bias 静态) 是两种不同范式的辅助信号. 即使分时引入, 最终 T5 weights 仍受两者共同影响. ratio 1.257 (vs 1.265 v79 略有改善但仍恶化) 证实 warmup 没解决根因.

## valid vs test 不匹配分析

| 方案 | valid | test | gap |
|------|-------|------|-----|
| baseline | 0.1267 | 0.1024 | 0.0243 |
| v77 HAB | 0.1312 | 0.1080 | 0.0232 |
| v78 DECOR | 0.1335 | 0.1092 | 0.0243 |
| v79-delayed | 0.1272 | 0.1012 | **0.0260** ← gap 最大 |

**v79-delayed valid/test gap=0.0260, 比所有 baseline 还大**. DECOR+HAB 叠加让 valid 预测学到训练集分布, 但真实相关性不传递.

## 路线全景 (Issue #71 完成 Phase A, 进入 Phase B)

| Issue | 方案 | test R@10 | vs baseline | ratio |
|-------|------|-----------|-------------|-------|
| baseline | T5 only | 0.1024 | — | 1.237 |
| #138 v74 | HAB frozen + WD + dropout | 0.1063 | +0.0039 | 1.234 |
| #141 v77 | Stage1 + HAB | 0.1080 | +0.0056 | 1.215 |
| #68 v78 | DECOR + 抗 trap | 0.1092 | +0.0068 | 1.225 |
| **#71 Phase A** | **v78 + Delayed HAB** | **0.1012** | **-0.0012** | **1.257 ← NO-GO** |

## 教训

**DECOR + HAB 叠加不可取**, 即使分时引入 (warmup) 也无法解决优化 landscape 冲突. 两种辅助范式同时作用于 T5 → valid/test 平衡破坏 → ratio 1.257 恶化.

**Why:** DECOR 通过 embedding 动态调节 (alpha gate + bos_diversity + attn_entropy), HAB 通过 encoder attention bias 静态调节. 两者作用点不同, 但都修改 T5 训练轨迹. 当同时存在时, T5 无法找到兼容两者的解 → 过拟合 (valid 好 test 差).

**How to apply:** Phase B (曲率差分 HAB) **不再叠加 DECOR**, 应作为纯曲率路线的下一阶段改进 (基于 v77 = Stage1+HAB). 若用户要求 DECOR+HAB 路线, 应回到 v78 + 抗 trap 优化, 不再尝试叠加 HAB.

## Phase B 下一轮 (Curvature-Differential HAB)

不再叠加 DECOR, 改为 v77 + 曲率差分 HAB:
- ΔD_l = ||D_hyp||_l - ||D_flat||_l (纯曲率贡献, 排除码字距离的尺度影响)
- B_l = -λ_l * ΔD_l
- 保留 v77 Stage1 per-item radius + Stage2 kappa_sync SID
- 新增 ΔD_l 计算 (Stage2 ckpt 需加载 flat 版本码字对比)
- 预期: 曲率信号比完整 D_hyp 更纯粹, T5 学到真正的曲率几何, ratio 维持 ~1.21

## 产物路径

- 代码: commit 9e6d27a (push done)
- 训练: taskA/_history/issue71_v79_decor_delayed_hab/HG_Rec_best.pth (ep35)
- 评估: taskA/_history/issue71_v79_decor_delayed_hab/eval_test/eval_test.json
- verdict: verdicts/issue71_phase_a_v79_decor_delayed_hab_nogo.md (本文档)
- SID: 06af0fed (Stage2 hyp_v2_capmatch_1000ep, 复用)

## 时间线

- 2026-08-07 12:35 — DDP 4 卡 v79 启动 (ep 0)
- 2026-08-07 12:35:35 — config 输出, warmup_T0=200/warmup_Tw=800 注入
- 2026-08-07 12:37 — ep5 first eval valid 0.1084
- 2026-08-07 12:48 — ep30 valid 0.1269
- 2026-08-07 12:50 — ep35 best valid 0.1272 (BG 触发)
- 2026-08-07 12:54 — ep45 valid 0.1267, ES 2/10
- 2026-08-07 ~13:30 — 用户指示用 best ckpt 跑 test (训练继续跑不停)
- 2026-08-07 ~13:32 — test eval 启动 (ep35 best ckpt)
- 2026-08-07 ~13:35 — test R@10 = 0.1012 (NO-GO)