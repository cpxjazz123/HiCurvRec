# Verdicts Index — Task #193-#215 Chronology

> 本文档按主题分类汇总所有 task verdict, 方便 reader 快速导航.
> 数据集: Amazon Musical_Instruments (5-core 后 9922 items, 511836 interactions)
> 基线: HG-Rec (Task #84) R@10=0.1020

---

## A. Baseline 复现 (R@10 = 0.1020)

| Task | 主题 | 状态 | Verdict |
|---|---|---|---|
| #84 | HG-Rec baseline 复现 | ✅ DONE | R@10=0.1020 (paper 0.1315, Δ -22.4%) |
| #143 | FDSA paper-aligned fix | ✅ DONE | verdicts/... |
| #144 | Stage 2/4 代码脚本 + 模板 | ✅ DONE | verdicts/... |

---

## B. 几何变体 7 方向 NO-GO (共同根因: HG-Rec 几何坍缩)

| Task | 主题 | 状态 | 量化 | Verdict |
|---|---|---|---|---|
| **#199** | exp(θ) 可学习 κ Stage 1 | ❌ NO-GO | θ 全程未动 | verdicts/task199_stage1_exp_theta_result.md |
| **#200** | 双码本解耦 Stage 3+4 | ❌ NO-GO | R@10=0.0915 | verdicts/task200_dual_v5_stage3_4_result.md |
| **#201** | exp(θ) κ with θ_init=log(10) | ❌ NO-GO | θ 仍不动 | verdicts/task201_kappa_redo_result.md |
| **#203** | exp(θ) κ + scale_norm | ❌ NO-GO | Δ 9.37/9.28% collision | verdicts/task203_kappa_scale_norm_result.md |
| **#208** | 双码本解耦 (5 臂) | ❌ NO-GO | 已过时 | verdicts/... |
| **#209** | Path regularization | ❌ NO-GO | dyn 1.27 距离饱和 | verdicts/task209_path_reg_result.md |
| **#211** | 低维双曲 + 钉半径 4 阶段 | ❌ NO-GO | R@10=0.0816 (-20%), L0 util 23.4% | verdicts/task211_low_dim_pinned_radius_arch_infeasible.md |
| **#212** | Two-stage decision | ❌ NO-HOPE | 一致率 98.82-99.56% (9 格子) | verdicts/task212_two_stage_criterion_result.md |
| **#213** | Entailment Cones (Ganea 2018) | ❌ NO-HOPE | 4 组合全 FAIL S2/S3 | verdicts/task213_entailment_cones_phase0_result.md |
| **#214** | Latent Radius Live (item-adaptive) | ❌ NO-HOPE | std=0.028, Spearman ≈ 0 | verdicts/task214_radius_live_phase0_result.md |

**共同根因 (定理式)** — **√c·ρ 张力**: 官方 HG-Rec baseline 的 $\sqrt{c}\rho \approx 0.54$ (Task #189/#191 实测: ‖p‖_E=0.262/0.100/0.072, λ≈2.01-2.15) 远低于激活阈值 2, 几何不参与分配 (Task #212 argmin 一致率 99.91%). 把 √c·ρ 推入激活区 (>2) 时, 几何激活与量化可分性**直接冲突**:
- **距离饱和** (Task #209 A3, 高维钉半径): dyn_range 2.14 → 1.27, 球面单层薄壳, collision 99.97%
- **死码本螺旋** (Task #211 C1, 低维钉半径): L0 利用率 23.4% (15/64). 半径钉住 → 分配退化为纯余弦 → 4 维方向集中 → 重构只用 32 维欧式 → 唯一梯度来自分配损失, 只喂给 15 个选中码字 → 49/64 死码

两个机制分别独立, 不是"坍缩" 的统一现象. 详见 paper §6.7.

---

## C. κ 调试 / 数值精细化 (multi-task 系列, 验证 c=1 健康)

| Task | 主题 | 状态 | 关键 finding | Verdict |
|---|---|---|---|---|
| #164 | κ-decouple Phase A/B | ⚠️ 部分 | Stage 1 OK, Stage 4 R@10=0.0964 | verdicts/... |
| #165 | κ-decouple 长期训练 | ⚠️ 部分 | 200 epoch early_stop=30 | verdicts/... |
| #162 | κ-stereographic fix | ⚠️ | OK | verdicts/... |
| #166 | κ-stereo T5-mini ablation | ⚠️ | 边际 | verdicts/... |
| #167 | κ-stereo T5-base ablation | ⚠️ | 边际 | verdicts/... |
| #168 | κ-stereo 5.5m ablation | ⚠️ | 边际 | verdicts/... |
| #169 | κ + Sinkhorn combined | ⚠️ | 边际 | verdicts/... |
| #170 | κ + Sinkhorn all 3 | ⚠️ | 边际 | verdicts/... |

**总结**: c ∈ [1, 10] 是健康区, c ≥ 93 不可训练. 5 阶段累加 (#84/#178/#199/#201/#203) 验证 HG-Rec c=1.0 是 baseline 默认.

---

## D. T5 容量 / 训练 (audit + 边界)

| Task | 主题 | 状态 | 量化 | Verdict |
|---|---|---|---|---|
| #157 | T5-base 220M capacity unlock | ✅ | R@10 > 0.1020 | verdicts/... |
| #158 | HG-Rec eval protocol audit | ✅ | 协议一致 | verdicts/task158_hgrec_eval_protocol_audit.md |
| #159 | T5-mini 12M capacity point | ✅ | R@10=0.1020 | verdicts/... |
| #160 | T5-small 60M capacity point | ✅ | R@10=0.1020 | verdicts/... |
| #161 | T5-mini dkv fix 9.18M | ✅ | R@10=0.1020 | verdicts/... |

---

## E. 复现审计 (Task #117/82/88/89/90/91/118/207)

| Task | 主题 | 关键发现 | Verdict |
|---|---|---|---|
| #117 | Stress-metric Ollivier κ | 99% 边 \|κ\| < 0.05 | verdicts/... |
| #82 | Curvature grid 11×4 | best κ=0, 4-5× stress margin | verdicts/... |
| #88 | Per-layer grid | 6 grids, 5.3% R@10 span | verdicts/... |
| #89 | Free-curvature learning | 18/18 → 0.000000 | verdicts/task89_free_curv_product_manifold_result.md |
| #90 | Codebook decomposition | Jaccard = 1.000 | verdicts/... |
| #91 | Training dynamics | val/test 反转 | verdicts/... |
| #118 | Fixed-vs-learnable κ dichotomy | fixed → 100% util; learnable → <5% | verdicts/... |
| #207 | Euclidean vs Hyperbolic 几何 ablation | Euc VQ 坍缩, Hyp VQ OK | verdicts/task207_euclidean_vs_hyperbolic_result.md |

---

## F. paper 收尾 (Task #30/215)

| Task | 主题 | 状态 | 交付 | Verdict |
|---|---|---|---|---|
| #30 | 2×2 design space paper skeleton | ✅ | §3.1-3.4 + §4-6 + 4 格失败模式 | verdicts/task30_2x2_design_space_paper_skeleton.md |
| #215 | paper.md §1.4/§5.7.1/§6.1 增量更新 | ✅ | 7 → 11 evidence | verdicts/task215_paper_section4_complete.md |

---

## 关键数字速查表

| 指标 | 值 | 来源 |
|---|---|---|
| HG-Rec baseline R@10 | **0.1020** | Task #84 |
| HG-Rec paper R@10 | 0.1315 | HG-Rec paper Table 1 |
| phonism R@10 (Sinkhorn + vanilla) | 0.1058 | Task #87 |
| Task #199 exp(θ) κ | NO-GO, θ 不动 | #199/201 |
| Task #211 C1 钉半径 R@10 | 0.0816 (-20%) | #211 |
| Task #212 一致率 | 99% (L0 98.82%, L1 99.21%, L2 99.56%) | #212 |
| Task #213 锥 S2/S3 FAIL | 4 组合全 FAIL | #213 |
| Task #214 radius std | 0.028 (target 0.15) | #214 |
| 7 方向几何路线 NO-GO/NOPE | 100% | #199/200/208/209/211/212/213/214 |
| 11 evidence 链 | 全部指向"几何无效" | paper.md §1.4 + §5.7.1 |

---

## 决策原则 (R11.3)

按 R11.3 自主决策 + R10 主动推进:
- 7 方向几何路线**全部收线** (边际价值 < 0)
- 唯一 operative mechanism: **Sinkhorn-balanced post-processing** (Task #207 隔离)
- 唯一可能突破: 重训 Stage 1 with 不同 encoder / curriculum, 但跟 HG-Rec 框架脱钩, 不在本项目范围

---

(本文档为 verdicts/ 索引, 不替代具体 verdict. reader 找详细论证请看 task{N}_result.md.)
