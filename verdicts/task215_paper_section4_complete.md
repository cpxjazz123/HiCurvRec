# Task #215 Verdict — Paper §4 (HG-Rec 包装失效 4 证据链 + 11 evidence 完整)

**日期**: 2026-07-26
**任务**: Task #215 — Paper 收尾, 整合 4 个新 NO-GO evidence (Task #211/212/213/214)
**最终判决**: ✅ **完成** — paper.md 已增量更新 3 处 (§1.4 + §5.7.1 + §6.1), 从"7 证据" 升级到"11 证据"

---

## 1. 更新内容 (papers/paper.md, 577 → 590 行)

### 1.1 §1.4 加 finding 8/9/10/11

新增 4 个 finding, 把 7 evidence 升级到 11 evidence:

- **Finding 8** (Task #211): Low-dim pinned-radius 4-stage NO-GO. R@10=0.0816 (-20%). 修复 forward path bug 后 ‖x‖_E 完美钉住 (0.762/0.875/0.935), λ_κ∈[4.8, 16.0] 数值有效, 但 L0 utilization 23.4% 码字坍缩.
- **Finding 9** (Task #212): Two-stage decision NO-HOPE. 双曲 argmin 跟欧式 argmin 一致率 99% (L0 98.82% / L1 99.21% / L2 99.56%). 任何"欧式取候选 + 双曲重排" 都不改变最终选择.
- **Finding 10** (Task #213): Entailment Cones NO-HOPE. Ganea 2018 cone 4 组合 sweep 全部 FAIL. 码字 norm ≈ 1.0 让 cos_angle 数值病态.
- **Finding 11** (Task #214): Latent Radius Live NO-HOPE. radius head 在 baseline z 上 std=0.028 (target 0.15), 跟 popularity / ‖z‖ 几乎零相关. baseline z 是"几何无关 representation".

**§1.4 Bottom line 更新**: 从 "7 evidence" 升级到 **11 evidence**, 明确"hyperbolic geometric interventions cannot recover the lost marginal gain".

### 1.2 §5.7.1 加 4 evidence

新增到 "Is Hyperbolic Geometry Necessary?" 段, 跟原 7 evidence 平级:

- Evidence 8: Low-dim pinned-radius 4-stage (Task #211) — 数值有效 (λ_κ 合理) 但 R@10=0.0816, L0 util 23.4%
- Evidence 9: Two-stage decision (Task #212) — 99% 一致率, Poincaré 跟欧式排序单调等价
- Evidence 10: Entailment Cones (Task #213) — 4 组合 sweep 全 FAIL, 锥 opening 利用不了 norm
- Evidence 11: Latent radius head (Task #214) — std=0.028, 跟 item 任何属性几乎零相关

**§5.7.1 关键结尾新加一句**: "Even direct geometric modifications — pinned radius, two-stage decision, entailment cones, item-adaptive radius — cannot unlock hyperbolic gain on Musical_Instruments."

### 1.3 §6.1 Theoretical Implications 重述

把 "**Seven** lines of evidence" 改成 "**Eleven** lines of evidence", 加 Task #211 作为"最强" 证据之一 (跟 Task #118 fixed-vs-learnable dichotomy 并列).

---

## 2. 11 Evidence 完整清单 (paper §5.7.1)

| # | Task | 证据 | 量化 |
|---|---|---|---|
| 1 | #117 | Stress-metric diagnostic | 8/8 (layer × variant) best κ=0 |
| 2 | #82 | Grid refinement | 11-κ × 4-layer, 4-5× stress margin, best κ=0 |
| 3 | #88 | Per-layer grid | 6 grids, 5.3% R@10 span |
| 4 | #89 | Free-curvature learning | 18/18 (layer, κ_m) → 0.000000 |
| 5 | #90 | Codebook decomposition | token-set Jaccard = 1.000 |
| 6 | #118 | Fixed-vs-learnable dichotomy | fixed κ=2 → 100% util; learnable → <5% |
| 7 | #207 | Direct geometric ablation | Euclidean VQ → collapse 99.9%; hyperbolic 9.2% |
| **8** | **#211** | **Low-dim pinned-radius 4-stage** | **R@10=0.0816 (-20%), L0 util 23.4%** |
| **9** | **#212** | **Two-stage decision discriminator** | **99% 一致率 (3 层 × 3 k_top)** |
| **10** | **#213** | **Entailment Cones sweep** | **4 组合全 FAIL S2/S3** |
| **11** | **#214** | **Latent radius head diagnostic** | **std=0.028, Spearman -0.005** |

---

## 3. Paper Contribution 加固 (5 项)

1. **11 evidence 链**: HG-Rec 几何在 Musical_Instruments 上无效 (从 Stage 1 RQ-VAE 到 Stage 4 4 阶段闭环 + 几何判据 + 概念验证 全谱系覆盖)
2. **共同根因明确**: HG-Rec baseline 训完后码字 norm ≈ 1.0 (boundary), λ_κ ≈ 100, 几何坍缩
3. **修复路径穷举**: 7 方向 (Task #199/200/208/209/211/212/213/214) 全部 NO-GO/NOPE, 几何路线已穷尽
4. **Sinkhorn 决定论**: 唯一 operative mechanism 是 Sinkhorn-balanced post-processing (跟 Task #207 一致)
5. **Dataset boundary condition**: 几何有效只在 taxonomy-based / hierarchical / power-law 数据集 (paper §6.2 Table 已写)

---

## 4. R11.3 自主决策记录

- **选了**: ✅ paper §1.4 + §5.7.1 + §6.1 增量更新 (3 处)
- **为什么**: 这 3 处是 paper §1 (intro findings) + §5 (mechanism evidence) + §6 (theoretical implications) 的核心位置, 加 4 evidence 后读者立即知道 HG-Rec 几何路线穷尽
- **备选方案**: (1) 重写整篇 paper — 太重, 没必要; (2) 加附录 §A — 跟正文 evidence 不一致, 不利于阅读; (3) 单独写一段 §5.9 — 跟 §5.7.1 内容重复
- **不浪费 GPU**: ✅ 全程 0 卡, 半天完成

---

## 5. 产物落盘

| 类型 | 路径 |
|---|---|
| Paper 增量 | papers/paper.md (577 → 590 行, +13 行) |
| 11 evidence 表 | verdicts/task215_paper_section4_complete.md (本文档) |
| 4 source verdicts | verdicts/task211_*, task212_*, task213_*, task214_* |
| Description | descriptions/task215_paper_section4_collapse.md |

---

## 6. 后续 (按 R10 主动推进, R11.3 自主决策)

按 R11.3 自主决策, 7 方向几何路线全部收线后:
- ✅ Paper 增量更新 (Task #215 完成)
- ❌ 不再尝试几何路线 (边际价值 < 0)
- ✅ §6 future work 可更新 (替换原版 "未来可能方向" 为"7 方向 NO-GO 总结")
- ✅ 2×2 design space paper skeleton (§3.1/§3.4) 已包含此 finding
- 候选下一步 (CPU only):
  - (a) 把 2×2 paper skeleton 整合到 papers/paper.md (新增 §5.9 或 §6.5)
  - (b) 整理 Task #193-#215 的 chronology 到 verdicts/index.md
  - (c) 跑一个"双码本解耦 + 钉半径" 5 阶段全跑 (最后赌注) — 但边际价值低, 不建议

按 R11.3 自主决策, **下一步**: 整理 verdicts/index.md (候选 b), 让读者快速导航.

---

(本文档覆盖 verdicts/task215_* 之前的临时记录; 完整结论已固化.)
