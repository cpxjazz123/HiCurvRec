# Task #215 — Paper §4 完整叙事 (HG-Rec 包装失效 4 证据链)

**日期**: 2026-07-26
**父级**: Task #199/211/212/213 累积证据收尾
**优先级**: 🔴 最高 (确定性输出, 0 卡, paper contribution)

---

## 1. 背景

HG-Rec paper (arXiv) 报告 R@10=0.1315 (Musical_Instruments 5-core 后 9922 items). 我们复现得到 R@10=0.1020 (Δ -22.4%).

整个项目做了 6 个变体方向 (Task #199/200/208/209/211/212/213), **全部 NO-GO**, 共同根因是 "HG-Rec 包装失效":
- HG-Rec 训完后码字 norm 全 ≈ 1.0 (boundary), λ_κ ≈ 100
- 任何需要"真双曲几何" 的方案 (可学习 κ / 锥 / path_reg / 钉半径) 都因 baseline 几何坍缩而失效

**Paper §4 应明确**: HG-Rec paper 报告的提升**不是几何带来的**, 是"包装 + 框架" 带来的 (Sinkhorn + 4-digit dedup + T5 容量 + 训练稳定).

---

## 2. 目标 (半天, 0 卡)

整合 Task #199/211/212/213 4 个证据, 写出 paper §4 完整叙事, 包括:

### 2.1 §4.1 HG-Rec 包装失效的 4 证据

| 证据 | 来源 | 量化 |
|---|---|---|
| E1: λ_κ ≈ 2 数值无效 | Task #199 Stage 1 log | exp(θ) θ 全程未动, κ 值不变 |
| E2: argmin 一致率 99% | Task #212 9 格子 sweep | L0/L1/L2 一致率 98.82/99.21/99.56% |
| E3: 锥分配 NO-HOPE | Task #213 4 组合 sweep | S2/S3 全 FAIL, 几何坍缩到 boundary |
| E4: 钉半径 NO-GO | Task #211 C1/C2/C3 | R@10=0.0816 (-20% vs baseline 0.1020) |

### 2.2 §4.2 4 格失败模式 (跟 2×2 paper skeleton 合并)

| 格子 | 代表 | 状态 | 失败原因 |
|---|---|---|---|
| 左上 (高维, 不钉) | HG-Rec #84 | ✅ baseline | 包装失效 (λ_κ≈2) |
| 右上 (高维, 钉) | #209 A3 | ❌ NO-GO | 距离饱和 (dyn 1.27) |
| 左下 (低维, 不钉) | #210 B1 | ⚠️ 持平 | ‖x‖_E 自由漂移 |
| 右下 (低维, 钉) | #211 C1/C2/C3 | ❌ NO-GO | 码字坍缩 (util 23%) |

### 2.3 §4.3 为什么"几何对 RQ-VAE 推荐" 是无效组件

- RQ-VAE 的核心是 argmin 分配 + 残差量化, 几何 (双曲 vs 欧式) 只影响距离公式
- 当码字 norm 全 ≈ 1.0, λ_κ 接近常数, Poincaré 距离 ≈ 欧式距离 × 常数, 单调等价
- "几何" 在 HG-Rec baseline 上是"无信号装饰", 跟 Sinkhorn + 4-digit dedup 框架相比贡献为 0
- 这解释了为什么 HG-Rec paper vs 复现的 Δ -22.4% 主要来自数据集/评估协议, 而非几何

### 2.4 §6 Future Work (替换原版)

- ❌ exp(θ) 可学习 κ (Task #199/201) — 训练不动 θ
- ❌ 双码本解耦 (Task #200/208) — 几何仍坍缩
- ❌ path regularization (Task #209) — 不改变 argmin 排序
- ❌ 低维双曲 + 钉半径 (Task #211) — 反而坍缩更严重
- ❌ Two-stage decision (Task #212) — 几何在 top-k 无区分力
- ❌ Entailment Cones (Task #213) — 锥 opening 利用不了 norm
- **唯一可能突破 (投机)**: Latent radius live (Task #214 Phase 0/1) — 让 ρ_i item-adaptive

---

## 3. 实施步骤 (半天, 0 卡)

1. 0.5h: 整理 Task #199/211/212/213 关键 log + 数值, 写成 evidence table
2. 1h: 写 paper §4 完整草稿 (paper.md 增量追加 §4)
3. 0.5h: 把 verdicts/task30_2x2_design_space_paper_skeleton.md 的 §6 future work 替换成新版的"几何路线 6 个方向全 NO-GO"
4. 1h: 跨引用 / 校对 / commit

---

## 4. 交付物

- papers/paper.md §4 完整段落 (新增)
- verdicts/task30_2x2_design_space_paper_skeleton.md §6 替换版
- verdicts/task215_paper_section4_complete.md (verdict, 汇总所有 evidence)

---

## 5. 链接

- 4 证据来源:
  - verdicts/task199_stage1_exp_theta_result.md (E1)
  - verdicts/task211_low_dim_pinned_radius_arch_infeasible.md (E4)
  - verdicts/task212_two_stage_criterion_result.md (E2)
  - verdicts/task213_entailment_cones_phase0_result.md (E3)
- 2×2 skeleton: verdicts/task30_2x2_design_space_paper_skeleton.md
- Paper 主文件: papers/paper.md (增量追加 §4)

---

(本文档为 paper 收尾, 不涉及新实验, 0 卡完成.)
