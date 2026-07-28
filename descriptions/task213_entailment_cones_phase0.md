# Task #213 — Entailment Cones Phase 0 (方向一, Ganea 2018)

**日期**: 2026-07-26
**父级**: 用户 2026-07-26 三方向提议 → 方向二 Task #212 ❌ NO-HOPE 收线 → 方向一 (Entailment Cones) 主攻
**优先级**: 🔴 最高 (用户明确: "第一个最有希望, 因为它绕开了那个死结的根源")

---

## 1. 背景与动机

### 1.1 死结 (argmin death loop)

之前 4 个调参方向 (Task #199/200/209/211) 的共同根因:
- 都让"几何"参与 argmin 分配 (即哪个码字最近)
- 但 HG-Rec baseline 训出的码本, 几何判据 (Poincaré) 跟欧式判据排序 99% 一致 (Task #212 验证)
- 所以"argmin 阶段加几何" = "加了个常数缩放" = 无信号
- 再叠 path_reg / norm_target / scale_norm / dual codebook 都改变不了 argmin 排序 (因为底层 argmin 是几何无关的)

### 1.2 方向一 (Entailment Cones) 如何绕开

Entailment Cones (Ganea et al. 2018) 是一个跟 argmin 完全独立的层级概念:
- 给每个 entity 一个 Poincaré 球内的 apex (顶点) + opening angle (开口角度)
- 锥定义一个包容区域: z 在锥内 = z 被该 entity "蕴含" (entail)
- opening angle 跟 ρ 半径反向: ρ 小 → 锥宽 (general concept); ρ 大 → 锥窄 (specific concept)
- 分配规则: 多个锥同时包含 z → 选最特异的 (largest ρ = narrowest cone) — 这是层级分配, 不是 argmin
- 跨层 transitivity: 粗层选 general → 细层选 specific, 自然适配 RQ-VAE 粗到细结构

关键: 锥分配 = "包含" 关系, 不是 "距离最近" 关系. 它利用几何的"半径信息", 而 HG-Rec baseline 的 λ_κ ≈ 2 (无信号) 主要是因为 argmin 没用上半径.

---

## 2. Phase 0 目标 (1-2 天, CPU only)

按用户 2026-07-26 "半天判据检查" 模式:
- 不训练, 不动 GPU
- 在 baseline 码本上, 模拟锥分配 看几何是否真的有信号
- 通过则进 Phase 1 (改造 train_hrqvae.py)

### 2.1 锥分配公式 (Ganea 2018, 简化版)

对 Poincaré 球内 apex p (||p||_E < 1), 锥 opening 半角 α:

z ∈ Cone(p, α) iff:
- z 在 half-aperture angle 内 (从 p 看 z 的角度 ≤ α)
- 等价于 (Ganea eq. 5):
  ⟨u, -p⟩ ≥ ((1 + ||p||²) cos α - sqrt((1 + ||p||²)² cos² α - (1 - ||p||²)²)) / ||p||
  其中 u = (z - p) / (1 - 2p·z + ||p||²||z||²) 是 p→z 的切空间单位向量

更简单的近似 (在小 ||p|| 时):
- cos α(z, p) = ((1 + ||p||² - ||z||² + 2 p·z) ||z||) / (2 ||p|| · (1 - p·z)) (Ganea eq. 6)
- z ∈ cone iff cos α(z, p) ≥ cos(α_apex)

### 2.2 三层锥设计 (RQ-VAE 粗到细)

| Layer | K (codes) | ρ 半角 (opening half-angle) | 概念 |
|---|---|---|---|
| L0 | 64 | α_0 大 (宽锥) | 最 general, 顶层分类 (e.g., 乐器 vs 配件) |
| L1 | 128 | α_1 中 | 中等特异 (e.g., 弦乐器 vs 管乐器) |
| L2 | 256 | α_2 小 (窄锥) | 最特异 (e.g., 小提琴 vs 中提琴) |

α 设计原则: α_L0 > α_L1 > α_L2 (粗锥宽, 细锥窄).

### 2.3 Phase 0 判据 (3 个指标)

| 指标 | 通过阈值 | 测量方法 |
|---|---|---|
| S1: 锥判据 vs 欧式 argmin 一致率 | < 60% (说明锥有独立信息) | baseline 码本, 模拟锥分配, 跟欧式 argmin 比 |
| S2: 跨层 transitivity 保持率 | > 80% (L0 选 general 后, L1/L2 选 specific 的概率) | L0 选 apex 后, 看 L1/L2 候选是否落在 L0 锥内 |
| S3: 锥分配 SID 唯一性 | > 95% (像 Sinkhorn 一样制造唯一性) | 模拟三阶段锥分配, 输出 SID, 算 unique 率 |

通过规则:
- S1 一致率 < 60% ✅ (锥有独立信息, 跟欧式 argmin 不同) — 核心
- S2 transitivity > 80% ✅ (跨层结构合理)
- S3 unique > 95% ✅ (不会跟现有 Sinkhorn 退化)

---

## 3. 实施步骤 (半天)

1. 0.5h: 重读 Ganea 2018 Section 3 (cone definition) + Section 4 (learning), 确认公式
2. 1h: 写 scripts/task213_entailment_cones_phase0.py:
   - 加载 baseline ckpt
   - 编码 items → latent (9922, 32)
   - 给每个码字随机一个 apex (用码字自身 norm × 0.5 作为 ρ)
   - 模拟锥分配 (三层, K=64/128/256)
   - 计算 3 个判据指标
3. 1h: 输出 JSON + 控制台打印决策 (S1/S2/S3)
4. 0.5h: 写 verdicts/task213_entailment_cones_phase0_result.md

### 3.1 关键参数 (R11.3 自主决策)

- α_0, α_1, α_2: α_L = arctan(K_L / scale_factor). 默认 scale_factor = 10 (码字数越多, 锥越宽)
- apex norm: ||p||_E = 0.5 (中等内推, 留出锥开口空间)
- 三层 ρ 衰减: ρ_L0 = 1.0, ρ_L1 = 1.5, ρ_L2 = 2.0 (粗层靠外, 细层靠内)

---

## 4. 备选实验 (Phase 0 通过后, Phase 1 计划)

如果 S1/S2/S3 全过 → 进入 Phase 1 (3-4 天, GPU):
1. 改造 train_hrqvae.py: 在 VQ 阶段加 cone loss (鼓励码字 apex 远离 origin)
2. 新增 loss: L_cone = mean(max(0, α_target - cos_angle(z, p_q))) (让选中码字的 α 跟 z 的实际 cos angle 匹配)
3. 改造 Stage 2 SID 推断: 用 cone-based assignment 替 argmin
4. 4-stage 闭环: Stage 1 train → Stage 2 SID → Stage 3 T5-mini → Stage 4 eval

GO 阈值: R@10 > 0.1020 (vs baseline); NDCG@10 > 0.0755

---

## 5. 关键风险 (R11.3 提前明示)

- Risk 1: 锥分配可能比 argmin 慢 10x (要算 cos angle for all K*N pairs). Phase 1 实现要 batched
- Risk 2: 锥 α 难调 (跟码字数 K 强耦合). Phase 1 需要 α sweep {0.5, 1.0, 1.5} × K ∈ {64, 128, 256}
- Risk 3: 锥分配可能跟 Sinkhorn 不兼容 (Sinkhorn 强求唯一性, 锥可能多个 apex 同时包含 z). Phase 1 需要 "选最 specific" 决策 + tie-break 规则

---

## 6. 链接

- 理论参考: Ganea et al. 2018 "Hyperbolic Entailment Cones for Learning Hierarchical Embeddings" (ICML)
- HG-Rec baseline ckpt: products/task84/ckpt/Instruments/Jul-23-2026_20-08-06_beta_1.000_codebook_[64,128,256]_sk_0.000/best_loss_model.pth
- 数据集: HG-Rec/dataset/Instruments/item_emb.parquet (9922 × 768)
- 上游代码: HG-Rec/model/utils.py (poincare_distance, expmap0), HG-Rec/model/hrqvae.py (HRQVAE class)

---

(本文档为 Phase 0 设计, 不锁定参数; 实际 α/ρ 数值在脚本里 sweep 后再固化.)
