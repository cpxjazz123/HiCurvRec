# Task #214 — Latent Radius Live Phase 0 (方向三, 最后赌注)

**日期**: 2026-07-26
**父级**: 方向二 (#212) ❌ + 方向一 (#213) ❌ → 方向三 (最后候选)
**优先级**: 🟡 投机, 用户原话 "Latent radius as live variable"

---

## 1. 背景与动机

### 1.1 前两方向失败根因

- **方向二 (Two-stage decision, Task #212)**: 双曲几何在 baseline top-k 内跟欧式 99% 一致, 几何在 top-k 无区分力
- **方向一 (Entailment Cones, Task #213)**: HG-Rec baseline 码字 norm ≈ 1.0 (boundary), 锥 opening 利用不了 norm 差异
- **共同根因**: HG-Rec baseline 训完后码字几何坍缩 (Task #199/205/211 多证据)

### 1.2 方向三 idea (Latent radius live variable)

用户 2026-07-26 原话: "Latent radius as live variable — item-adaptive hierarchy strength"

**核心 idea**:
- 现有 baseline: 每个 item 被分到固定 ρ (码字 norm 固定)
- 方向三: 让 radius ρ 跟 item 相关 — 不同 item 的"层级强度" 不同
- 例如: 流行 item ρ 小 (general, 跨类比); 长尾 item ρ 大 (specific, 单类比)
- 实现: 给 encoder 加 radius head, 输出 ρ_i ∈ [0.3, 1.0]; 用 ρ_i 调 vq 分配 + Stage 2 SID 推断

**为什么可能绕开坍缩**:
- 不再"一个码字对应一个 ρ", 而是"每个 item 有自己的 ρ"
- 码字 norm 仍 ≈ 1.0, 但 item latent 的 effective ρ 由 head 调节
- 锥/argmin 公式里的"item 维度" 提供新信号

---

## 2. Phase 0 目标 (半天, CPU only)

跟 Task #212/213 同模式: 不训练, 不占卡, 在 baseline ckpt + data 上测信号.

### 2.1 三个判据

| 指标 | 通过阈值 | 测量方法 |
|---|---|---|
| **R1**: item-level radius 离散度 | std(ρ_i) > 0.15 (有差异化) | 用 baseline encoder + 临时 radius head (e.g., σ(W·z+b)) |
| **R2**: ρ_i 跟 item 流行度相关 | Spearman > 0.10 (positive: 流行→general) | Musical_Instruments item popularity (interaction count) |
| **R3**: ρ_i 跟 item embedding norm 相关 | Spearman > 0.05 (positive: 越远→specific) | baseline encoder 输出 norm |

**通过规则**:
- R1 std > 0.15 ✅ (item 间 ρ 有差异, 不是 constant)
- R2 OR R3 Spearman > 0.10 ✅ (ρ 跟 item 语义/统计相关, 不是 noise)

---

## 3. 实施步骤 (半天)

1. 0.5h: 写 scripts/task214_radius_head_phase0.py
   - 加载 baseline ckpt
   - 给 encoder 加临时 radius head: ρ = sigmoid(W·z + b), W ∈ R^(32→1)
   - 编码所有 items → (ρ_i, z_i)
2. 1h: 计算 3 个判据
   - R1: ρ_i 的 std / mean (离散度)
   - R2: 跟 Musical_Instruments interaction_count 相关 (需要读 .csv 数据)
   - R3: 跟 ‖z_i‖ 相关 (baseline encoder 输出 norm)
3. 0.5h: 输出 JSON + verdict

### 3.1 关键参数 (R11.3 自主决策)

- radius head: W ∈ R^(32→1), 初始化小 (W = 0.01·randn, b = 0), 让 ρ 初始在 sigmoid(0) = 0.5 附近
- 范围约束: ρ ∈ [0.3, 1.0] (跟 baseline norm 范围一致)
- 不训练 radius head (Phase 0 用 random init), 只测随机参数的 ρ 分布是否已经有差异

---

## 4. Phase 1 计划 (Phase 0 通过后, 5-7 天 GPU)

如果 R1+R2/R3 通过 → 进入 Phase 1:
1. 改造 train_hrqvae.py:
   - encoder 加 radius head (可学习 ρ_i)
   - VQ 阶段: 用 ρ_i 调距离权重 (e.g., d_eff = d_euc + λ · (1 - ρ_i))
2. Stage 2 SID 推断:
   - 每个 item 用自己的 ρ_i, 算 effective 距离
   - 三层 SID 各自 ρ-aware
3. Stage 3 T5-mini 训练 (跟 baseline 一样)
4. Stage 4 eval: R@10 > 0.1020 GO

---

## 5. 风险 (R11.3 提前明示)

- **Risk 1**: Phase 0 即使通过, Phase 1 训练可能仍因 baseline 几何坍缩失败 (跟前两方向同根因)
- **Risk 2**: radius head 是新组件, 需要额外调参 (lr, weight_decay, init)
- **Risk 3**: ρ_i 跟 T5 输入对齐 (Stage 3 用 SID, 不用 ρ), 所以 radius 只能影响 Stage 1-2, 不直接影响 T5 学到的东西

---

## 6. 链接

- Task #212 verdict (方向二 NO-HOPE): verdicts/task212_two_stage_criterion_result.md
- Task #213 verdict (方向一 NO-HOPE): verdicts/task213_entailment_cones_phase0_result.md
- HG-Rec baseline ckpt: products/task84/ckpt/Instruments/Jul-23-2026_20-08-06_beta_1.000_codebook_[64,128,256]_sk_0.000/best_loss_model.pth
- 数据集: HG-Rec/dataset/Instruments/item_emb.parquet (9922 × 768)

---

(本文档为 Phase 0 设计, 不锁定参数.)
