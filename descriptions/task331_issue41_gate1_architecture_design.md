# Task #331 / Issue #41 Gate 1 — Architecture Design (no training)

**日期**: 2026-07-30 14:15
**状态**: 🔄 DESIGN PHASE (Gate 0 PASS 后, 进入 Gate 1 设计阶段)
**前置**: Gate 0 verdict `verdicts/task331_issue41_gate0_h_mds_input_space.md` ✅ PASS
**约束**: per Issue #41 §实验设计: **本 Gate 只交付设计方案 + 可行性评估, 不跑任何训练**

---

## 1. Gate 1 设计目标

按 Issue #41 §实验设计 Gate 1 原文:
> "若 Gate 0 确认输入空间存在显著、可信的最优曲率估计，下一步是设计'如何把这个曲率信号从输入空间搬运到实际的 codeword 分配决策里'——这是一个新的架构问题，不是简单地把这个 κ 值塞进现有 quantizer（那样等于重复 Task #80 已经否证的路径）。本 Gate 只交付设计方案 + 可行性评估，不跑任何训练。"

**核心挑战**: Task #80 已否证"把 κ 塞进 quantizer cost matrix"路径. Gate 1 必须找**新的架构层搬运机制**.

---

## 2. 4 个候选 Gate 1 方向 (R11.5 待 owner 决策)

### 2.1 方向 A: 预量化双曲感知 (Pre-quantization Hyperbolic Awareness)

**机制**: 在 RQ-VAE encoder 之前, 对输入 768d embedding 做"双曲度增强"预处理:
- 用 Ollivier κ=-0.74 估计 (Gate 0 H2 结果) 初始化 Poincaré map
- 对每个 768d 输入 x: compute x_hyp = poincare_map(x, κ=-0.74), 然后 norm-clip 到 ball 内 ‖x_hyp‖ < 1/√0.74 ≈ 1.16
- 把 x_hyp 喂给 RQ-VAE encoder (替代 raw x)

**优势**:
- ✅ 不修改 quantizer (避免 Task #80 否证路径)
- ✅ 把输入空间真双曲结构显式搬运到 encoder 输入
- ✅ encoder 自然学到双曲特征, residual 也可能保留部分双曲结构

**风险**:
- ❓ 768d → 32d Poincaré map 可能丢失信息
- ❓ ‖x_hyp‖ 在 ball 内 → encoder 可能 collapse 到 origin
- ❓ 没有先验支持 "encoder 看到双曲输入会保留双曲结构"

**理论依据**: Sala 2018 Theorem 2 (树状组合) — 如果输入真的是树状, Poincaré 投影保距. 但 Ollivier 给出的是图结构, 不等于 embedding 几何.

**预期增益**: 未量化 (需要 Gate 2 训练才能验证).

### 2.2 方向 B: 后量化双曲 audit (Post-quantization Hyperbolic Audit)

**机制**: RQ-VAE 训练 loop 中, 每 N epoch 触发"双曲 audit":
- 提取当前每层 residual 点云 (N × 32d)
- 计算每层 Ollivier κ (or Ollivier-like)
- 跟 H2 类目树估计 κ=-0.74 比较
- 若偏差 > 阈值 → 触发 auxiliary loss: λ * ‖κ_layer - κ_target‖²

**优势**:
- ✅ 不修改 quantizer 主体 (避免 Task #80 否证路径)
- ✅ auxiliary loss 在 *训练* 时把双曲信号注入, 不在 *推理* 时改 cost matrix
- ✅ 可叠加在现有任何 recipe (跟 #30 per-layer Codebook Transforms 兼容)

**风险**:
- ❌ Ollivier κ 计算昂贵 (Hungarian O(deg^3)) → 不能每 step 都算
- ❌ κ 在不同层不同 (L0 可能 0, L3 可能 -0.5) → 单 target 值不准
- ❓ auxiliary loss λ 难调 (太弱无效果, 太强覆盖 main loss)

**理论依据**: Task #70 Ollivier 方法, 跨层级做 regularization.

**预期增益**: 需 λ sweep + per-layer κ 估计才能评估.

### 2.3 方向 C: 曲率锚定初始化 (Curvature-Anchored Initialization)

**机制**: 替代 kmeans_init, 用 Ollivier κ 锚定 codebook 初始位置:
- 对每个 codebook, 用 Poincaré ball with κ=-0.74 做 geometric k-means
- codebook 初始位置满足"双曲度量聚类"而非欧氏
- 后续 RQ-VAE 训练保持 codebook 在 ball 内

**优势**:
- ✅ 不修改 training loop, 只改 init
- ✅ 不改 cost matrix (避免 Task #80 否证路径)
- ✅ 双曲结构从一开始就在 codebook 几何里

**风险**:
- ❌ kmeans_init 已有 evidence 是好 init (Task #84 baseline R@10=0.1020). 改 init 可能损害.
- ❓ "geometric k-means on Poincaré ball" 算法不一定 well-defined (需要 Riemannian k-means 实现)
- ❓ Init 差异可能在训练中被梯度覆盖

**理论依据**: Gu 2019 mixed-curvature product spaces — 在不同空间做 kmeans 可能保结构.

**预期增益**: 未量化 (跟 kmeans_init A/B test 才能验证).

### 2.4 方向 D: 双流并联 (Dual-Stream Hybrid)

**机制**: 不替换 RQ-VAE, 而是并行跑两路:
- Stream 1: baseline RQ-VAE (欧氏 residual)
- Stream 2: hyperbolic-aware RQ-VAE (用方向 A 或 B)
- 最终 SID = f(stream1_sid, stream2_sid), 例如 concat [stream1_digit, stream2_digit]

**优势**:
- ✅ 完全并行, 任何现有 recipe 都不动
- ✅ Stream 1 fallback 保证不退化 (north star: 不可降级 baseline)
- ✅ Stream 2 探索双曲信号

**风险**:
- ❌ SID 长度翻倍 → T5 输入变长, 训练慢
- ❌ 双流权重难调 (Stream 2 多大影响?)
- ❌ 复杂度爆炸, 调试困难

**理论依据**: 多视角融合 (multi-view fusion) — 在推荐系统用过, 但没在 SID 检索用过.

**预期增益**: 未量化.

---

## 3. R11.5 决策矩阵

| 方向 | 跟 Task #80 否定路径的差异 | 实施复杂度 | 预期 ROI | 推荐度 |
|------|------------------------|-----------|---------|--------|
| A. 预量化双曲感知 | ✅ 不改 quantizer | 中 (需要 Poincaré map 工具) | 中 (理论干净) | ⭐⭐⭐ |
| B. 后量化 audit | ✅ 不改 cost matrix | 中-高 (Ollivier 在 loop 里) | 中 (reg loss 风险) | ⭐⭐ |
| C. 曲率锚定 init | ✅ 不改 cost matrix | 中 (geometric kmeans) | 低-中 (init 风险) | ⭐⭐ |
| D. 双流并联 | ✅ 不改现有 recipe | 高 (双流并行) | 不确定 | ⭐ |

**R11.5 透明建议**: 方向 A 最干净, 跟 Task #80 否证路径最远, 且实施复杂度可控. 建议 Gate 2 第一阶段跑 A.

---

## 4. Gate 1 设计交付物 (本 issue 要求)

按 Issue #41: "本 Gate 只交付设计方案 + 可行性评估, 不跑任何训练."

**本次交付**:
1. ✅ 4 个候选方向设计 (A/B/C/D 上文)
2. ✅ 风险/优势/理论依据逐项分析
3. ✅ R11.5 决策矩阵
4. ⏭️ 等待 owner 选方向 (per Issue #41 §反证 + Step 0.B 跟 R11.4 AskUserQuestion 限制)
5. ⏭️ Gate 2: 训练实施 (owner 决策后启动, NOT 本 Gate 范围)

---

## 5. 关键 caveats (R11.5 transparent)

1. **方向 A/B/C/D 都没在本仓库实证过**, 任何方向都可能在 Gate 2 失败 (NO-GO)
2. **NORTH STAR 限制**: 任何方向必须不可降级 baseline (Issue #30 GO R@10=0.1020)
3. **R7 GPU 约束**: Gate 2 训练需 GPU, 当前 4×L40S 满载, 等空闲后再启动
4. **R12 checkpoint**: Gate 2 训练必须每 epoch 强制 save + delete old (R12 硬规则)
5. **R9 编号**: Gate 2 实施 = task332 (max(331)+1)
6. **跟当前活跃任务的兼容性**: task327 K=256 Stage 3 已 early-stop + task328 R-Drop 4-arm 进行中 → Gate 2 等 GPU 空闲 (~14:30 AEST)

---

## 6. 关联引用

- Issue #41 (主, owner R11.5 自启)
- verdicts/task331_issue41_gate0_h_mds_input_space.md (Gate 0 PASS)
- Task #70 (Ollivier κ input space)
- Task #80 (residual κ=0 否证)
- Task #82 (|κ|≈0.05 弱信号下限)
- Issue #30 (per-layer Codebook Transforms — 跟 Gate 1 设计协同候选)
- Issue #34 D9 (per-layer 异构 hash — 跟 Gate 1 设计方向 A 概念重叠)
- paper.md §6.7.4 (R10 backlog 真空 housekeeping 决策)

---

result: Issue #41 Gate 1 设计交付 — 4 个候选方向 (A 预量化感知 / B 后量化 audit / C 曲率锚定 init / D 双流并联), R11.5 推荐 A 但等 owner 决策. 实施需要 GPU, 当前满载等空闲.