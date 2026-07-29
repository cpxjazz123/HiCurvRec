# Task #314 — Issue #38 Stage 3/4 训练协议改造 (NORTH STAR §3 实质突破方向)

**日期**: 2026-07-30
**状态**: 4-arm 设计修订 (Arm α/γ 已 NO-GO 收口, 加入 Arm δ 立即可跑)
**目的**: 按 owner task292 closure verdict §横向联立 "后续应该攻 [Stage 3/4 训练协议] 而非 [Stage 1/2 quantizer 架构]" + Issue #37 meta-eval 评估 + NORTH STAR §3 评估, 启动 Stage 3/4 训练协议改造方向.

**关键 insight (Issue #37)**:
- baseline #84 R@10=0.1020 **不是** Stage 4 ceiling. **task194_k0256 R@10=0.1053 是已实证的更高 baseline** (+3.3%).
- owner #30 GO marginal 0.1022 vs baseline 0.1020 = +0.2% (实质对齐 baseline) vs **task194_k0256 0.1053 = -3.0% 退化**.
- 24 方向 Stage 1/2 quantizer 改造全部 NO-GO 收口, #30 GO marginal 不构成 NORTH STAR §3 实质突破.
- **Stage 3/4 训练协议** = owner task292 §横向联立未 issue 化方向 = 真正新思路.

**关联**:
- Issue #37 (closed, meta-eval, R11.5 自主启动)
- Issue #38 (待起草, #37 评估结论 → 4-arm Stage 3/4 改造)
- owner task292 closure verdict §横向联立
- task278 (12 ckpt batch Stage 4 eval, task194_k0256 ⭐0.1053 anchor)
- task309 (T5-mini → T5-small 升级, **Arm α NO-GO** R@10=-4.0%)
- task312/313 (Issue #35 r_l/s_l 隔离, **Arm γ NO-GO** R@10=-17%)
- task301/307/309b (Issue #30 K-sweep, **Arm ε FULL closed** K=100 ceiling +2.5%)
- task194_k0256 (anchor R@10=0.1053 = +3.3%)
- task84 baseline (catastrophic at K=100, 不算 Arm β 杠杆)

**4-arm 设计 (R11.5 自主决策, 2026-07-30 修订)**:
- ~~**Arm A (baseline #84)**: Stage 3 T5-mini 200 epoch + Stage 4 beam=20 → R@10=0.1020 (anchored)~~ — 已 anchored
- ~~**Arm B (Task #309 anchor)**: Stage 3 T5-small 60M 200 epoch + Stage 4 beam=50 (K14) → 验证 Arm α~~ — **NO-GO** (R@10=0.0979 -4.0%, task309)
- ~~**Arm C (#30 GO + T5-small)**: Stage 3 T5-small on #30 SID + Stage 4 beam=50 → 验证 α × #30 协同~~ — 同 Arm α 失败原因推断 NO-GO
- ~~**Arm D (task194_k0256 anchor + Stage 4 K=100 + rerank)**: 验证 task194 anchor + ε K 增大 + rerank~~ — 待实施
- **NEW Arm δ (Stage 4 post-process rerank)**: 不重新训练, 在 evaluate() 的 preds.reshape 之后插入 rerank hook. 利用 per-item prior (frequency / popularity / embedding similarity) 重排 top-K. **立即可跑** (10 min/eval, 无 GPU training). 3 子方向:
  - δ1: Frequency prior — P(item) ∝ log(freq) 调整 logit
  - δ2: Embedding centroid similarity — 用 item_emb (Stage 1) 余弦相似度
  - δ3: Sequence-level rerank — 同一 sequence 内不同 beam 之间的多样性奖励
- **NEW Arm ε (K-sweep)**: ✅ FULL closed K=100 ceiling R@10=0.1045 (Issue #30 specific)

**当前 backlog (按 ROI 排序)**:
1. **Arm δ1-δ3 (Stage 4 rerank)** — 最高 ROI, 不需要重新训练, 立即可跑
2. **Arm β (HNSW + rerank)** — 中 ROI, 需要额外实现 (~1 day)
3. **task194_k0256 @ K=100 test** — 验证 task194 anchor 是否也享受 K=100 amplifier

**通过条件**:
- Arm δ 任一子方向 R@10 > 0.1022 → Stage 4 后处理杠杆发现
- Arm δ 任一子方向 R@10 > 0.1045 → Issue #30 K=100 ceiling 也可超越
- Arm β R@10 > 0.1053 → NORTH STAR §3 实质突破 (task194 anchor 之上)

**反证 / 失败条件**:
- Arm δ 全部子方向 ≤ 0.1022 → Stage 4 后处理无杠杆
- Arm β ≤ 0.1022 → Stage 3 训练+Stage 4 协议路径也 NO-GO

**预期**: Arm δ = 30 min GPU 时间 (3 sub × 10 min); Arm β = 4-8h (HNSW build + rerank). 当前 4 GPU 全空闲.

**关键决策点 (R11.3 透明)**:
- 选 task194_k0256 R@10=0.1053 作为 anchor 因为 (a) 是已实证最高 R@10 (b) 实质突破 baseline 0.1020 (+3.3%) = NORTH STAR §3 实质突破门槛候选
- 选 4-arm 而不是 1-arm 因为 (a) owner task292 §横向联立暗示 Stage 3/4 多个改造点 (b) 4-arm ablation 找最强 Stage 3/4 改造杠杆
- 不替换 owner #30 GO 端点, 而是在 #30 + task194_k0256 双 anchor 基础上改造 Stage 3/4
- 加 Arm δ 立即可跑: 不需要 Stage 3 retraining (节省 ~6 GPU-hour), 直接 hack evaluate() post-processing. R11.3 接受这是低成本探针.
- **2026-07-30 修订**: Arm α (T5-small NO-GO) + Arm γ (Issue #35 NO-GO) 收口, 加入 Arm δ rerank. 总方向不变但 ROI 排序重排.

**GPU**: GPU 0/1/2/3 全空闲 (R7 ✅)
**已用 GPU**: 无 (task304/309/312/313 全部完成, products/ 落盘)
result: Task #314 — Issue #38 Stage 3/4 训练协议改造 description (Arm α/γ NO-GO 收口, 加入 Arm δ 立即可跑). 见 verdicts/task316 NO-GO 收口
