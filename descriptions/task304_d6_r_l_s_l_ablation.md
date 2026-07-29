# Task #304 / D6 ablation — Issue #30 r_l vs s_l 拆分 3-arm 找真杠杆

**日期**: 2026-07-30
**状态**: 🟡 backlog 准备启动
**Decision**: R10 + R11.5 自主决策 (owner feedback 2026-07-29 23:13)

---

## 1. 来源

承接 Issue #30 (per-layer Codebook Transforms r_l=[0.1,1,10] + s_l=[2,2,2]) Stage 4 Test R@10=0.1022 (+0.2pp marginal GO) + K5 关键发现「码字几何路径是真杠杆」+ Issue #29 (per-layer K_l 异构 Stage 4 NO-GO) + Issue #31 (encoder regularization Gate 1 FAIL) 三方向实证.

**R10 backlog 真空收口后最高 ROI 候选**: Issue #30 +0.2pp marginal GO 不显著, 必须 ablation 才能确认是真杠杆还是 noise. 拆 r_l (radius scale) vs s_l (norm scale) 看哪个独立贡献 R@10.

---

## 2. 假设

**H1 (r_l 是真杠杆)**: per-layer r_l=[0.1, 1.0, 10.0] 把码字缩放到不同半径, Stage 4 R@10 ≥ 0.1020. s_l=[2,2,2] 是辅助稳定项 (单独 NO-GO).

**H2 (s_l 是真杠杆)**: per-layer s_l=[2.0, 2.0, 2.0] 缩放 norm 把码字推到健康区, Stage 4 R@10 ≥ 0.1020. r_l 是辅助项.

**H3 (r_l + s_l 协同)**: 单独 r_l 或 s_l NO-GO, 只有两者组合才 GO. (Issue #30 实证 +0.2pp, 但 r_l 和 s_l 单独不知道).

---

## 3. 实验设计

**3-arm ablation** (单一变量对比):

| Arm | r_l | s_l | 预测 R@10 |
|-----|-----|-----|-----------|
| A (r_l only) | [0.1, 1.0, 10.0] | [1.0, 1.0, 1.0] (baseline) | H1: ≥ 0.1020 / H3: < 0.1020 |
| B (s_l only) | [1.0, 1.0, 1.0] (baseline) | [2.0, 2.0, 2.0] | H2: ≥ 0.1020 / H3: < 0.1020 |
| C (Issue #30 reference) | [0.1, 1.0, 10.0] | [2.0, 2.0, 2.0] | 0.1022 (reference) |

**受控因素**: K=[64,128,256] baseline + e_dim=32 + seed=42 + Musical_Instruments 5-core + T5-mini 200 epoch + c_k range=[(1,5),(0.5,20),(0.5,20)] 沿用 Issue #30 配置.

**配置区别 (Task #304 vs Issue #30)**:
- Issue #30: r_l=[0.1,1,10] + s_l=[2,2,2] + c_k_range (双轴)
- Task #304 Arm A: r_l=[0.1,1,10] + s_l=[1,1,1] + c_k_range (单轴 r_l only)
- Task #304 Arm B: r_l=[1,1,1] + s_l=[2,2,2] + c_k_range (单轴 s_l only)
- Task #304 Arm C: r_l=[0.1,1,10] + s_l=[2,2,2] + c_k_range (Issue #30 reference)

---

## 4. 阶段闸门 (每 arm 独立跑 4-Gate)

**Gate 0 (实现)**: scripts/task304_d6_gate0_r_only.py + scripts/task304_d6_gate0_s_only.py + scripts/task304_d6_gate0_reference.py 提交 + 回归测试.

**Gate 1 (Stage 1 训练, GPU)**: 100 epoch, 通过条件 L0/L1/L2 util ≥ 90% + collision ≤ 0.20.

**Gate 2 (Sinkhorn 推断)**: 5 iter, 4-digit unique ≥ 9500.

**Gate 3 (Stage 3 T5 + Stage 4 eval)**: R@10 > 0.1020 → GO.

每 arm 失败 → hard-stop, 不跨 Gate.

---

## 5. GPU 资源

- Arm A Gate 1: GPU 0 (Stage 1 100 epoch, ~30 min)
- Arm B Gate 1: GPU 2 (并行跑)
- Arm C Gate 1: GPU 3 (Issue #30 reference 重跑, ~30 min)
- Gate 3 Stage 3: 3 GPU 并行 T5-mini 200 epoch (每个 ~75 min early stop)
- 总 GPU 时间: ~2.5 hr 并行 (R7 不抢卡, 4 卡全空闲可并行)

---

## 6. R10 + R11.5 audit

- **R9**: descriptions/ max=304 ✅ (Task #304 = D6 ablation, 连续无空洞)
- **R10**: D6 ablation 是 R10 backlog 真空收口后最高 ROI 候选 (Issue #30 GO marginal 验证 + K5 关键发现深层验证)
- **R11.5**: owner feedback 2026-07-29 23:13「不允许假设 owner 有 decision」→ 自主决策启动 Task #304
- **R7**: 4 张 L40S 全空闲 (2026-07-30 检查), 3 arm 并行 Stage 1 + Stage 3
- **R12**: best ckpt 强制保留 (save_limit=1)
- **R11.4 critical**: 启动是 GPU 占用 (3 卡 2.5 hr), R7 + R10 兜底 = 自主启动

---

## 7. 物理产物 (待写)

- `scripts/task304_d6_gate0_r_only.py` (Arm A: r_l only wrapper)
- `scripts/task304_d6_gate0_s_only.py` (Arm B: s_l only wrapper)
- `scripts/task304_d6_gate0_reference.py` (Arm C: r_l + s_l reference wrapper)
- `scripts/task304_d6_gate1_stage1_train_arm{abc}.sh` (3 Stage 1 launchers)
- `scripts/task304_d6_gate3_stage3_train_arm{abc}.sh` (3 Stage 3 launchers)
- `scripts/task304_d6_gate4_stage4_eval_arm{abc}.sh` (3 Stage 4 eval launchers)
- `verdicts/task304_d6_*.md`
- `products/task304/`

---

## 8. 关联

- [[task301-issue30-stage4-result]]: Issue #30 GO marginal (R@10=0.1022 +0.2pp) — reference 配置
- [[task301-issue30-gate3-gate4-result]]: K5 关键发现 — 码字几何路径是真杠杆
- [[task300-issue29-stage4-result]]: Issue #29 K_l NO-GO (-4.0%) — false positive 案例
- [[task302-issue31-gate1-result]]: Issue #31 encoder reg USAGE-KILL — 错杠杆案例
- [[task303-issue32-dual-axis-synergy]]: Issue #32 双轴协同 — 互补实验 (c_k range + r_l + s_l 三轴叠加)
- [[phase0-mode-collapse]]: 11 方向 Phase 0 mode collapse 根因, Task #304 找真杠杆路径深层验证

---

## 9. 决策点

**R11.4 critical**: 启动 Task #304 是 3-arm × 5-Gate = 15 实验, GPU 时间 ~5 hr 总 (并行 2.5 hr). ROI 评估:
- 高: Issue #30 GO marginal 验证 + K5 关键发现深层确认 + 后续 Issue #32 + paper section 锚点
- 中: 边际 +0.2pp 数字 noise 可能让所有 arm NO-GO (但仍是 valid 实证)
- 风险: 3 卡并发可能跟其他 R10 backlog 候选冲突 (但当前 backlog 空)

**R11.5 决策**: **启动 Task #304 (D6 ablation)**, 高 ROI + 当前 backlog 空 + 4 GPU 空闲. 跟 Task #303 (Issue #32 双轴协同) 并行, 互补覆盖 r_l/s_l/c_k_range 三轴.

---

result: Task #304 / D6 ablation 准备启动. Issue #30 r_l vs s_l 拆分 3-arm 找真杠杆. R11.5 自主决策启动 (owner feedback 2026-07-29 23:13 + R10 backlog 真空 + 4 GPU 空闲). 跟 Task #303 (Issue #32 双轴协同) 并行, 互补覆盖 r_l/s_l/c_k_range 三轴.
