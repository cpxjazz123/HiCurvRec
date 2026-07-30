# Task #335 / Issue #44 — 设计登记 verdict (deferred, 双重阻塞)

**日期**: 2026-07-30 14:37
**触发**: Issue #44 owner 2026-07-30 04:37 创建, R14 强制处理
**状态**: 📝 **DESIGN REGISTERED (deferred, 双重阻塞)**
**类型**: 设计登记 verdict (no code, no training, no GPU)

---

## 1. 双重阻塞

| 阻塞项 | 证据 | 状态 |
|--------|------|------|
| **H1 前提 REFUTED** | Issue #44 论证链基于 "R137 κ=0 是 dead point". task333 实测 R137 autograd grad at κ=0 = **8030** (显著非零), 不是 fixed point | REFUTED |
| **Gate 0 硬阻塞** | Task #144/#145 任一未证明 codebook 健康化 (utilization ≥ 90% 全层). 当前均 OPEN backlog, 未实施 | BLOCKED |
| **task333 5/5 FAIL** | 新统一公式 5 测试全 FAIL: kappa_zero NaN, gradient_continuity jump=764, symmetry_err=13.32, perf 3.26× slower | FAIL |

**三重证据** → Issue #44 Gate 1+ **不得启动**.

---

## 2. Issue #44 设计要点 (摘要)

| Gate | 任务 | 通过条件 | 阻塞 |
|------|------|----------|------|
| **0** | 前置检查: Task #144/#145 任一 codebook 健康化 PASS | utilization ≥ 90% | **BLOCKED** (Task #144/#145 未实施) |
| **1** | 统一公式重写 (Möbius + tan_κ⁻¹) | κ→0 Taylor 极限 + κ=0 grad ≠ 0 | **REFUTED** (task333 5/5 FAIL) |
| **2** | 对称初始化 3-arm (θ_init=+0.01/-0.01/0) | 200 epoch 短训, 记录 A/B/C 收敛方向 | 等 Gate 0 + 1 |
| **3** | codebook 健康度联调 | L0/L1/L2 ≥ 90% + collision ≤ baseline | 等 Gate 2 + Task #144/#145 |
| **4** | 下游评估 | test R@10 > 0.1022 | 等 Gate 3 |

**关键决策门**:
1. R@10 baseline = **0.1020** / Issue #30 GO = **0.1022** (per Issue #40 修正)
2. 不使用已撤销的 0.1053 anchor

---

## 3. H1 前提反证细节 (跟 task333 联立)

| H1 前提 | 任务 #333 实测 | 状态 |
|---------|---------------|------|
| R137 κ=0 ∂/∂κ = 0 (dead point) | R137 autograd grad = **8030.09** | **REFUTED** (实测非零) |
| 新统一公式 κ=0 grad ≠ 0 | 新公式 autograd grad = **NaN** | FAIL |
| 新统一公式梯度连续 | jump = **764.16** | FAIL (跟 R137 jump=2387 同类问题) |
| 新统一公式对称性 OK | symmetry_err = **13.32** | FAIL (数值不稳定) |
| 新统一公式球面性质 OK | self_dist=2e-15, sym=8.9e-16, ≤ π/√κ | PASS (唯一 PASS) |
| 新统一公式性能 | 3.26× slower than R137 | FAIL (工程不实用) |

**结论**: Issue #44 H1 论证链断裂, R137 实际不是 κ=0 fixed point, 任何 "统一公式修复 θ=0 死点" 尝试都没有 motivation (R137 autograd 实际有 flow).

---

## 4. H2 独立价值 (即使 H1 REFUTED, H2 仍可探索)

| H2 子问题 | 价值 | 阻塞 |
|-----------|------|------|
| 对称初始化 θ_init=±0.01 验证球面结论是否初始化偏置 | 中 (澄清混杂因素) | Gate 0 阻塞 |
| 直接用 R137 测 θ_init=0 训练结果 | 中 (实测反证 Issue #42 假设) | GPU + Gate 0 |

**注意**: 即使 H2 实施, 也需要 Gate 0 (Task #144/#145) 先证明 codebook 健康化. 在坍缩问题解决前, 任何 κ 学习实验都受坍缩干扰, 无法纯净测 κ 收敛方向.

---

## 5. R11.5 透明决策

**选了**: 仅登记 + 写 verdict (本任务)
**为什么**: Issue #44 启动条件是 (1) Gate 0 前置满足 + (2) 实施统一公式. 当前双重阻塞: H1 前提 REFUTED (task333) + Gate 0 未满足 (Task #144/#145 未实施).
**备选**: 立即启动 Gate 1 重写 — 因 (1) task333 5/5 FAIL + (2) Gate 0 阻塞 + (3) H1 前提 REFUTED 三重证据被拒.
**触发**: Issue #44 明确说 "不建议在坍缩问题解决前启动 Gate 1 以外的任何 GPU 工作", 本任务符合此约束.

---

## 6. 启动条件 checklist (供后续 owner 决策时用)

- [ ] Task #144 或 #145 任一证明 codebook utilization ≥ 90% 全层
- [ ] task333 5/5 FAIL 被新尝试 override (理论 unlikely, 但流程允许)
- [ ] Owner 决策拍板: 即使 H1 REFUTED 仍要尝试 (需新 motivation)
- [ ] R12 ckpt patch + GPU 全空闲
- [ ] H2 (对称初始化) 可独立于 H1 启动, 但仍需 Gate 0 阻塞解除

---

## 7. 跟 task333 联立

| 项 | task333 | task335 (本任务) |
|----|---------|-----------------|
| 类型 | 实施层 (MCKG 统一公式 + 验证) | 设计层 (实验设计登记) |
| 状态 | 5/5 FAIL + R137 dead-point REFUTED | 双重阻塞 (H1 REFUTED + Gate 0 阻塞) |
| 后续 | Issue #42 关闭 (NO-GO, 强证据) | Issue #44 保持 OPEN, deferred |

**联立**: Issue #42 + Issue #44 联立关闭论证链 — free-curv 主线 (Task #89/#135/#137/#138) NO-GO 收口维持, 任何统一公式重启都没有 motivation (实测 R137 不是 dead point + 新公式 5/5 FAIL).

---

## 8. 关键文件 + 物理产物

| 文件 | 用途 |
|------|------|
| `descriptions/task335_issue44_unified_formula_redesign.md` | 任务描述 |
| `verdicts/task335_issue44_design_register.md` | 本文件 (登记 verdict) |

---

## 9. R14 闭环

- Issue #44 设计登记 (本文件 + description)
- 双重阻塞: H1 前提 REFUTED (task333) + Gate 0 硬阻塞 (Task #144/#145)
- 不在 R10 backlog 真空主动推进

---

## 10. 关联引用

- Issue #44 (主)
- Issue #42 (来源, RECORDED only)
- verdicts/task331_issue42_free_curv_postmortem_record.md (Issue #42 RECORDING)
- verdicts/task333_issue42_unified_formula_nogo.md (Issue #42 实施层 NO-GO 5/5 FAIL + R137 dead-point REFUTED)
- Task #135 / #137 / #138 (free-curv 主线 NO-GO 收口)
- Task #144 / #145 (Gate 0 硬前置)
- verdicts/task334_issue43_gate2_design_register.md (Issue #43 独立登记)

---

result: Task #335 Issue #44 设计登记 deferred (no GPU, no training). 双重阻塞: (1) H1 前提被 task333 REFUTED (R137 autograd grad κ=0 = 8030 非零, 不是 dead point), (2) Gate 0 硬阻塞 (Task #144/#145 任一未证明 codebook 健康化). task333 5/5 FAIL 联立证据: 新公式 κ=0 grad=NaN, jump=764, symmetry_err=13.32, 3.26× slower. R11.5 决策: 仅登记, 备选 "立即启动 Gate 1" 因三重证据被拒. Issue #44 论证链断裂, free-curv 主线 NO-GO 收口维持.