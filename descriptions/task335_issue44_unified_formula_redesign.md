# Task #335 / Issue #44 — Issue #42 统一公式重写 + 对称初始化 (受 Task #144/#145 解耦调度前置门控)

**日期**: 2026-07-30 14:36
**触发**: Issue #44 owner 2026-07-30 04:37 创建, R14 强制处理
**状态**: 📝 **DESIGN REGISTERED (deferred, 前置门控阻塞, no GPU, no training)**
**类型**: 实验设计登记 (跟 task334 模式一致)

---

## 1. Issue #44 核心主张 (来自 owner, 完整摘录)

| 项 | 内容 |
|----|------|
| 来源 | 承接 Issue #42 (RECORDED only) — 统一 κ-stereographic 公式 + 对称初始化 |
| H1 | 用 tan_κ/tanh_κ 连续函数替换 3-branch torch.where, 让 θ=0 梯度非零 |
| H2 | 用对称初始化 (θ_init=±0.01) 验证"球面结论"是否初始化偏置产物 |
| Gate 0 前置 | **必须 Task #144/#145 任一证明 codebook 健康化**, 否则不启动 Gate 1+ |
| Gate 3 前置 | 必须 Task #144/#145 的健康化方案可用 (硬阻塞) |
| 决策依据 | R@10 > 0.1022 (Issue #30 GO 端点) |
| 状态 | "不建议在坍缩问题解决前启动 Gate 1 以外的任何 GPU 工作" |

**重要修正** (Issue #44 显式声明): 不使用已撤销的 0.1053 anchor, 用 R@10 baseline=0.1020 / Issue #30 GO=0.1022.

---

## 2. ⚠️ 关键前提反证 (task333 REFUTED Issue #42)

**Issue #44 的 H1 前提被 Task #333 实证 REFUTED**:

| H1 前提 | Task #333 实测 | 状态 |
|---------|---------------|------|
| R137 在 κ=0 是 "数学 fixed point" (∂/∂κ = 0) | R137 autograd grad at κ=0 = **8030** (显著非零) | **REFUTED** |
| 新统一公式 κ=0 梯度非零 | 新公式 autograd grad at κ=0 = **NaN** | **FAIL** |
| 新公式梯度连续性 OK | jump = **764.16** (跟 R137 同样不连续) | **FAIL** |
| 新公式对称性 OK | symmetry_err = **13.32** | **FAIL** |
| 新公式性能 OK | 3.26× slower than R137 | **FAIL** (工程不实用) |

**联立结论**:
- R137 实际不是 κ=0 数学 fixed point (autograd 通过 .abs() 在 κ=0 grad 非零)
- 新统一公式实施 5/5 测试 FAIL
- Issue #44 的 H1 (动机) 被实测 REFUTED, **H2 (对称初始化) 仍有学术价值**

---

## 3. 当前状态 (Gate 0 前置阻塞)

- **Task #144** (κ+codebook 解耦 Stage 2 only): OPEN backlog, **未证明能维持 codebook 健康**
- **Task #145** (软量化退火): OPEN backlog, **未证明能维持 codebook 健康**
- **Issue #44 Gate 0 硬条件**: 上述任一被证明 codebook utilization ≥ 90% 全层 → 才能启动 Gate 1+
- **R10 backlog 真空**: §16 17 方向 × 17 verdict 收口, Issue #30 唯一 GO marginal

**R11.5 决策**: Issue #44 Gate 1+ **不得启动** (per Issue #44 自身 Gate 0 硬条件).

---

## 4. R11.5 决策矩阵 (本任务)

| 选项 | 收益 | 成本 | 决策 |
|------|------|------|------|
| 立即启动 Gate 1 (重写统一公式) | 无 (task333 已证 5/5 FAIL) | 高 (GPU) | ❌ NO (前提 REFUTED + Gate 0 阻塞) |
| 仅登记设计 (本任务) | 低 (合规 Issue #44) | 零 | ✅ YES (本任务) |
| 完全 ignore Issue #44 | 零 | 零 (破 R14) | ❌ NO |

**R11.5 透明选择**: 仅登记 + 写 verdict (本任务), 显式标注 H1 前提被 task333 REFUTED + Gate 0 硬阻塞. 备选 "立即启动 Gate 1" 因 (1) 前提 REFUTED + (2) Gate 0 硬阻塞 + (3) task333 实证 5/5 FAIL 三重证据被拒.

---

## 5. 关键 caveats

1. **H1 前提 REFUTED**: Issue #44 论证链基于 Issue #42 "R137 κ=0 dead point" 假设, task333 实测 REFUTED (R137 grad=8030 非零)
2. **Gate 0 硬阻塞**: Task #144/#145 任一未证明 codebook 健康 → Gate 1+ 不得启动 (per Issue #44 自身 §Gate 0)
3. **基线**: R@10 baseline=0.1020 / Issue #30 GO=0.1022 (per Issue #40 修正)
4. **R7 约束**: GPU 1 task194 K=64 Stage 3 占用
5. **H2 仍有价值**: 即使 H1 REFUTED, H2 (对称初始化验证球面结论是否偏置) 仍是独立可探索的子问题, 但需要 Gate 0 满足后才能启动

---

## 6. 关键文件 + 物理产物

| 文件 | 用途 |
|------|------|
| `descriptions/task335_issue44_unified_formula_redesign.md` | 本文件 (任务描述) |
| `verdicts/task335_issue44_design_register.md` | 登记 verdict (deferred + premise REFUTED + Gate 0 阻塞) |
| Issue #44 GitHub 评论 | 登记 verdict 路径 + R11.5 决策 |

---

## 7. R14 闭环

- Issue #44 设计已登记 (本文件)
- H1 前提被 task333 REFUTED (R137 autograd grad at κ=0 = 8030 非零, 不是 dead point)
- Gate 0 硬阻塞: Task #144/#145 任一未证明 codebook 健康
- 不在 R10 backlog 真空内主动启动 Gate 1+

---

## 8. 关联引用

- Issue #44 (主)
- Issue #42 (Issue #44 来源, RECORDED only)
- verdicts/task331_issue42_free_curv_postmortem_record.md (Issue #42 RECORDING)
- verdicts/task333_issue42_unified_formula_nogo.md (task333 5/5 FAIL + R137 dead-point REFUTED)
- Task #135 (4 处硬分支 bug 诊断)
- Task #137 (R137 修复 + θ=0 fixed point 自承 — **task333 证伪**)
- Task #138 (codebook 坍缩终审 NO-GO)
- Task #144 / #145 (Gate 0 硬前置)
- Task #80 (residual κ=0 否证)

---

result: Task #335 Issue #44 设计登记 (deferred, 双重阻塞). 仅登记 Issue #44 5-Gate 实验设计 (0 前置 / 1 统一公式 / 2 对称初始化 / 3 codebook 联调 / 4 下游评估), 不立即启动. 双重阻塞: (1) H1 前提被 task333 REFUTED (R137 autograd grad κ=0 = 8030 非零, 不是 dead point), (2) Gate 0 硬阻塞 (Task #144/#145 任一未证明 codebook 健康化). R11.5 决策: 仅登记, 备选 "立即启动 Gate 1" 因前提 REFUTED + Gate 0 阻塞 + task333 5/5 FAIL 三重证据被拒.