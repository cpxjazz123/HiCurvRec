# Task #334 / Issue #43 — Gate 2 实验设计 (Issue #41 方向A 预量化双曲感知映射)

**日期**: 2026-07-30 14:36
**触发**: Issue #43 owner 2026-07-30 04:37 创建, R14 强制处理
**状态**: 📝 **DESIGN REGISTERED (deferred, no GPU, no training)**
**类型**: 实验设计登记 (跟 task331 RECORDING 模式一致 — 不实施)

---

## 1. Issue #43 核心主张 (来自 owner, 完整摘录)

| 项 | 内容 |
|----|------|
| 来源 | 承接 Issue #41 (Gate 0 全 PASS + Gate 1 架构设计交付) |
| 方向 A | 预量化双曲感知映射 (RQ-VAE encoder 之前, 对 768d 输入做 Poincaré map) |
| H1 | 用 expmap0(x_768d, κ=-0.74) 预处理, 让 encoder 不必"压平"双曲信号 |
| H2 (反证) | encoder 压平可能是数据真实偏好, 强加双曲输入反而损害稳定性 |
| 决策依据 | R@10 > 0.1022 (Issue #30 GO 端点) / R@10 ≤ 0.1020 = NO-GO |
| 状态 | "未运行，纯实验设计提案。启动条件：GPU 空闲 + 决策拍板" |

**重要修正** (Issue #43 显式声明): 不使用已撤销的 0.1053 anchor (Issue #40 Gate 0 FAIL), 改用 **R@10 baseline 0.1020 / Issue #30 GO 0.1022** 作为唯一有效基准。

---

## 2. 当前状态 (R10 backlog 真空 + Issue #30 marginal GO)

- **R@10 杠杆 NO-GO 收口** (§16 backlog 17 方向 × 17 verdict): Issue #30 唯一 GO marginal (+0.2pp)
- **GPU 状态**: GPU 1 被 task194 K=64 Stage 3 占用 (88% util), GPU 0/2/3 空闲
- **决策门**: Issue #43 明确要求 "GPU 空闲 + 决策拍板" 才启动 Gate 2a 实施
- **R12 强制**: Gate 2b Stage 1 训练必须每 epoch 强制 save + delete old (R12)
- **R7 约束**: 不抢已用卡 (GPU 1 task194 不能用)

---

## 3. R11.5 决策矩阵 (本任务)

| 选项 | 收益 | 成本 | 决策 |
|------|------|------|------|
| 立即启动 Gate 2a (代码实施) | 中 (路径清晰) | 低-中 (R11.4 critical 改 src/) | ❌ DEFER (等决策拍板) |
| 仅登记设计 (本任务) | 低 (合规 Issue #43) | 零 | ✅ YES (本任务) |
| 完全 ignore Issue #43 | 零 | 零 (破 R14) | ❌ NO |

**R11.5 透明选择**: 仅登记设计 + 写 verdict (本任务). 备选 "立即启动 Gate 2a" 因 (1) R11.4 critical decision (改 src/) + (2) GPU 不全空闲 (GPU 1 task194 占) 被拒.

---

## 4. 关键 caveats

1. **Issue #43 是设计 only**, 不预支 Gate 2b/2c 一定能跑
2. **基线**: R@10 baseline=0.1020 / Issue #30 GO=0.1022 (per Issue #40 Gate 0 修正)
3. **R7 约束**: GPU 1 当前 task194 K=64 Stage 3 占用, Gate 2b 启动必须用 GPU 0/2/3
4. **R12 ckpt**: 实施前必须 patch train loop 每 epoch save + delete old
5. **R9 编号**: 实施时 = task336 (max(335)+1, 给 task335 Issue #44 留号)

---

## 5. 关键文件 + 物理产物

| 文件 | 用途 |
|------|------|
| `descriptions/task334_issue43_gate2_design.md` | 本文件 (任务描述) |
| `verdicts/task334_issue43_gate2_design_register.md` | 登记 verdict (deferred) |
| Issue #43 GitHub 评论 | 登记 verdict 路径 + R11.5 决策 |

---

## 6. R14 闭环

- Issue #43 设计已登记 (本文件)
- 启动条件: GPU 空闲 (0/2/3) + owner 决策拍板
- 不在 R10 backlog 真空内主动启动 (R11.4 critical decision 改 src/ 上游)

---

## 7. 关联引用

- Issue #43 (主)
- Issue #41 (Gate 0 PASS + Gate 1 设计交付)
- verdicts/task331_issue41_gate0_h_mds_input_space.md (Gate 0)
- verdicts/task331_issue41_gate1_architecture_design.md (Gate 1 设计)
- verdicts/task333_issue42_unified_formula_nogo.md (Issue #42 NO-GO 实施层)
- Task #80 (residual κ=0 否证)
- Issue #30 (当前唯一有效 GO 端点 R@10=0.1022)

---

result: Task #334 Issue #43 Gate 2 设计登记 (deferred, no GPU). 仅登记 Issue #43 4-Gate 实验设计 (a 实现/b 训练/c 下游/反证), 不立即启动. 启动条件: GPU 0/2/3 空闲 + owner 决策拍板. R@10 基准 0.1020 / Issue #30 GO 0.1022 (per Issue #40 修正). R11.5 决策: 仅登记, 备选 "立即启动 Gate 2a" 因 R11.4 critical + GPU 不全空闲被拒.