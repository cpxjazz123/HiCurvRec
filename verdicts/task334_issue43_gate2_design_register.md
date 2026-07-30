# Task #334 / Issue #43 — Gate 2 设计登记 verdict (deferred, no GPU)

**日期**: 2026-07-30 14:37
**触发**: Issue #43 owner 2026-07-30 04:37 创建, R14 强制处理
**状态**: 📝 **DESIGN REGISTERED (deferred)**
**类型**: 设计登记 verdict (no code, no training, no GPU)

---

## 1. Issue #43 设计要点 (摘要)

| Gate | 任务 | 通过条件 | 决策 |
|------|------|----------|------|
| **2a** | 实现 expmap0(x_768d, κ=-0.74) + κ=0 退化验证 | 代码 commit + 回归 PASS | R11.4 critical (改 src/上游) → 需 owner 拍板 |
| **2b** | Stage 1 100 epoch 健康度 | L0/L1/L2 ≥ 90% + collision ≤ baseline + loss 收敛 | GPU 必须空闲 (R7) |
| **2c** | Stage 2/3/4 下游评估 | test R@10 > 0.1022 (vs Issue #30 GO) | ~5h GPU |
| 反证 | H1 vs H2 (数据真实偏欧氏?) | 实测记录, 不预设赢家 | — |

**关键决策门**:
1. R@10 baseline = **0.1020** (HG-Rec Task #84) / Issue #30 GO = **0.1022** (唯一有效 GO 端点)
2. 不使用已撤销的 0.1053 anchor (Issue #40 Gate 0 FAIL, Issue #43 显式修正)
3. 中性区间 (0.1020~0.1022): 判定中性, 不视为突破
4. ≤ 0.1020: NO-GO
5. > 0.1022: GO

---

## 2. 实施 status

| 阶段 | 当前 | 阻塞 |
|------|------|------|
| Issue #41 Gate 0 | ✅ DONE (3-segment PASS) | — |
| Issue #41 Gate 1 设计 | ✅ DONE (4 候选方向) | — |
| **Issue #43 Gate 2a (实施)** | ❌ NOT STARTED | (1) GPU 不全空闲 (GPU 1 task194 占用) + (2) R11.4 critical decision (改 src/) 需 owner 拍板 |
| **Issue #43 Gate 2b (Stage 1)** | ❌ BLOCKED | 等 2a + GPU 全空闲 |
| **Issue #43 Gate 2c (Stage 2/3/4)** | ❌ BLOCKED | 等 2b PASS |

---

## 3. R11.5 透明决策

**选了**: 仅登记 + 写 verdict (本任务)
**为什么**: Issue #43 明确要求 "GPU 空闲 + 决策拍板". 当前 GPU 1 被 task194 K=64 Stage 3 占用 (88% util, 6137 MiB), GPU 0/2/3 空闲但 (1) 启动 Gate 2a = 改 src/ 上游 = R11.4 critical decision + (2) 启动后还有 Stage 1/2/3/4 长链路, 没 owner 拍板启动 = 越权.
**备选**: 立即启动 Gate 2a 实施 — 因 R11.4 critical + 没 owner 拍板被拒.
**触发**: Issue #43 是单纯设计, 启动条件是 "GPU + 拍板" 两个外部条件, 当前都不满足.

---

## 4. 跟当前 backlog 真空的兼容性

| 项 | 状态 |
|----|------|
| §16 R10 backlog 17 方向 × 17 verdict | NO-GO 收口 |
| Issue #30 | 唯一 GO marginal (R@10=0.1022) |
| Issue #43 (本任务) | 设计登记 deferred, 不在 R10 backlog 主动推进范围 |
| Task #144 / #145 (解耦调度) | OPEN backlog, 跟 Issue #43 设计兼容 (但当前未实施) |

**联立**: Issue #43 Gate 2a 启动会跟 task194 (GPU 1 Stage 3) 撞 GPU, 必须等 task194 完成或换 GPU 0/2/3.

---

## 5. 启动条件 checklist (供后续 owner 决策时用)

- [ ] GPU 0/2/3 空闲 (task194 K=64 Stage 3 必须完成)
- [ ] Owner 决策拍板: 改 src/ 上游 (R11.4 critical)
- [ ] R12 ckpt patch: 每 epoch save + delete old
- [ ] Issue #30 当前唯一 GO 0.1022 仍有效 (无新 issue 推翻)
- [ ] Gate 2a 回归测试脚本准备好 (κ=0 退化验证)

---

## 6. 关键文件 + 物理产物

| 文件 | 用途 |
|------|------|
| `descriptions/task334_issue43_gate2_design.md` | 任务描述 |
| `verdicts/task334_issue43_gate2_design_register.md` | 本文件 (登记 verdict) |

---

## 7. R14 闭环

- Issue #43 设计登记 (本文件 + description)
- 启动条件: GPU 0/2/3 空闲 + owner 拍板
- 不在 R10 backlog 真空主动推进 (R11.4 critical)

---

## 8. 关联引用

- Issue #43 (主)
- verdicts/task331_issue41_gate0_h_mds_input_space.md (Gate 0 PASS)
- verdicts/task331_issue41_gate1_architecture_design.md (Gate 1 设计)
- verdicts/task333_issue42_unified_formula_nogo.md (Issue #42 5/5 FAIL — 跟 Issue #43 无关但同方向)
- verdicts/task335_issue44_design_register.md (Issue #44 双重阻塞)

---

result: Task #334 Issue #43 设计登记 deferred (no GPU, no training). 启动条件: GPU 0/2/3 空闲 + owner 决策拍板 (R11.4 critical 改 src/). R11.5 决策: 仅登记, 备选 "立即启动 Gate 2a" 因 R11.4 critical + GPU 不全空闲被拒. Issue #43 决策依据 = R@10 baseline 0.1020 / Issue #30 GO 0.1022 (per Issue #40 修正).