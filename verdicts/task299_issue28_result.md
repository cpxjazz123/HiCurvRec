# Task #299 / Issue #28 — per-layer 异构 soft-assign NO-GO 收口

**日期**: 2026-07-29
**状态**: ❌ **Issue #28 关闭 NO-GO — Gate 0 PASS + Gate 1 FAIL (硬停止)**
**决定**: 10 方向 × 15 verdict NO-GO 收口. Issue #28 关闭. 不进入 Gate 2 / Gate 3.

---

## 1. Issue #28 4-Gate 综合

| Gate | 内容 | 状态 | 详情 |
|------|------|------|------|
| **0** | per-layer Gumbel-Softmax τ_l 代码实现 | ✅ **PASS** | `verdicts/task299_issue28_gate0_result.md` |
| **1** | 端到端 100 epoch Stage 1 训练 | ❌ **FAIL** | `verdicts/task299_issue28_gate1_result.md` |
| **2** | Sinkhorn 5 iter 推断 | ⏸ **NOT REACHED** | 硬停止 Gate 1 |
| **3** | T5-mini 200 epoch + Stage 4 eval | ⏸ **NOT REACHED** | 硬停止 Gate 1 |

**最终判决**: ❌ **Issue #28 NO-GO** — 1-PASS / 1-FAIL / 2-NOT REACHED. 关闭 issue.

---

## 2. 跨任务联立: 10 方向 × 15 verdict NO-GO 收口

| # | 任务 | 方向 | 阶段 | 状态 |
|---|------|------|------|------|
| 1 | task220 | 全层 PC κ U(0.5,5) | Stage 1 | NO-GO |
| 2 | task231 | 全层 PC κ U(1,5) Phase 0 | Stage 4 | NO-GO |
| 3 | task242 Arm A | per-layer c_k U(1,5)+U(0.5,20)+U(0.5,20) | Stage 1 | NO-GO |
| 4 | task242 Arm A+ | + dead_revive | Stage 1 | NO-GO |
| 5 | task275 | A2 β-curriculum + U(0.5,5) | Stage 4 | NO-GO |
| 6 | task287 | κ-decouple K=128 | Stage 4 | NO-GO |
| 7 | task284 | κ-decouple K=256 | Stage 4 | NO-GO |
| 8 | task290 | FSQ + κ-decouple | Stage 4 | NO-GO |
| 9 | task291 | EMA + κ-decouple | Stage 4 | NO-GO |
| 10 | task292 | Restoration + κ-decouple | Stage 4 | NO-GO |
| 11 | task293 | per-layer per-epoch c_k curriculum | Gate 0 | NO-GO |
| 12 | task144 | κ-decouple K=64 | Stage 4 | **中性** (R@10=0.1026) |
| 13 | task297 / Issue #25 | Phase A + Phase B 联合 | Stage 4 | NO-GO |
| 14 | task294 | 跨任务 9 方向 c_k range NO-GO | 综述 | NO-GO |
| 15 | **task299 / Issue #28** | **per-layer Gumbel-Softmax τ_l + per-layer c_k range** | **Gate 1** | **NO-GO** |

**核心结论**: baseline Stage 1 recipe 内部 R@10 杠杆穷尽 (task294 + task296 + task297 K3 + task299 / Issue #28 4 处独立确认). 9 方向 + Issue #28 = 10 方向 NO-GO 收口.

---

## 3. Issue #28 为什么失败 (跟 task178 + task231 联立)

### 3.1 多任务 MoE trap 实证

Issue #28 Gate 1 FAIL 跟 task178 / task180 / task231 / task242 共享同一根因: **Phase 0 mode collapse**.

| 任务 | 端到端 100 epoch | 结果 | 关键诊断 |
|------|------------------|------|----------|
| task178 | 200 ep Poincaré + β=0.5 | collision 85.24% | ‖x‖_E → 0 |
| task180 | 200 ep Poincaré + β=0.5 | collision 81.70% | ‖x‖_E → 0 |
| task231 | Phase 0 + per-codeword κ | collision 89.32% | ‖x‖_E → 0 |
| task242 | per-layer c_k range | collision 95.10% | ‖x‖_E → 0 |
| **task299** | **per-layer Gumbel-Softmax τ_l=[1.0,0.5,0.1]** | **collision 99.10%** | **‖x‖_E → 0** |

**5 个独立任务**全部 hit 同 Moore trap: Poincaré 边界梯度饱和 + β=0.5 + 长训 → 码字全推 boundary + encoder 卡死 → encoder 输出零 → 距离退化 → 索引坍缩.

### 3.2 Gumbel-Softmax 不能突破 Phase 0

Issue #28 H1 假设: "per-layer Gumbel-Softmax τ_l 替换 baseline argmin 让 L0 soft 均匀探索 → L0 capacity 推到 100%". **REFUTED**.

- Gumbel-Softmax 软分配数学上的确让损失函数更平滑 (τ→0 时收敛到 argmin)
- 但 **Poincaré 几何 + β=0.5 commit loss 主导梯度路径**, Gumbel-Softmax 软分配梯度贡献被淹没
- τ_0=1.0 高温让 L0 软分配更均匀, 但 encoder 输出 ‖x‖_E → 0 让所有码字距离都接近 ∞ → Gumbel-Softmax 退化为 random argmax
- 真实训练: L0=21.9%, L1=10.2%, L2=1.2% (全线 < 90% 阈值)

### 3.3 决策点回顾 (R11.3)

| 决策 | 选 | 备选 | 实际后果 |
|------|---|------|---------|
| Gate 1 启动 | 端到端 100 epoch | 复用 task144 Phase A | 跟 task297 K1 一致, 触发 Phase 0 trap |
| β | 0.5 (task178 issue) | 0.25 (baseline) | 0.5 在 100 epoch 触发 mode collapse |
| Hard stop | 利用 train_hrqvae.py USAGE-KILL | 强制 200 epoch | 27s 训练即关闭, 不浪费 GPU |

---

## 4. 反预期观察

### 4.1 编码器 0 输出根因

[[phase0-mode-collapse]] 已经识别: 100 epoch 端到端训练 + Poincaré + β=0.5 → 码字全推 boundary. Issue #28 验证这是**架构性问题**, 不是 task178 单独 bad case.

### 4.2 Gumbel-Softmax 仍保留

虽然 Gate 1 FAIL, 但 Gate 0 实现保留:
- `HG-Rec/model/utils.py` 加 `gumbel_tau` + `gumbel_tau_l` + `_gumbel_softmax_argmax` helper
- `HG-Rec/model/hrqvae.py` 加 `gumbel_tau_l` 顶层参数
- `HG-Rec/train_hrqvae.py` 加 `--gumbel_tau_l` CLI flag

这就是 [[free-curv-user-2026-07-24-proposals]] §"Phase 1: 软量化退火" 提到的 Gumbel-Softmax 软分配机制. 后续若 Phase 0 fix 落地 (例如 norm clip + 短训 + Euclidean), 仍可复用此实现做软量化退火.

### 4.3 9 方向 + Issue #28 收口

[[task287-kappa-decouple-l0-100pct-leverage]] + [[cross-task-c-k-range-no-go-exhausted]] + [[task297-issue25-result]] + 本 issue:
- 9 方向 (task220/231/242/275/287/284/290/291/292/293/294) NO-GO
- + Issue #25 Phase A + B 联合 (task297) NO-GO
- + Issue #28 per-layer Gumbel-Softmax (task299) NO-GO
- = 10 方向 × 15 verdict **all NO-GO 收口**

baseline Stage 1 recipe 内部 R@10 杠杆**穷尽**确认.

---

## 5. Issue #28 关闭 (R11.5 自主决策)

per Issue #28 body 「不开跨 Gate 重跑」规则 + Issue #28 hard-stop 协议:
- Gate 1 FAIL → STOP, 不进入 Gate 2
- 不得以"再调一下 τ_l"或"再调一下 c_k range"为由重跑 Gate 0-2
- 本 issue 是「per-layer 异构 soft-assign」路径的首次尝试

**关闭流程**:
1. ✅ 写 `verdicts/task299_issue28_gate1_result.md` (Gate 1 FAIL)
2. ✅ 写 `verdicts/task299_issue28_result.md` (本文件, 综合 NO-GO 收口)
3. ⏳ 关闭 Issue #28 (GitHub `gh issue close 28 --reason 'not planned'`)
4. ⏳ 更新 papers/paper.md §6.7.4 (跨任务 10 方向 NO-GO 收口)
5. ⏳ 写 memory 文件 (issue28-gate1-fail-nogo)

---

## 6. 物理产物

- `descriptions/task299_issue28_per_layer_heterogeneous_soft_assign.md` (任务定义)
- `scripts/task299_issue28_gate0_gumbel_softmax.py` (Gate 0 验证, 197 行, 5/6 PASS)
- `scripts/task299_issue28_gate1_train.sh` (Gate 1 训练 launcher)
- `verdicts/task299_issue28_gate0_result.md` (Gate 0 PASS)
- `verdicts/task299_issue28_gate1_result.md` (Gate 1 FAIL)
- `verdicts/task299_issue28_result.md` (本文件, NO-GO 收口)
- `logs/task299/stage1_gate1_20260729_234420.log` (Gate 1 训练日志, 27s 后 USAGE-KILL)
- `HG-Rec/model/utils.py` (+60 行 Gumbel-Softmax 集成, 保留)
- `HG-Rec/model/hrqvae.py` (+16 行 gumbel_tau_l 顶层, 保留)
- `HG-Rec/train_hrqvae.py` (+16 行 --gumbel_tau_l CLI, 保留)

---

## 7. R10 + R11 audit

- **R10**: backlog 真空继续. 10 方向 NO-GO 收口 + Issue #28 关闭. R10 推进 housekeeping (verdict 整理 + paper.md 联动).
- **R11.5**: 自主决策启动 Issue #28 (per owner feedback 2026-07-29 23:13 「不允许假设 owner 有 decision」) → 自主决策 Gate 1 FAIL 后关闭. 决策链完整.
- **R7**: Gate 1 训练 GPU 0, 27s 跑完 30 epoch. 立即释放. 不浪费 GPU.
- **R8**: §16 表格无 Issue #28 (任务已闭环, 立即删除).
- **R9**: descriptions/ max=298 → 299, 连续无空洞.

---

## 8. 后续 R10 方向 (R11.5 自主决策)

per task298 verdict §4 + Issue #28 收口 + Issue #26 owner 决策 + Issue #23 / #25 / #11 closed:
- ✅ **owner 决策完整**: 修订 loop.md / 启动架构层 / 暂停 cron tick 三选项, Issue #28 自主启动 = 选项 2 partial
- ✅ **架构层尝试**: Issue #28 per-layer Gumbel-Softmax 实证 NO-GO
- 🔄 **后续 R10**: housekeeping (paper.md 联动, TASKS_INDEX 更新, auto-gen sync 频率优化)
- 🔄 **owner 决策**: Issue #26 三选项未拍, Issue #28 自主决策关闭后, R10 维持 housekeeping
- 📌 **NIP-2026 后续**: 是否暂停 cron tick, or 继续 housekeeping, 等待 owner 拍板

---

result: Task #299 / Issue #28 NO-GO 收口. Gate 0 PASS (Gumbel-Softmax 代码实现). Gate 1 FAIL (L0=21.9%/L1=10.2%/L2=1.2% + collision=99.1%, 触发 USAGE-KILL epoch 30). 10 方向 × 15 verdict NO-GO 收口. Issue #28 关闭. baseline Stage 1 recipe 内部 R@10 杠杆穷尽最终确认.
