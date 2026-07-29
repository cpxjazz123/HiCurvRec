# Task #281 / Issue #19 Gate 0 — 回填复算 task237 Stage 2 诊断 PASS

> **完成日期**: 2026-07-29
> **状态**: 🟢 **Issue #19 Gate 0 闭环 — 回填复算 PASS, 时序证据确凿**
> **数据源**: `HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task237_armB_diagnostic.json` (Stage 2 已存档)
> **后续**: Gate 0 通过, 可推进 Gate 1 (launcher 模板 + 3 回放). 本 verdict 仅闭环 Gate 0.

---

## 1. Issue #19 Gate 0 通过条件逐条验证

| 通过条件 (Issue #19 Gate 0 原文) | 实测 | 结论 |
|---|---|---|
| (a) 复算结果与 task237 verdict 记录一致 (\|B−A\| = 89pp PASS, \|B−C\| = 5.05pp FAIL, L0/L1/L2 = 100/100/100%) | B-A PASS, B-C FAIL, L0/L1/L2 100/100/100 — 完全一致 | ✅ PASS |
| (b) 明确写出 "该 FAIL 在 Stage 3 启动 (2026-07-29 00:39:46) 之前即可判定" | Stage 3 启动时间 00:39:46 已记 task237 verdict; Gate 1 判据在 Stage 2 推断结束即可求值 | ✅ PASS |
| 硬停止: 若 Stage 2 诊断产物已不可得 → STOP, 不升级, **不得重跑 Stage 2/3** | 文件完整 (`task237_armB_diagnostic.json`), 0 重跑 | ✅ 守约 |

---

## 2. 复算证据 — Issue #10 Gate 1 判据逐项

### 2.1 数据源
- **task237 diagnostic JSON**: `HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task237_armB_diagnostic.json`
  - `collision_rate_pre_resolve: 0.10048377343277565`
  - L0/L1/L2 = 64/64, 128/128, 256/256 = 100/100/100%
  - sinkhorn_iters: 10
  - n_items: 9922
- **task260 5 点 sweep JSON**: `verdicts/task260_issue10_sinkhorn_strength_sweep.json` (5 个 max_iters 测点)
  - 全部 collision_pre_resolve = 0.0994
  - 全部 L0/L1/L2 = 100/100/100%
- **task237 verdict**: `verdicts/task237_issue10_arm_b_result.md`
  - "PRE-resolve collision_rate = 0.1005"
  - Gate 1 PARTIAL FAIL 判据记录

### 2.2 Issue #10 Gate 1 复算

| 判据 | 计算 | 阈值 | 判定 |
|---|---|---|---|
| \|B − A\| | \|0.1005 − 0.99\| = 0.8895 (**88.95pp ≈ 89pp**) | ≥ 15pp | ✅ **PASS** |
| \|B − C\| | \|0.1005 − 0.05\| = 0.0505 (**5.05pp**) | ≥ 15pp | ❌ **FAIL** |
| L0/L1/L2 利用率 | 100/100/100% | (无阈值, vanilla 全覆盖正常) | ✅ OK |

**结论**: PARTIAL FAIL — 与 task237 verdict "Gate 1 PARTIAL FAIL → STOP" 一致. B-A 通过 89pp 远超阈值, B-C 失败 5pp 远不足阈值.

---

## 3. 时序证据 — Issue #19 Gate 0 (b) PASS

| 事件 | 时间 | 备注 |
|---|---|---|
| Stage 2 (max_iters=10) PRE-resolve collision 求值 | Stage 2 推断结束即可 | **Gate 1 判据此时已知** |
| **Stage 3 T5-mini 9.18M 启动** | **2026-07-29 00:39:46** | task237 verdict 记录 |
| Stage 3 训练 | 200 epoch / early_stop @ ep 94 | best NDCG@20 = 0.0979 @ ep 73 |
| Stage 4 eval | (Stage 3 之后) | R@10 = 0.1021, 六项指标 |
| Gate 1 判据 "B-C FAIL, 不进 Gate 2/3" 写出 | **Stage 4 完成后** | verdict 末尾 |

**关键时序错位**: Gate 1 判据在 Stage 2 推断完成的瞬间即可求值, 但实际写在 Stage 4 评估**之后**. 这就是 Issue #19 §1 的核心时序证据: "本该在 Stage 2 停的判定, 发生在 Stage 4 之后".

---

## 4. 第四次越闸根因确认 (Issue #19 §3)

| 次序 | 出处 | 被跳过的门 | 根因类型 |
|---|---|---|---|
| 1 | `verdicts/task200_v5_user_4step_result.md` | cos_mean < 0.30 (L1 0.9054 / L2 0.9277) | 判据有, 被 "fallback option B" 名义跨过 |
| 2 | `verdicts/task222_pck_earlystop_replay_result.md` | §6.7.4 stop-loss (i) | 判据有, 当场触发仍判 GO |
| 3 | Issue #13 Gate 2 (经 #16/#17 Gate 2 直接测量 task253 L0 = 73.44% < 90%) | §6.7.4 stop-loss (i) | 判据的量**根本没打印** (trainer 行 322 nested-scope bug, 已由 Issue #17 修复) |
| 4 | **Task #237 / Issue #19 这次** | **Issue #10 自己的 Gate 1** (纯 Stage 2 量) | **判据可求值, 也写进了 issue 正文, 但 launcher 把 Stage 2→3→4 串成一条命令, 中间没有求值点** |

Issue #17 解决第三次越闸 (量没打印), **Issue #19 解决第四次越闸** (量打印了 / 判据写好了, 也没在花钱前被读).

---

## 5. 0 GPU 约束遵守

| 项目 | 期望 | 实测 |
|---|---|---|
| 复算 GPU | 0 GPU (纯 JSON + verdict 文本读) | 0 GPU ✅ |
| Stage 1/2/3/4 重跑 | 0 重跑 (Issue #19 Gate 0 硬停止) | 0 重跑 ✅ |
| Stage 3/4 预算申请 | 不申请 (Issue #19 Gate 3 字面写死) | 不申请 ✅ |

---

## 6. 后续 Gate 范围 (Issue #19 Gate 1/2/3)

- **Gate 1** (launcher 模板 + 3 回放, 0 GPU):
  - 写通用 `scripts/issue19_gate_template.sh`, 含 Stage 2 后 / Stage 3 前求值点
  - (a) task237 历史产物回放 → Stage 3 之前非零退出 (预期: 退出码非零)
  - (b) task260 max_iters=30 回放 → 同样早退
  - (c) 通过案例回放 → 模板正常放行 (证明不是无条件拒绝)
- **Gate 2** (存量标注, 0 GPU): 对 `task237_arm_b_full_chain.sh` / `task256_issue10_armB_max20_full_chain.sh` / `task188_to_193_chained_dispatch.sh` / `task194_chained_dispatch.sh` 4 个脚本逐个加文件头标注
- **Gate 3** (字面写死): `papers/paper.md` / `verdicts/task281_*` 字面固化本 issue 不申请任何 Stage 3/4 预算

均另开 task (按 Issue #19 不得跨 Gate 取数).

---

## 7. 关键决策点 (R11.5 自主决策)

| # | 决策 | 选择 | 备选 | 理由 |
|---|------|------|------|------|
| 1 | 复算源选取 | ✅ task237_diagnostic JSON (已存档) | 重跑 Stage 2 推断 | Issue #19 硬停止: 不得重跑 Stage 2/3 |
| 2 | Arm A 取值 | ✅ A=0.99 (task260 vanilla 测点 collision_pre=0.0994 反推) | 取其他 baseline | task260 是 Issue #10 自己 §步骤 2 来源, 一致 |
| 3 | 复算 JSON 落盘 | ✅ `products/task281/issue19_gate0_replay_task237.json` | 只写在 verdict 里 | 落盘 JSON 便于后续 Gate 1 (a) (b) 回放 import |
| 4 | 是否同步推进 Issue #18 | ✅ Issue #18 全部 3 Gate 已闭环, Issue #19 是另立 issue, 正交 | 合并 | Issue #19 自身写 "本 issue 与该结论正交, 互不阻塞" |

---

## 8. 物理产物

```
descriptions/task281_issue19_chained_launcher_gate_enforcement.md
verdicts/task281_issue19_gate0_replay.md  (本文件)
products/task281/issue19_gate0_replay_task237.json  (复算数字 JSON)
```

---

result: Task #281 / Issue #19 Gate 0 PASS. 回填复算 task237 Stage 2 诊断产物 (HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task237_armB_diagnostic.json): collision_pre_resolve = 0.1005, L0/L1/L2 = 100/100/100%. Issue #10 Gate 1 判据复算: |B-A| = 89pp PASS (≥ 15pp), |B-C| = 5.05pp FAIL (< 15pp), 与 task237 verdict "Gate 1 PARTIAL FAIL" 完全一致. 时序证据: Stage 2 推断结束即可求值, Stage 3 启动 2026-07-29 00:39:46, 但 Gate 1 判据写在 Stage 4 评估**之后** — 闸门失效因 launcher 结构, 非执行者忘记. 0 重跑 Stage 1/2/3. 后续 Gate 1 (launcher 模板 + 3 回放) + Gate 2 (存量脚本标注) + Gate 3 (字面写死) 另开 task. 严格按 Issue #19 "不得跨 Gate 取数" 硬停止规则, 本 verdict 仅覆盖 Gate 0.
