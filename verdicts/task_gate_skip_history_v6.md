# Task #290 Gate 0 — 第六次越闸历史记录盘点 (Issue #21 Gate 0 闭环)

> **完成日期**: 2026-07-29
> **状态**: ✅ **Gate 0 PASS — 六次越闸记录完整, 每条判据可独立复核**
> **核心结论**: 仓库中六次越闸的 verdict 文件齐全, 跨过理由可追溯. 本盘点为 Issue #21 Gate 1 (launcher header 约束) 提供完整证据链.

---

## 1. 六次越闸汇总表

| 次序 | 出处 verdict | 被跳过判据 | 实际执行 launcher | Stage 4 端点 | 跨过理由 |
|------|--------------|-----------|------------------|-------------|---------|
| **1** | `verdicts/task200_v5_user_4step_result.md` | cos_mean < 0.30 (L1 0.9054 / L2 0.9277) | task200_v5 (4-step user recipe) | Stage 4 R@10=0.0938 (Task #225 复跑 baseline) | "fallback option B" (Issue #21 body 概括; verdict 原文无该字符串) |
| **2** | `verdicts/task222_pck_earlystop_replay_result.md` | §6.7.4 stop-loss (i) L0 ≥ 90% (实测 65.62% ep29) | task222 PCK earlystop replay → task223 Stage 2 | task223 Stage 2 SID 推断启动, 后续 Stage 4 NO-GO (task225 R@10=0.0938) | "🟢 GO Task #223 Stage 2 SID 推断 (用 best_collision ep29 ckpt)" — 当场触发仍判 GO |
| **3** | `verdicts/task263_issue17_gate2_task253_direct_utilization_result.md` + `verdicts/task265_issue17_gate1_fix_apply_result.md` | §6.7.4 stop-loss (i) (Issue #13 Gate 2 经 #16/#17 复核) | Issue #13 Gate 2 launchers | Issue #13 Gate 2 Stage 4 R@10=0.000403 (越闸 audit) → task225 R@10=0.0938 (退回) | **量没打印** (trainer nested-scope bug, Issue #17 修复) |
| **4** | `verdicts/task237_issue10_arm_b_result.md` | Issue #10 Gate 1 \|B−C\| ≥ 15pp (实测 5pp FAIL) | `scripts/task237_arm_b_full_chain.sh` (Issue #19) | Arm B R@10=0.1021 vs Arm C 0.1058 (-3.5%) | "Gate 1 PARTIAL → STOP" — launcher 串成一条命令, 没有求值点; Issue #19 修复 |
| **5** | Issue #18 (closed) | §6.7.4 stop-loss (i) 阈值口径 (Stage 1 argmin vs Stage 2 Sinkhorn 差 45.31pp) | Issue #18 Gate 1 launchers | 阈值口径锁定为 Stage 1 argmin | 量测法本身有歧义; Issue #18 修复 |
| **6** | `verdicts/task275_a2_a3_curriculum_parallel_result.md` + `verdicts/task276_a2_stage4_eval_result.md` | §6.7.4 stop-loss (i) L0 ≥ 90% (A2 实测 89.1% × 3 续训) | `scripts/task270_utilization_curriculum.sh` RECIPE=A2 | task276 Stage 4 R@10=0.0985 (NO-GO -3.4% vs baseline 0.1020) | "R11.4 自主决策 (§6.7.4 stop-loss (i) L0 ≥ 90% threshold 视为 proxy, 真正瓶颈是 collision_rate. A2 L0=89.1% + collision_rate=0.1293 是已知可工作组合 (Task #178 precedent 验证). Proceed Stage 2 → 3 → 4)" — task275 §2.4 原文 |

---

## 2. 每次越闸详细证据

### 1. Task #200 — fallback option B (Issue #21 body 概括)

- **verdict**: `verdicts/task200_v5_user_4step_result.md`
- **判据**: cos_mean < 0.30 (判 L1/L2 utilization); 实测 L1=0.9054 / L2=0.9277 (远超阈值, 不应是 PASS 状态)
- **跨过理由**: Issue #21 body 写 "task200 「fallback option B」"; **verdict 原文无 "fallback option B" 字符串** (R11.3 透明记录: 这是 Issue 作者从 verdict 摘要的语义, 不是 verdict 原文)
- **下游**: task200 后续 4-step user recipe 继续, Stage 4 R@10=0.0938 (Task #225 复跑)
- **根因类型**: 判据有, 被以 "fallback option B" 名义跨过

### 2. Task #222 — §6.7.4 stop-loss (i) 当场触发仍判 GO

- **verdict**: `verdicts/task222_pck_earlystop_replay_result.md`
- **判据**: §6.7.4 stop-loss (i) L0 ≥ 90%; 实测 ep29 L0=65.62%
- **跨过理由原文**: "🟢 GO Task #223 Stage 2 SID 推断 (用 best_collision ep29 ckpt)" — **当场触发 stop-loss 仍判 GO**
- **下游**: task223 Stage 2 推断 + Stage 4 eval; 最终 task225 R@10=0.0938 (NO-GO -8.1%)
- **根因类型**: 判据有, 当场触发仍判 GO

### 3. Issue #13 Gate 2 / task253 — §6.7.4 stop-loss (i) 量没打印

- **verdict**: `verdicts/task263_issue17_gate2_task253_direct_utilization_result.md` + `verdicts/task265_issue17_gate1_fix_apply_result.md`
- **判据**: §6.7.4 stop-loss (i) L0 ≥ 90%; **trainer nested-scope bug 让 utilization 量根本没打印**
- **跨过理由**: 量不可见, 闸门无求值; **不是执行侧"主动跨过", 而是工具缺陷**
- **下游**: Issue #13 Gate 2 Stage 4 R@10=0.000403 (越闸 audit) → task225 R@10=0.0938 (退回)
- **根因类型**: 判据的量根本没打印 (trainer nested-scope bug, 已由 Issue #17 修复)
- **修复**: Issue #17 trainer 修复让 utilization 打印; Issue #16/#17/#18 Gate 1/Gate 2 实测确认

### 4. Task #237 — Issue #10 Gate 1 launcher 没求值点

- **verdict**: `verdicts/task237_issue10_arm_b_result.md`
- **判据**: Issue #10 Gate 1 |B−C| ≥ 15pp collision 差; 实测 5pp (FAIL)
- **跨过理由原文**: "Gate 1 PARTIAL → STOP" — **launcher 串成一条命令, 没有求值点; 量在 launcher 末尾才被读, 但命令已启动 Stage 2 推断**
- **下游**: Arm B R@10=0.1021 (跟 Arm A 0.1020 持平) — 3-arm 曲线崩塌, 不进入 Gate 2/3
- **根因类型**: 判据可求值, launcher 把 Stage 2→3→4 串成一条命令, 没有求值点
- **修复**: Issue #19 闸门模板 (scripts/issue19_gate_template.sh) 强制把求值点写在 Stage 2 启动前

### 5. Issue #18 — §6.7.4 stop-loss (i) 阈值口径歧义

- **verdict**: Issue #18 (closed, completed); 验证证据 `verdicts/task280_*.md` (Issue #18 闭环)
- **判据**: §6.7.4 stop-loss (i) L0 ≥ 90%; **口径歧义**: Stage 1 argmin vs Stage 2 Sinkhorn 差 45.31pp
- **跨过理由**: 量测法本身有歧义, 不同口径给出不同结论
- **下游**: Issue #18 闭环: 口径锁定为 Stage 1 argmin (Issue #18 Gate 1), 重新求值 task253 L0=73.44% / task222 ep29 L0=65.62% 复算 PASS
- **根因类型**: 判据的量测法本身有歧义
- **修复**: Issue #18 口径锁定 + Issue #19 闸门模板延伸

### 6. Issue #20 — §6.7.4 stop-loss (i) 已被严格化, 仍被 "proxy + precedent" 跨过

- **verdict**: `verdicts/task275_a2_a3_curriculum_parallel_result.md` + `verdicts/task276_a2_stage4_eval_result.md` + `verdicts/task271_a1_beta_zero_run_result.md` + `verdicts/task288_issue20_l0_utilization_3recipe_result.md`
- **判据**: §6.7.4 stop-loss (i) L0 ≥ 90% at epoch ≥ 30; 实测 A2 L0=89.1% (57/64) × 3 续训 plateau
- **跨过理由原文** (task275 §2.4):
  > §6.7.4 stop-loss (i) L0 ≥ 90% threshold 视为 proxy, 真正瓶颈是 collision_rate. A2 L0=89.1% + collision_rate=0.1293 是已知可工作组合 (Task #178 precedent 验证). Proceed Stage 2 → 3 → 4.
- **下游**: task276 Stage 4 R@10=0.0985 (NO-GO -3.4% vs baseline 0.1020); task271 A1 FAIL (β=0.0 mode collapse); task288 Issue #20 全谱 NO-GO
- **根因类型**: 判据已严格化, launcher 有求值点 (Issue #19 模板), 量测法已统一 (Issue #18 口径), **执行侧仍以 R11.4 自主决策名义用 proxy + precedent 跨过**
- **修复**: **本 issue (Issue #21) 即修复这一类** — launcher header 约束 (`GATE_DECLARATION` + `precedent_override=forbidden` 默认值)

---

## 3. 跨过理由分类 (R11.3 透明记录)

| 跨过理由类型 | 出现次数 | 对应越闸次序 | 治理手段 |
|--------------|---------|-------------|---------|
| **判据有, 被以 "fallback" 名义跨过** | 1 (task200) | 1st | Issue #19 闸门模板 (量在 launcher 末尾被读) + Issue #21 header 约束 (precedent_override=forbidden) |
| **判据有, 当场触发仍判 GO** | 1 (task222) | 2nd | Issue #19 闸门模板 (非零退出) + Issue #21 header 约束 |
| **判据的量根本没打印** | 1 (Issue #13) | 3rd | Issue #17 trainer 修复 (已闭环) |
| **判据可求值, launcher 没求值点** | 1 (task237) | 4th | Issue #19 闸门模板 (已闭环) |
| **判据的量测法本身有歧义** | 1 (Issue #18) | 5th | Issue #18 口径锁定 (已闭环) |
| **判据已严格化, 执行侧仍以 "proxy + precedent" 跨过** | 1 (Issue #20) | 6th | **Issue #21 launcher header 约束 + precedent_override=forbidden 默认值 (本 issue 修复中)** |

---

## 4. 关键决策点 (R11.3 + R11.5)

| # | 决策 | 选择 | 备选 | 理由 |
|---|------|------|------|------|
| 1 | Gate 0 记录粒度 | ✅ 6 次越闸各 1 行 | 6 次 × 5 行 (跨过理由原文 + 判据 + launcher + Stage 4 + 根因) | Issue #21 body 要求"每次越闸记录完整, 每条判据可独立复核"; 当前表格 + 详细段已覆盖 |
| 2 | task200 fallback option B 标记 | ✅ "Issue #21 body 概括; verdict 原文无该字符串" | 照搬 Issue #21 body 原文 | R2 + R11.3 透明记录, 不允许 verdict 跟上游 issue body 串通 |
| 3 | Gate 0 通过条件 | ✅ "六次越闸记录完整, 每条判据可独立复核" → 全部达标 | 退回补充 | 当前 10 个 verdict 文件齐全, 跨过理由原文可追溯 |
| 4 | 不启动 Gate 1 之前先 commit Gate 0 | ✅ commit 后启动 Gate 1 | Gate 1 + Gate 0 一起 commit | Issue #21 body "执行顺序: Gate 0 → Gate 1 → Gate 2 → Gate 3", 每 Gate 失败即 STOP, 不跨 Gate 取数 |

---

## 5. 物理产物

```
verdicts/task_gate_skip_history_v6.md  (本文件, Gate 0 闭环)
descriptions/task290_issue21_gate_enforcement_v3.md  (任务定义)
verdicts/task290_issue21_gate_enforcement_result.md  (最终 verdict, 待 Gate 1/2/3 完成)
```

---

## 6. R2 KB 更新 (后续 backlog 引用)

- **第六次越闸根因 = 治理问题**: Issue #17/#18/#19 切掉了前五类的越闸根因 (量没打印 / 口径不一致 / launcher 没求值点), 但**第六类 (判据已严格化, 执行侧仍以 "proxy + precedent" 跨过) 是治理问题, 不是工具问题**. Issue #21 的 launcher header 约束 + precedent_override=forbidden 默认值是当前唯一可行的机械阻断
- **越闸治理历史**: Issue #16 → #17 → #18 → #19 → #21 是同一方向的逐步收紧. **任何后续越闸都需要先在本表登记 (Gate 0), 再启动 Gate 1 launcher header 修改, 最后用 Gate 2 越闸计数暴露**
- **task275 §2.4 「R11.4 自主决策」非恶意**: 原文是「用户 override 『do by yourself』」(CLAUDE.md R11). Issue #21 不指责执行侧越权, 而是把 owner 自己定的硬线 (Issue #18/#19 闸门 + Issue #20 Gate 1/2/3) 与执行侧自主决策权 (R11.x) 的边界写明. **后续 owner override 必须显式标注「本 override 是否覆盖 §6.7.4 stop-loss (i)」, 不标注则默认不覆盖**

result: Task #290 Gate 0 — 第六次越闸历史记录盘点. **六次越闸记录完整, 每条判据可独立复核**, Gate 0 PASS. 跨过理由分类: fallback / 当场 GO / 量没打印 / launcher 没求值点 / 量测法歧义 / proxy+precedent. 第六类 (本 issue 修复中) 是治理问题, 不是工具问题. 进入 Gate 1 launcher header 约束.