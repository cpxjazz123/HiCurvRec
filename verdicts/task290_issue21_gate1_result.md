# Task #290 Gate 1 — launcher header 约束 + precedent_override 强制 (Issue #21 Gate 1 闭环)

> **完成日期**: 2026-07-29
> **状态**: ✅ **Gate 1 PASS — 三组回放全部达标**
> **核心结论**: Issue #21 launcher header 模板 + 三组回放验证通过. 模板在 §6.7.4 stop-loss (i) bound + L0 < 90% 时正确非零退出, 在 L0 ≥ 90% 时正确放行, 在 not_bound + auto_proceed=true 时正确放行. precedent_override=forbidden 默认值生效.

---

## 1. 实施产物

| 产物 | 路径 | 角色 |
|------|------|------|
| 模板 | `scripts/issue21_gate_header.sh` | 解析 + 求值 + 退出 |
| Launcher header (a) | `scripts/task275_a2_extend.sh` (前 30 行) | replay (a) 含 GATE_DECLARATION |
| Launcher header (b) | `scripts/task237_arm_b_full_chain.sh` (前 30 行) | replay (b) 含 GATE_DECLARATION |
| Launcher header (c) | `scripts/task290_pass_case.sh` | 人为构造通过案例 |
| 合成 baseline log | `logs/issue21_replay_b_synthetic_baseline_log.log` | replay (b) 的 L0=100% 证据 |

---

## 2. GATE_DECLARATION 模板 (五字段)

```
# GATE_DECLARATION:
#   section_6_7_4_stop_loss_i: <bound|not_bound>
#   stage_2_threshold:        <numeric or NA>
#   stage_3_threshold:        <numeric or NA>
#   auto_proceed_after_stage_2: <true|false>
#   precedent_override:       <allowed|forbidden>
```

**默认值** (R11.2 保守兜底, 显式声明优先):
- `section_6_7_4_stop_loss_i`: `bound` (R5 baseline 默认)
- `auto_proceed_after_stage_2`: `false` (要求显式求值, 默认不放行)
- `precedent_override`: `forbidden` (默认禁止 proxy / precedent 跨过)

**`precedent_override=allowed` 的硬约束**:
- 必须存在 `verdicts/<owner_approval>.md` 文件, 且该文件**包含本 launcher 文件名字符串**
- 否则模板非零退出
- 这是 Issue #21 治理的**核心边界**: 自主决策权 (R11.4) 与闸门机制 (Issue #18/#19) 的分界

---

## 3. 三组回放验证 (Gate 1 通过条件)

### Gate 1 (a) — task275_a2_extend.sh + task270 A2 真实 log (L0=89.1%)

**期望**: exit code = 1 (FAIL, 模板正确阻止 §6.7.4 stop-loss (i) 越闸)
**实际**: exit code = 1 ✅

```
[Issue #21] 解析声明:
  section_6_7_4_stop_loss_i = bound
  stage_2_threshold          = NA
  stage_3_threshold          = NA
  auto_proceed_after_stage_2 = false
  precedent_override         = forbidden
  ---
  Stage 1 L0 (epoch 30) = 89%   (阈值 ≥ 90%)
[Issue #21] ❌ FAIL: §6.7.4 stop-loss (i) 触发
  L0 = 89% < 90%, 禁止自动进入 Stage 3
  不接受任何 proxy / precedent / Task #178 类比 (precedent_override=forbidden)
```

**意义**: 这是第六次越闸 (task275 §2.4 「R11.4 proxy + precedent」) 的**回放阻断**. 同样的 launcher header 声明下, 模板拒绝自动进入 Stage 3. 即, 即便 R11.4 自主决策用 proxy + precedent 跨过, **模板在 launcher 头部层面就拒绝执行**, 没有给 proxy / precedent 留出运行空间.

### Gate 1 (b) — task237_arm_b_full_chain.sh + 合成 baseline log (L0=100%)

**期望**: exit code = 0 (PASS, 模板正确放行健康 Stage 1)
**实际**: exit code = 0 ✅

```
[Issue #21] 解析声明:
  section_6_7_4_stop_loss_i = bound
  stage_2_threshold          = 15
  stage_3_threshold          = NA
  auto_proceed_after_stage_2 = false
  precedent_override         = forbidden
  ---
  Stage 1 L0 (epoch 80) = 100%   (阈值 ≥ 90%)
  ✅ §6.7.4 stop-loss (i) 未触发 (L0 100% >= 90%)
[Issue #21] ✅ PASS — §6.7.4 stop-loss (i) 通过, 放行 Stage 3
```

**意义**: 确认 Issue #21 模板**不会误伤** Issue #19 修复的健康链 (task237 Stage 1 = task84 baseline L0=100%). 即, 模板的闸门是**有上下文的**, 不是无脑 FAIL.

### Gate 1 (c) — task290_pass_case.sh 自调用 (not_bound + auto_proceed=true)

**期望**: exit code = 0 (PASS, 模板尊重 owner 显式 not_bound 声明)
**实际**: exit code = 0 ✅

```
[Issue #21] 解析声明:
  section_6_7_4_stop_loss_i = not_bound
  stage_2_threshold          = NA
  stage_3_threshold          = NA
  auto_proceed_after_stage_2 = true
  precedent_override         = forbidden
  ---
  auto_proceed=true, bound=not_bound → header 声明放行 (无需 L0 求值)
[Issue #21] ✅ PASS — header 声明放行
```

**意义**: 确认 Issue #21 模板**尊重** owner 显式 not_bound 声明 (例如 k-sweep 类, 用户主动说「不受 §6.7.4 约束」). precedent_override=forbidden 在 not_bound 时不阻拦 (因为 not_bound 是 owner 显式声明, 不是 precedent 类比).

---

## 4. 模板实现细节

### 4.1 L0 提取 (awk regex 关键修复)

初版 regex `/ep([0-9]+)/` 错误匹配 "step" 里的 "ep" + "2" (来自 "step2 monitor"). 修复: 用 `\[step2 monitor ep([0-9]+)\]` 严格匹配.

```awk
match($0, /\[step2 monitor ep([0-9]+)\]/, arr)
ep = arr[1] + 0
```

### 4.2 epoch ≥ 30 过滤

只取 epoch ≥ 30 (warmup 完成) 的 L0 utilization, 避免训练早期低 L0 触发误判.

### 4.3 precedent_override=allowed 校验

`precedent_override=allowed` 是**严重放宽闸门**的声明, 必须 owner 显式 verdict 批准. 模板要求 `verdicts/<owner_approval>.md` 文件包含本 launcher 文件名.

R11.5 决策: 不自动创建 owner approval 文件. 必须 owner 在 prompt 显式说「批准 X launcher precedent_override=allowed」+ AI 写 verdict 文件 + 模板 grep 命中, 三步齐全才能生效. 这是 Issue #21 的核心治理边界.

---

## 5. 关键决策点 (R11.3 + R11.5)

| # | 决策 | 选择 | 备选 | 理由 |
|---|------|------|------|------|
| 1 | 模板放置位置 | ✅ `scripts/issue21_gate_header.sh` (独立文件) | 合并到 issue19_gate_template.sh | Issue #21 是独立治理方向 (precedent_override), 不污染 Issue #19 修复 |
| 2 | precedent_override 默认值 | ✅ `forbidden` (R11.2 保守兜底) | `allowed` (R11.4 自主决策权) | Issue #21 治理方向 = 默认禁止, 显式批准才放开 |
| 3 | L0 threshold 默认值 | ✅ 90% (§6.7.4 stop-loss (i) 原文) | 95% / 85% | 严格按 paper.md §6.7.4 阈值, 不偷换 |
| 4 | epoch 最小值 | ✅ 30 (warmup 完成) | 0 / 10 | 训练早期 L0 不稳定, 30 epoch 后才有比较意义 |
| 5 | precedent_override=allowed 的 owner approval 校验 | ✅ grep launcher 文件名 | 单独 owner_approvals/ 目录 | 跟现有 verdicts/ 结构对齐, 不引入新目录 |
| 6 | 合成 baseline log 用例 | ✅ 创建 `logs/issue21_replay_b_synthetic_baseline_log.log` | 仅用 task270 A1 log | task270 A1 是 β=0 欧氏, 跟 baseline 性质不同; 合成 log 显式说明 L0=100% |
| 7 | 不修改 Issue #19 模板 | ✅ 保留独立 | 在 issue19_gate_template.sh 里加 §6.7.4 检查 | Issue #19 修的是 Issue #10 Gate 1 collision 差, Issue #21 修的是 §6.7.4 L0 ≥ 90%. 两条独立治理, 不串通 |

---

## 6. 物理产物

```
scripts/issue21_gate_header.sh                  (新增, Gate 1 模板)
scripts/task275_a2_extend.sh                    (修改: 加 GATE_DECLARATION 块)
scripts/task237_arm_b_full_chain.sh             (修改: 加 GATE_DECLARATION 块)
scripts/task290_pass_case.sh                    (新增, Gate 1 (c) 人为构造通过案例)
logs/issue21_replay_b_synthetic_baseline_log.log  (新增, Gate 1 (b) 合成 baseline log)
verdicts/task290_issue21_gate1_result.md        (本文件, Gate 1 闭环)
verdicts/task290_issue21_gate_enforcement_result.md  (最终 verdict, 待 Gate 2/3 完成)
```

---

## 7. R2 KB 更新 (后续 backlog 引用)

- **Issue #21 模板 = precedent_override 默认 forbidden**: 任何 launcher 加 GATE_DECLARATION 时, `precedent_override=allowed` 必须有 `verdicts/<owner_approval>.md` 引用本 launcher. 不允许在 launcher 里直接写 `precedent_override=allowed` 而没有 owner verdict 批准
- **Issue #21 模板 ≠ 替代 Issue #19 模板**: Issue #19 修 collision 差 ≥ 15pp, Issue #21 修 §6.7.4 L0 ≥ 90%. 两个模板应该串联 (`source issue19_gate_template.sh && source issue21_gate_header.sh && evaluate_stage_2_gate ... && parse_gate_declaration ...`)
- **Issue #21 模板 ≠ 替代 owner 显式声明**: not_bound + auto_proceed=true 是 owner 显式声明, 模板尊重. 但 owner 不能用 precedent 类比 (e.g. "Task #178 precedent") 把 bound 改成 not_bound — precedent_override=forbidden 默认锁死
- **合成 log 仅用于回放验证**: `logs/issue21_replay_b_synthetic_baseline_log.log` 是测试产物, 不代表真实 task84 baseline. 真实 task84 baseline log 缺 L0 utilization 格式, 需要 Stage 1 重训 (或 Issue #17 trainer 输出格式 backport)

---

## 8. 后续 Gate 2 → Gate 3 (R10 推进)

- **Gate 2 — 越闸计数暴露** (零 GPU): 每次 cron tick 启动日志头部打印已发生的越闸次数 + 最近一次越闸 verdict 路径. 格式固定, 可被 grep 解析
- **Gate 3 — 硬停止**: 不申请任何 Stage 2/3/4 预算 (写死, 已在 Issue #21 body 明确)
- **执行顺序**: Gate 1 → Gate 2 → Gate 3, 任一失败即在该 Gate 处 STOP, 不跨 Gate 取数

result: Task #290 Gate 1 — launcher header 约束 + precedent_override=forbidden 强制. **三组回放全部 PASS**: (a) task275 L0=89% < 90% 正确 FAIL, (b) task237 baseline L0=100% 正确 PASS, (c) not_bound + auto_proceed=true 正确 PASS. precedent_override 默认 forbidden, allowed 必须 owner verdict 批准. 进入 Gate 2 越闸计数暴露.