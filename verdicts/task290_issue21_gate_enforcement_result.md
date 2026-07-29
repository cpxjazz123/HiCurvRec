# Task #290 — Issue #21 第六次越闸治理 (Gate 0 → 1 → 2 → 3 全部闭环)

> **完成日期**: 2026-07-29
> **状态**: ✅ **Issue #21 全部闭环 — Gate 0/1/2/3 PASS, 全程零 GPU**
> **核心结论**: Issue #21 治理闭环完成. 三层防护 (Issue #17 trainer fix / Issue #18 口径锁定 / Issue #19 launcher 求值点) 全部失效的最后根因 (R11.4 proxy + precedent 跨过) 由 launcher header 约束 + precedent_override=forbidden 默认 + 越闸计数暴露 三件套终结. 跨过理由分类 6 类全部闭环: fallback / 当场 GO / 量没打印 / launcher 没求值点 / 量测法歧义 / proxy+precedent (本 issue 修复).

---

## 1. Gate 总览

| Gate | 状态 | 产物 | 通过条件 | 实际 |
|------|------|------|---------|------|
| **Gate 0** | ✅ CLOSED | `verdicts/task_gate_skip_history_v6.md` | 六次越闸记录完整, 每条判据可独立复核 | ✅ 6/6 记录完整, 跨过理由可追溯 |
| **Gate 1** | ✅ CLOSED | `scripts/issue21_gate_header.sh` + 3 launcher headers | (a)(b)(c) 三组回放同时达标 | ✅ (a) FAIL 正确 / (b) PASS 正确 / (c) PASS 正确 |
| **Gate 2** | ✅ CLOSED | `scripts/issue21_skip_counter.sh` + template hook | 日志格式固定, 可被 grep 解析 | ✅ `[Issue #21 Skip Counter] key=value` 格式, grep 验证 PASS |
| **Gate 3** | ✅ 写死 | 全程零 GPU | 不申请 Stage 2/3/4 预算 | ✅ 三 Gate 全部用历史产物回放, 无 Stage 2/3/4 启动 |

---

## 2. 物理产物总清单

```
descriptions/task290_issue21_gate_enforcement_v3.md  (任务定义, Issue #21 Gate 0 → 3)
verdicts/task_gate_skip_history_v6.md                (Gate 0 历史越闸记录盘点)
verdicts/task290_issue21_gate1_result.md             (Gate 1 launcher header 约束闭环)
verdicts/task290_issue21_gate2_result.md             (Gate 2 越闸计数暴露闭环)
verdicts/task290_issue21_gate_enforcement_result.md  (本文件, Issue #21 整体闭环)
scripts/issue21_gate_header.sh                       (Gate 1 模板)
scripts/issue21_skip_counter.sh                      (Gate 2 计数器)
scripts/task275_a2_extend.sh                         (修改: 加 GATE_DECLARATION)
scripts/task237_arm_b_full_chain.sh                  (修改: 加 GATE_DECLARATION)
scripts/task290_pass_case.sh                         (新增, Gate 1 (c) 人为通过案例)
logs/issue21_skip_counter.log                        (Gate 2 append-only 日志)
```

---

## 3. 跨过理由分类闭环 (Issue #21 Gate 0 证据)

| 跨过理由类型 | 出现次数 | 对应越闸次序 | 治理手段 | 闭环状态 |
|--------------|---------|-------------|---------|---------|
| 判据有, 被以 "fallback" 名义跨过 | 1 (task200) | 1st | Issue #19 + Issue #21 | ✅ 已闭环 |
| 判据有, 当场触发仍判 GO | 1 (task222) | 2nd | Issue #19 + Issue #21 | ✅ 已闭环 |
| 判据的量根本没打印 | 1 (Issue #13) | 3rd | Issue #17 | ✅ 已闭环 |
| 判据可求值, launcher 没求值点 | 1 (task237) | 4th | Issue #19 | ✅ 已闭环 |
| 判据的量测法本身有歧义 | 1 (Issue #18) | 5th | Issue #18 | ✅ 已闭环 |
| 判据已严格化, 执行侧仍以 "proxy + precedent" 跨过 | 1 (Issue #20) | 6th | **Issue #21 (本 issue)** | ✅ 已闭环 |

**结论**: 6 类跨过理由全部有对应 Issue 闭环. Issue #21 是治理层的最后一根钉子, 把前 5 类工具修 (Issue #17/#18/#19) 没办法触及的「执行侧仍以 R11.4 自主决策权跨过闸门」类根治.

---

## 4. 关键决策点 (R11.3 + R11.5 全程透明记录)

| # | 决策 | 选择 | 备选 | 理由 |
|---|------|------|------|------|
| 1 | Issue #21 治理方向 | ✅ launcher header 约束 + precedent_override=forbidden | 增加 R10 主动治理 / 改 CLAUDE.md | launcher header 是治理层可执行的最强约束, 改 CLAUDE.md 容易被忽略 |
| 2 | precedent_override 默认值 | ✅ forbidden (R11.2 保守兜底) | allowed (R11.4 自主决策权) | Issue #21 治理方向 = 默认禁止, 显式批准才放开 |
| 3 | 模板放置 | ✅ 独立 `scripts/issue21_gate_header.sh` | 合并到 issue19_gate_template.sh | Issue #21 是独立治理方向 (precedent_override), 不污染 Issue #19 修复 |
| 4 | L0 threshold 默认值 | ✅ 90% (§6.7.4 stop-loss (i) 原文) | 95% / 85% | 严格按 paper.md §6.7.4 阈值, 不偷换 |
| 5 | epoch 最小值 | ✅ 30 (warmup 完成) | 0 / 10 | 训练早期 L0 不稳定, 30 epoch 后才有比较意义 |
| 6 | precedent_override=allowed 的 owner approval 校验 | ✅ grep launcher 文件名 (in verdicts/) | 单独 owner_approvals/ 目录 | 跟现有 verdicts/ 结构对齐, 不引入新目录 |
| 7 | 合成 baseline log 用例 | ✅ 创建 `logs/issue21_replay_b_synthetic_baseline_log.log` | 仅用 task270 A1 log | task270 A1 是 β=0 欧氏, 跟 baseline 性质不同; 合成 log 显式说明 L0=100% |
| 8 | Gate 2 hook 位置 | ✅ 模板顶部 + 独立脚本双 hook | 仅 cron tick 启动 | 双保险: cron tick 启动 + 任何闸门求值都 emit, 覆盖更广 |
| 9 | Gate 2 日志格式 | ✅ key=value 形式 | JSON / free-form | grep 正则友好, 简洁可读 |
| 10 | Gate 2 总数计数源 | ✅ verdicts/task_gate_skip_history_v6.md 汇总表 | 扫所有 verdicts/ 找 "越闸" 关键词 | Gate 0 闭环产物是真源, 跨引用稳定 |
| 11 | Gate 3 实施 | ✅ 写死 (本 issue 全程零 GPU) | 主动终止某些任务 | Issue #21 body 明确禁止以 "header 模板做好了, 跑一轮验证一下" 为名启动 Stage 3 |
| 12 | 不修改 cron 配置 | ✅ Issue #21 body 范围 = 「运行日志头部打印」 | 在 .claude/hooks/ 加 hook | R13 禁止 worktree; cron 配置修改超出 Issue #21 body 范围 |

---

## 5. 物理验证结果

### 5.1 Gate 1 三组回放

| 回放 | 输入 | 期望 | 实际 |
|------|------|------|------|
| (a) task275_a2_extend.sh | task270 A2 log (L0=89%) | exit 1 (FAIL) | ✅ exit 1 |
| (b) task237_arm_b_full_chain.sh | 合成 baseline log (L0=100%) | exit 0 (PASS) | ✅ exit 0 |
| (c) task290_pass_case.sh | not_bound + auto_proceed=true | exit 0 (PASS) | ✅ exit 0 |

### 5.2 Gate 2 grep 解析

```
$ bash scripts/issue21_skip_counter.sh
[Issue #21 Skip Counter] tick_id=20260729_200739 timestamp=2026-07-29 20:07:39 total_skips=6 last_verdict=verdicts/task275_a2_a3_curriculum_parallel_result.md gate0=CLOSED gate1=CLOSED

$ grep -oE 'total_skips=[0-9]+' logs/issue21_skip_counter.log | tail -3
total_skips=6
total_skips=6
total_skips=6
```

✅ 格式固定, grep 解析 PASS

### 5.3 Gate 3 硬停止

| 项 | 状态 |
|----|------|
| Stage 2 推断启动次数 | 0 |
| Stage 3 T5 训练启动次数 | 0 |
| Stage 4 端到端评估启动次数 | 0 |
| 任何 GPU kernel 启动次数 | 0 |

✅ 全程零 GPU, Gate 3 硬停止达成

---

## 6. 后续 backlog (R10 推进)

### 6.1 已闭环 (本 issue 内)
- ✅ Gate 0: 历史越闸记录盘点
- ✅ Gate 1: launcher header 约束 + 三组回放
- ✅ Gate 2: 越闸计数暴露
- ✅ Gate 3: 硬停止

### 6.2 不在 Issue #21 body 范围 (out-of-scope, backlog 候选)
- **越闸自动检测 + 阻断**: 任何后续实验启动时, 自动检查实验本身的 §6.7.4 stop-loss (i) 触发状态, 触发则直接 abort. 这是 Issue #21 的自然延伸, 但 Issue #21 body 没要求, 不在本次范围
- **precedent_override=allowed 的 owner approval workflow**: 当前只支持 grep 校验, 没有专门 owner_approvals/ 目录 + 显式批准 workflow. 这是治理改进项, 不在本次范围
- **跨 task290 的越闸追溯链**: 当前 total_skips=6 是静态值, 直到 v7+ 出现. owner 提交新越闸证据 → AI 写历史 → counter 自动增长, 这是 owner / AI 双人舞. 自动扫描所有 verdicts/ 找 "越闸" 关键词 改进计数源, 不在本次范围
- **D-curriculum / D-结构改动 (task272 m-arm κ-Stereographic v9+)**: 仍然 backlog 唯一候选, **不主动启动** (R11.4 critical decision 需要明确批准)

### 6.3 用户主动启动 (监测不干预)
- **Task #291 EMA codebook + κ 验证**: Stage 1 已完成, Stage 2/3/4 未启动 (无 chain waiter). 不主动启动, 等用户指示
- **Task #292 Restoration + EMA + κ 验证**: 同上

---

## 7. R2 KB 更新 (后续 backlog 引用)

- **Issue #21 是治理层最后一根钉子**: 跨过闸门的 6 类根因全部闭环. 任何后续越闸需要先更新 `verdicts/task_gate_skip_history_v6.md` (或 v7+), total_skips 才会变化. owner 提交越闸证据 → AI 写历史 → counter 自动增长
- **Issue #21 模板 ≠ 替代 Issue #19 模板**: 两个模板应该串联 (`source issue19_gate_template.sh && source issue21_gate_header.sh && evaluate_stage_2_gate ... && parse_gate_declaration ...`). 任何链式 launcher 应该**双 hook**, 不是替换
- **precedent_override=forbidden 是默认值**: 任何 launcher 加 GATE_DECLARATION 时, `precedent_override=allowed` 必须有 `verdicts/<owner_approval>.md` 引用本 launcher. 不允许在 launcher 里直接写 `precedent_override=allowed` 而没有 owner verdict 批准
- **Issue #21 价值诚实降级**: 本 issue 不产出 R@10 增益. Stage 4 R@10 端点的提升仍然依赖后续 per-layer-curvature 提案独立通过 Gate 0/1/2/3 全套. Issue #21 价值是「让默认路径是停 + 让每一次越闸可追溯」, 不是「杜绝越闸」(杜绝越闸需要 owner override 流程 + 经济激励, 不在工具层)

---

## 8. Commit 记录

| Commit | 内容 |
|--------|------|
| eb11744 | Gate 0 历史越闸记录盘点 |
| 40f4362 | Gate 1 launcher header 约束 + 三组回放 PASS |
| 008f18c | Gate 2 越闸计数暴露 + grep 解析 PASS |

---

result: Task #290 — Issue #21 第六次越闸治理 **全部闭环 (Gate 0/1/2/3 PASS, 全程零 GPU)**. 三层防护 (Issue #17/#18/#19) 全部失效的最后根因 (R11.4 proxy + precedent 跨过) 由 launcher header 约束 + precedent_override=forbidden 默认 + 越闸计数暴露 三件套终结. 跨过理由分类 6 类全部闭环. 后续 backlog: 用户主动启动的 Task #291/292 (EMA + Restoration) 监测不干预, D3 (task272 m-arm κ-Stereographic v9+) 不主动启动 (R11.4 critical decision).
