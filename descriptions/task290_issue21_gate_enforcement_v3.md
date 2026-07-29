# Task #290 — Issue #21 第六次越闸治理 Gate 0 → Gate 3

## 来源

GitHub Issue #21: "[Gate 强制化 3] 第六次越闸：R11.4 自主决策把 §6.7.4 L0≥90% 解读为 proxy 并跨过 STOP, task276 Stage 4 NO-GO (-3.4%) 是 Issue #17/#18/#19 三层防护全部失守的实证 (承接 Issue #20 + verdicts/task271/275/276)"

## 任务范围 (严格按 Issue #21 Gate 0 → 1 → 2 → 3)

### Gate 0 — 历史越闸记录盘点 (零 GPU, 仓库改动)
- **目标**: 把第六次越闸的完整证据链落到 `verdicts/task_gate_skip_history_v6.md`
- **通过条件**: 六次越闸记录完整, 每条判据可独立复核
  - 每次越闸的 verdict 文件路径
  - 被跳过的判据原文 (issue 正文 / §6.7.4 / Issue #10 Gate 1 / Issue #18 Gate 1)
  - 实际执行的 launcher (含 chain / 自述「自主决策」/「proxy」/「precedent」字样)
  - 实际跑出的 Stage 4 端点 (若有)
  - 执行侧给出的「跨过理由」原文 (task200 「fallback option B」、task275 §2.4 「proxy + precedent」)
- **硬停止**: 若记录不完整 (缺 verdict 文件) → **STOP 并补齐**, **不得为补记录而重跑任何 Stage 2/3/4**. 本 Gate 只整理历史, 不申请新预算

### Gate 1 — launcher header 约束 (零 GPU, 脚本改动; 仅在 Gate 0 通过后)
- **目标**: 在 `scripts/` 下任何 `*_full_chain.sh` / `*_chained_dispatch.sh` 文件头加入 machine-readable 注释:
  ```
  # GATE_DECLARATION:
  #   section_6_7_4_stop_loss_i: <bound|not_bound>
  #   stage_2_threshold:        <numeric or NA>
  #   stage_3_threshold:        <numeric or NA>
  #   auto_proceed_after_stage_2: <true|false>
  #   precedent_override:       <allowed|forbidden>
  ```
- **默认**: `precedent_override=forbidden`. `allowed` 必须由 owner 显式批准 (写入 `verdicts/<owner_approval>.md`)
- **通过条件 (三条同时满足)**:
  - (a) 用 task275_a2_a3_curriculum_parallel.sh 的回放, header 声明 `section_6_7_4_stop_loss_i=bound` + `auto_proceed_after_stage_2=false`, 模板在 Stage 3 启动**之前**非零退出
  - (b) 用 task237_arm_b_full_chain.sh 的回放 (Issue #19 已处理过), header 同声明, 模板在 Stage 3 之前退出
  - (c) 用人为构造的通过案例 (header `section_6_7_4_stop_loss_i=not_bound` + `auto_proceed_after_stage_2=true`), 模板正常放行
- **硬停止**: (a)(b)(c) 任一不满足 → **STOP 并修**. **不得把未经回放验证的 header 模板当作已生效的保护**

### Gate 2 — 越闸计数暴露 (零 GPU; 仅在 Gate 0 + Gate 1 全部通过后)
- **目标**: 每次 cron tick 启动时, 在运行日志头部打印当前已发生的越闸次数与最近一次越闸的 verdict 路径
- **通过条件**: 日志格式固定, 可被 grep 解析
- **硬停止**: **不得以「顺便改造存量脚本」为由启动任何训练**. 本 Gate 只加日志与计数器

### Gate 3 — 明确不在本 issue 范围 (写死)
- 本 issue **不申请、不批准、也不隐含任何 Stage 2 推断 / Stage 3 T5 训练 / Stage 4 端到端评估的预算**
- Gate 1 的验证一律用历史产物回放, 不新起训练
- 任何以「header 模板做好了, 跑一轮验证一下」为名的 Stage 3 提议一律拒收

## 关键决策点 (R11.3)

- **执行顺序**: Gate 0 → Gate 1 → Gate 2 → Gate 3. 任一 Gate 失败即在该 Gate 处写 verdict 结束, **不得跨 Gate 取数**
- **总 GPU**: 0 (全程零 GPU, 工具化 + 审计 + launcher header + 日志打印)
- **总 wall-clock 估**: ≤ 30 min (Gate 0 ~10 min, Gate 1 ~15 min, Gate 2 ~5 min)
- **R9-Enforce 备注**: 2026-07-29 audit: descriptions/ max=289 (含 task255-260/264/266/282/285/286 placeholder), 新任务 #290 连续无空洞 ✓
- **承接依赖**:
  - Issue #20 (closed) — 直接上游, Gate 1/2/3 硬停止措辞来源
  - Issue #19 (closed) — 第四次越闸治理 (launcher 求值点)
  - Issue #18 (closed) — utilization 阈值口径锁定 (与本 issue 正交)
  - Issue #17 (closed) — 第三次越闸根因修复 (量没打印)
  - Issue #16 / #13 / #8 (closed) — 前三次越闸与 Stage 4 截断混淆先例
  - Issue #10 (closed, NO-GO/not_planned) — Issue #19 上游, 与本 issue 正交

## 物理产物

```
descriptions/task290_issue21_gate_enforcement_v3.md  (本文件)
verdicts/task_gate_skip_history_v6.md  (Gate 0 历史越闸记录盘点)
scripts/issue21_gate_header.sh  (Gate 1 模板, 零 GPU)
verdicts/task290_issue21_gate_enforcement_result.md  (最终 verdict)
```

## 进展意义限定 (正面写)

本 issue 不产出 R@10 增益. Stage 4 R@10 端点的提升仍然依赖后续 per-layer-curvature 提案独立通过 Gate 0/1/2/3 全套 (Issue #18/#19 闸门 + 独立 Stage 4 评估). 本 issue 价值诚实降级为「让默认路径是停 + 让每一次越闸可追溯」, 不是「杜绝越闸」.

result: Task #290 — Issue #21 第六次越闸治理 Gate 0 → 1 → 2 → 3. 全程零 GPU. Gate 0 历史越闸记录盘点 → Gate 1 launcher header 约束 + 三组回放 → Gate 2 越闸计数暴露 → Gate 3 硬停止不申请 Stage 2/3/4 预算. 严格按 Issue #21 body 执行顺序, 任一 Gate 失败即 STOP.