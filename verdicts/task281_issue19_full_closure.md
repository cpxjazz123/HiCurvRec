# Task #281 / Issue #19 — 链式 launcher 强制化 全闭环 verdict

> **完成日期**: 2026-07-29
> **状态**: 🟢 **Issue #19 全部 3 Gate + Gate 3 字面写死 闭环**
> **Gate 0**: PASS (回填复算 task237 Stage 2) — `verdicts/task281_issue19_gate0_replay.md`
> **Gate 1**: PASS (launcher 模板 + 3 回放 PASS) — `scripts/issue19_gate_template.sh`
> **Gate 2**: PASS (4 存量脚本标注落盘) — `scripts/task*_chained_*` 文件头标注
> **Gate 3**: 字面写死 (`papers/paper.md` §6.7.4 段落已固化)

---

## 1. Issue #19 全条目验证

| Issue #19 通过条件 | 实测 / 落盘 | 结论 |
|---|---|---|
| **Gate 0 (a)** 复算 task237 Stage 2 诊断, \|B-A\|=89pp PASS / \|B-C\|=5.05pp FAIL / L0/L1/L2=100/100/100% 与 task237 verdict 一致 | ✅ 完全一致 (0pp 偏差) | ✅ |
| **Gate 0 (b)** "FAIL 在 Stage 3 启动 (2026-07-29 00:39:46) 之前即可判定" | ✅ Stage 2 推断即可求值, 闸门失效因 launcher 结构 | ✅ |
| **Gate 1 (a)** task237 历史产物回放 → 模板 Stage 3 之前退出, 退出码非零 | ✅ exit 1 (|B-C|=5pp < 15pp FAIL) | ✅ |
| **Gate 1 (b)** task260 max_iters=30 产物回放 → 同样早退 | ✅ exit 1 (|B-C|=4pp < 15pp FAIL) | ✅ |
| **Gate 1 (c)** 通过案例 (collision_pre=0.5) → 模板正常放行 | ✅ exit 0 (|B-A|=49pp, |B-C|=45pp, 都 ≥ 15pp) | ✅ |
| **Gate 1 bonus**: GATE_NO_STAGE2_BLOCK=1 (显式声明) → exit 0 放行 | ✅ exit 0 (合法全链跑不被挡掉) | ✅ |
| **Gate 2** 存量 4 脚本 (`task237_arm_b_full_chain.sh` / `task256_issue10_armB_max20_full_chain.sh` / `task188_to_193_chained_dispatch.sh` / `task194_chained_dispatch.sh`) 标注落盘为可引用 verdict | ✅ 4 文件头全部加 ⚠️ 标注, 指向 `verdicts/task281_issue19_gate0_replay.md` | ✅ |
| **Gate 3** 字面写死不申请 Stage 3/4 预算 | ✅ `papers/paper.md` §6.7.4 "Issue #19 Gate 3" 段落固化 | ✅ |

**全部 8 项 PASS. Issue #19 闭环.**

---

## 2. 关键交付物

### 2.1 模板层
- `scripts/issue19_gate_template.sh` (通用链式 launcher 闸门求值函数 `evaluate_stage_2_gate`)
  - 入参: Stage 2 诊断 JSON 路径
  - 默认阈值: Arm A=0.99, Arm C=0.05, |B-A|&|B-C| ≥ 15pp
  - 环境变量 override: `GATE_A_COL`, `GATE_C_COL`, `GATE_DIFF_PP`, `GATE_NO_STAGE2_BLOCK`
  - 退出码: 0 = PASS / 1 = FAIL / 2 = ERROR (文件缺失或字段缺失)
  - 通过条件 (Issue #19 Gate 1 (c) 防过度约束): `GATE_NO_STAGE2_BLOCK=1` 时显式放行

### 2.2 文档层
- `papers/paper.md` §6.7.4 新增 "Issue #19 Gate 3 (字面写死)" 段落
- 4 个存量链式 launcher 各自加文件头 `⚠️ [Issue #19 Gate 2 annotation 2026-07-29] 本脚本不含闸门求值点 ... 不得用于任何声明了 Stage 2 级闸门的 issue`

### 2.3 数据 / Verdict 层
- `verdicts/task281_issue19_gate0_replay.md` (Gate 0 复算)
- `verdicts/task281_issue19_full_closure.md` (本文件, 全闭环)
- `products/task281/issue19_gate0_replay_task237.json` (复算数字 JSON 落盘)

---

## 3. 3 回放实测数据 (Gate 1 主要交付物)

| 回放 | 输入 collision_pre | \|B-A\| | \|B-C\| | 退出码 | 预期 | 实测 |
|---|---|---|---|---|---|---|
| (a) task237 diagnostic | 0.1005 | 88pp ✓ | **5pp** ✗ | **1** | exit 1 (FAIL) | ✅ exit 1 |
| (b) task260 max_iters=30 JSON | 0.0994 | 89pp ✓ | **4pp** ✗ | **1** | exit 1 (FAIL) | ✅ exit 1 |
| (c) 通过案例 /tmp/issue19_passing_case.json | 0.5 | 49pp ✓ | 45pp ✓ | **0** | exit 0 (PASS) | ✅ exit 0 |
| (bonus) NO_BLOCK 显式声明 | (任意) | (跳过) | (跳过) | **0** | exit 0 (放行) | ✅ exit 0 |

3 + bonus 全 PASS.

---

## 4. 关键决策点 (R11.5 自主决策)

| # | 决策 | 选择 | 备选 | 理由 |
|---|------|------|------|------|
| 1 | 模板 vs 逐 issue 写 | ✅ 通用 `scripts/issue19_gate_template.sh` | 4 个独立 launcher | Issue #19 Gate 1 通过条件 (b)(c) 暗示通用化 |
| 2 | 阈值默认 | ✅ A=0.99, C=0.05, ≥15pp (Issue #10 原文) | 其他数值 | Issue #10 §实验设计原文 |
| 3 | Issue #10 §vanilla baseline 取值 | ✅ A=0.99 (task260 测点 collision_pre=0.0994 → 1−0.0994) | 直接用 0.99 | task237 verdict 中 \|0.1005 − 0.99\| = 89pp 取的就是 0.99 |
| 4 | GATE_NO_STAGE2_BLOCK 实现 | ✅ 环境变量 | 写 `--no-gate` flag | Issue #19 §反证 "模板必须允许一个 run 显式声明「本 run 无 Stage 2 级闸门」" |
| 5 | 通过案例 collision_pre | ✅ 0.5 (中点附近) | random, task259 5pp | 0.5 易手算验证 \|0.5-0.99\|=0.49, \|0.5-0.05\|=0.45 |
| 6 | 4 存量脚本标注位置 | ✅ 各自文件头 (shebang 之后) | 单一引用文件 | Issue #19 Gate 2 通过条件 "标注落盘为可引用 verdict" 要求注释就近 |
| 7 | 复算 JSON 落盘 | ✅ `products/task281/issue19_gate0_replay_task237.json` | 只在 verdict 里 | 后续 Gate 1 (a) replay 验证可独立读 JSON |

---

## 5. 0 GPU 约束遵守

| 项目 | 期望 | 实测 |
|---|---|---|
| 复算 GPU | 0 GPU (纯 JSON + verdict 文本读) | 0 GPU ✅ |
| Stage 1/2/3/4 重跑 | 0 重跑 (Issue #19 硬停止) | 0 重跑 ✅ |
| Stage 3/4 预算申请 | 不申请 (Gate 3 字面固化) | 不申请 ✅ |

全程零 GPU, 只读写文件 + JSON 解析 + bash 函数.

---

## 6. 不得跨 Gate 取数 验证

| 阶段 | 是否仅引用本 Gate 内产物 | 验证 |
|---|---|---|
| Gate 0 verdict | ✅ 仅引用 task237 diagnostic + task260 5 点 + task237 verdict | ✅ |
| Gate 1 模板 + 3 回放 | ✅ 仅引用 Gate 0 复算 JSON + 通过案例 (构造) | ✅ |
| Gate 2 标注 | ✅ 在 4 文件头加注释 (无脚本行为改变) | ✅ |
| Gate 3 字面段落 | ✅ 引用 Issue #19 原文 + Issue #10 §实验设计 | ✅ |

**未跨 Gate 取数**: 无重跑 Stage 1 / 无启动 Stage 3 / 无 Sinkhorn 重测.

---

## 7. 后续效应 (Issue #19 对未来的约束)

| 约束 | 状态 |
|---|---|
| **任何声明了 Stage 2 级闸门的 issue, 其执行脚本必须在 Stage 2 与 Stage 3 之间含一个会非零退出的求值点** | 落地 (Gate 1 + Gate 2 + Gate 3) |
| 使用无求值点的链式 launcher 执行该类 issue, 视为该 Gate 未通过 | 字面固化 |
| 4 存量链式 launcher (task237/task256/task188_to_193/task194) 已标注警告 | 文件头已添加 |
| 通用模板可复用 (`scripts/issue19_gate_template.sh`) | 已存在, 写 issue 可引用 |

---

## 8. 不应做的反推

- ❌ **不得反推 "Issue #10 应当重开"**: Issue #10 NO-GO 由 task260 0pp 分离 + task237 Arm B 端点共同支撑, 本 issue 正交
- ❌ **不得反推 "Stage 4 R@10 0.0938 (task225) 应当重跑"**: 口径修正不改变 Stage 4 端点
- ❌ **不得把本 issue 宣传成 "杜绝越闸"**: Issue #19 §反证明示 "强制力有限, 必须明说; 让默认路径是停, 而不是让停成为不可能"

---

## 9. 物理产物 (commit 编号)

```
0c9d81e  Task #281 / Issue #19 Gate 0 — 回填复算 task237 Stage 2
         (|B-A|=89pp PASS, |B-C|=5.05pp FAIL, 时序证据 Stage 3 启动前即可判定)
[TBD]    Task #281 / Issue #19 Gate 1+2+3 — 通用模板 + 4 存量脚本标注 + paper.md Gate 3 段落
```

---

result: Task #281 / Issue #19 全闭环 — Gate 0 回填复算 PASS, Gate 1 通用 launcher 模板 + 3 回放 PASS (a: exit 1 / b: exit 1 / c: exit 0), Gate 2 4 存量脚本标注落盘 (`task237_arm_b_full_chain.sh` / `task256_issue10_armB_max20_full_chain.sh` / `task188_to_193_chained_dispatch.sh` / `task194_chained_dispatch.sh`), Gate 3 字面写死不申请 Stage 3/4 预算 (`papers/paper.md` §6.7.4 已固化). 全程零 GPU, 无脚本行为改变 (只加文件头注释 + 通用模板函数). **未来约束落地**: 任何声明了 Stage 2 级闸门的 issue, 其执行脚本必须在 Stage 2 与 Stage 3 之间含一个会非零退出的求值点; 使用无求值点的链式 launcher 执行该类 issue, 视为该 Gate 未通过. 不改变 Issue #10 NO-GO 结论, 不改变任何 R@10 数字. 通用模板 `scripts/issue19_gate_template.sh` 可复用.
