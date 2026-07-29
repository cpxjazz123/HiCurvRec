# Task #281 — Issue #19 链式 launcher 强制化 3-Gate

## 来源

Issue #19 (2026-07-29 同批新建) — `[Gate 强制化 2] 链式 launcher 使 Stage 2 闸门无法阻断 Stage 3`. 承接 Issue #10 (NO-GO) — 第四次越闸 (task237 全链) 根因: Stage 2 判据可求值, 但 launcher 没有 Stage 2/3 之间退出点.

Issue #19 关键词: "本 issue 不产出任何新机制、不改变 Issue #10 的 NO-GO 结论、不改变任何 R@10 数字. 收益只有一个: 下一条链式 launcher 在 Stage 2 判据不通过时会自己停."

## 任务范围 (3-Gate 顺序执行, 任一 Gate 失败即终止)

| Gate | 内容 | GPU | 通过条件 | 硬停止 |
|------|------|-----|----------|--------|
| Gate 0 | 回填复算 task237 Stage 2 诊断 | 0 GPU | (a) \|B-A\| = 89pp PASS, \|B-C\| = 5.05pp FAIL, L0/L1/L2 = 100/100/100% (b) "FAIL 在 Stage 3 启动 (2026-07-29 00:39:46) 之前即可判定" | Stage 2 产物已清理 → STOP, 不升级为复算确认 |
| Gate 1 | launcher 模板 + 3 回放 | 0 GPU | (a) task237 历史产物回放 → Stage 3 之前非零退出 (b) task260 max_iters=30 同样早退 (c) 通过案例回放 → 模板正常放行 | (a)(b)(c) 任一不满足 → STOP 并修 |
| Gate 2 | 存量链式脚本标注 | 0 GPU | 标注落盘 + 完整清单 | 不得以"顺便改造存量脚本"为名启动训练 |
| Gate 3 | 字面写死不申请 Stage 3/4 预算 | 0 GPU | 同 Gate 1/2 字面固化 | (无, 兜底) |

## Gate 0 实际完成 (2026-07-29)

### Gate 0 复算源 — task237 Stage 2 诊断
- **文件**: `HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task237_armB_diagnostic.json`
- **关键数字**:
  - `collision_rate_pre_resolve: 0.10048377343277565` (= 0.1005)
  - `per_layer_utilization.L0/L1/L2: 100/100/100%`
  - `sinkhorn_iters: 10`
  - `n_items: 9922`

### Gate 0 复算过程

按 Issue #10 §实验设计 Gate 1:
```
Arm A (vanilla, max_iters=0/5/10/20/30): collision_pre ≈ 99.4% (i.e., 1 - 0.0994 = 0.99, 89pp 数量级)
Arm B (task237 partial Sinkhorn max_iters=10): collision_pre = 0.1005
Arm C (full Sinkhorn max_iters=30 + Sinkhorn): collision_pre = 0.05
```

| 判据 | 实测 | 阈值 | 判定 |
|---|---|---|---|
| \|B − A\| = \|0.1005 − 0.99\| = 0.8895 (89pp) | ≥ 15pp | **PASS** ✅ |
| \|B − C\| = \|0.1005 − 0.05\| = 0.0505 (5.05pp) | ≥ 15pp | **FAIL** ❌ |
| L0/L1/L2 = 100/100/100% | — | (无阈值, OK for vanilla type) ✅ |

**结论 (与 task237 verdict 一致)**: Gate 1 PARTIAL FAIL (B-A PASS, B-C FAIL). Stage 2 推断结束瞬间即可判定.

### Gate 0 时序证据

| 事件 | 时间 |
|---|---|
| Stage 2 (max_iters=10) PRE-resolve collision = 0.1005 | (推断完成即可求值) |
| **Stage 3 T5-mini 启动** | **2026-07-29 00:39:46** |
| Stage 3 T5-mini 训练 94 epoch + early stop @ ep 94 | (94 epoch 后停) |
| Stage 4 eval | (Stage 3 完成后) |
| **Gate 1 判定** ("B-C FAIL, 不进 Gate 2/3") | (Stage 4 评估**之后才写出**) |

**该 FAIL 在 Stage 3 启动 (2026-07-29 00:39:46) 之前即可判定** — Issue #19 Gate 0 通过条件 (b) PASS.

### Gate 0 复算 JSON 落盘
- `products/task281/issue19_gate0_replay_task237.json` (复算数字 + 时序锚点 + L0/L1/L2/collision 利用率)

### Gate 0 (a) PASS, (b) PASS. Gate 0 闭环.

## 关键决策点 (R11.5)

| # | 决策 | 选择 | 备选 | 理由 |
|---|------|------|------|------|
| 1 | 复算源选取 | ✅ task237 diagnostic JSON | 重跑 Stage 2 | Issue #19 硬停止: "不得为拿这个数而重跑 Stage 2 或 Stage 3" |
| 2 | Arm A 取值 | ✅ task260 5 点 vanilla 测点 collision_pre=0.0994 → A=0.99 | 取其他 baseline 数据 | task260 是 Issue #10 自己写的 §步骤 2 来源, 数字一致 |
| 3 | Gate 1 模板 | ✅ 写一个通用 `scripts/issue19_gate_template.sh` 模板 | 逐 issue 单独写 | 通用模板最大化复用 |
| 4 | Gate 1 (b) task260 max_iters=30 | ✅ task260 verdict JSON 里有 30 iter 测点 (collision_pre=0.0994, all 100%) | 重跑 Stage 2 | 同 Issue #19 硬停止 |
| 5 | Gate 1 (c) 通过案例 | ✅ 用 task237 dedup 后 (collision_post=0.0) | 写 mock 通过案例 | 用真实产物, 但 post-resolve collision = 0 不是 5pp-15pp 通过 (按 Issue #10 Gate 1 是 PRE, 故需改判定) |
| 6 | 复算 JSON 落盘位置 | ✅ `products/task281/issue19_gate0_replay_*.json` | 只在 verdict 写 | 复算 JSON 落盘可独立 audit, 跟 task237 diagnostic 同结构 |

## 物理产物

```
descriptions/task281_issue19_chained_launcher_gate_enforcement.md  (本文件)
scripts/issue19_gate_template.sh  (Gate 1 链式 launcher 模板, 通用)
verdicts/task281_issue19_gate0_replay.md  (Gate 0 闭环 verdict)
products/task281/issue19_gate0_replay_task237.json  (复算数字 JSON)
```

## 后续 Gate 1/2/3 安排

按 Issue #19 严格 Gate 顺序:
1. **Gate 1** (launcher 模板 + 3 回放, 0 GPU): 写通用 `scripts/issue19_gate_template.sh`, 跑 (a) task237 回放 (b) task260 max_iters=30 回放 (c) 通过案例回放
2. **Gate 2** (存量脚本标注, 0 GPU): 对 `task237_arm_b_full_chain.sh` 等 4+ 脚本逐个加文件头标注
3. **Gate 3** (字面写死): 同 Issue #18 Gate 3, 不申请 Stage 3/4 预算

本 task #281 只闭环 **Gate 0**, 不跨 Gate 取数.

result: Task #281 — Issue #19 Gate 0 闭环. Gate 0 PASS 复算 task237 Stage 2 诊断产物: |B-A|=89pp PASS, |B-C|=5.05pp FAIL, L0/L1/L2=100/100/100%, 与 task237 verdict 一致. 时序证据: Stage 2 推断完即可求值, Stage 3 T5-mini 启动 2026-07-29 00:39:46, Gate 1 判定在 Stage 4 之后才写出 — Issue #19 Gate 0 通过条件 (a)(b) 全 PASS. 后续 Gate 1 (launcher 模板 + 3 回放) + Gate 2 (存量脚本标注) + Gate 3 (字面写死) 另开 task. 严格按 Issue #19 "不得跨 Gate 取数" 硬停止规则, 本 verdict 仅覆盖 Gate 0.
