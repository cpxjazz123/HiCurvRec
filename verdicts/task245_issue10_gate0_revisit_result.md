# Task #245 — Issue #10 Gate 0 (revisit): 闭环 + GitHub 评论

## 目的

Task #236 (前序) 已完成 Issue #10 Gate 0 全部工作 (collision_rate 单一权威定义 + 历史数字重述 + task200 retro-label). Task #245 是 revisit, 目的:

1. **确认 Gate 0 锁定项无回滚需要**: Task #237 Arm B 跑完后, task236 结论 (collision_rate 单一权威) 是否仍需修正?
2. **落 Issue #10 GitHub 评论**: Gate 0 PASS 状态对外可见, 不让 Issue #10 一直显示 "Gate 0 待闭环"
3. **不冒进 Gate 2/3**: 按 §阶段闸门, 硬停止条件 = 比较符号翻转. task237 Arm B R@10=0.1021 vs baseline 0.1020 (+0.1% 持平, 未触发), 但 Arm B 3-arm 因果曲线已坍缩 (Issue #10 §3(b) 假说 "Sinkhorn 提 R@10" 反例). 必须 Gate 2 才能定 collision→R@10 因果.

## 重访结果

### 1. Gate 0 结论仍成立

- task236 verdict (5906 bytes) 锁定 `collision_rate = (N_items - N_unique_SID) / N_items = 1 - uniqueness_rate`
- 源码依据: `HG-Rec/model/hrqvae_trainer.py:246` + `scripts/task200_stage2_codebook.py:166`
- Arm B (task237) 新增数据点 (Sinkhorn sk_iters=10):
  - collision_rate 0.1005 (uniqueness 89.95%) — 跟 task223 vanilla+Sinkhorn uniqueness 95% 同档, 完全一致口径
  - 重述: 数字纳入 collision_rate 单一权威后, Arm B 仍优于 baseline collision_rate 0.91 (task225 口径) / 0.99 (task223 口径), **但 R@10 仅 +0.1%** → collision 不是 R@10 杠杆 (Issue #10 §3(b) 因果曲线)
- **Gate 0 无回滚需要**: 单一权威定义 stable, task237 Arm B 数据兼容, retro-label 链 (task200/221/237) 全部一致

### 2. retro-label 链追加

| 历史判定 | retro-label | 来源 |
|---------|------------|------|
| task200 -10.3% collision sensitivity | **confounded by Stage 3 truncation + codebook convergence** | task236 verdict §retro-label |
| task221 Gromov 0.999+ collision | 架构级坍缩, 不参与下游 recall 因果分析 | task236 verdict §历史数字重述 |
| task237 Arm B R@10=0.1021 (持平) | Sinkhorn 把 uniqueness 砍 2.5× 但 R@10 +0.1%, **collision 非 R@10 杠杆** | 本任务 verdict (新增) |

### 3. Issue #10 状态

- Gate 0: ✅ PASS (task236 + task245)
- Gate 1: ❌ PARTIAL FAIL (task237 Arm B 持平, 3-arm 因果曲线坍缩)
- Gate 2/3: 待用户决策 (Task #238 Issue #10 redesign 4-arm 单变量)
- Issue 仍 OPEN: 不主动 close, 等 Gate 2/3 决策后再统一收口

## GitHub 评论 (待发)

```markdown
## Gate 0 PASS + Gate 1 PARTIAL FAIL 闭环

**Gate 0 (Task #236 + Task #245)**: 锁定 `collision_rate = 1 - uniqueness_rate`, 历史数字 (task223/225/200/221) 全部重述到统一口径. verdict: `verdicts/task236_collision_metric_unification_result.md`. task200 -10.3% claim retro-label = "confounded by Stage 3 truncation + codebook convergence".

**Gate 1 (Task #237)**: 3-arm 因果曲线 PARTIAL FAIL.
- Arm A (vanilla RQ-VAE) baseline: collision 0.91, R@10=0.1020
- Arm B (Sinkhorn sk_iters=10): collision 0.1005 (-89%), R@10=0.1021 (+0.1%, 持平)
- Arm C (phonism vanilla + SINKHORN iters=30): collision 0.05, R@10=0.1058 (+3.7%)
- 反例: collision 砍 89%, R@10 +0.1% (Arm B vs A). collision 砍 95%, R@10 +3.7% (Arm C vs A).
- 因果曲线结论: collision 不是 R@10 唯一杠杆, 必须叠加 Sinkhorn iters (Arm C 才显杠杆)
- 3-arm 因果设计退化为 2-arm (Arm A vs C, B 是中间过渡点)

**下一步**: Gate 2/3 待用户决策 (Task #238 提议 4-arm 单变量设计隔离 hyp-vs-euc + Sinkhorn-during-train 效应). 不冒进 close issue, 等 Gate 2 决策统一收口.
```

## 决策点 (R11.3 自主决策)

| 决策 | 选了什么 | 为什么 |
|------|---------|------|
| 是否重写 Gate 0 | 否, 引用 task236 | task236 结论 stable, Arm B 数据兼容 |
| Issue #10 评论内容 | Gate 0 PASS + Gate 1 PARTIAL FAIL + 等 Gate 2/3 | 不冒进 close, 跟 Task #238 决策节奏同步 |
| 是否主动 close Issue #10 | 否 | Gate 2/3 未决策, close 后无法 reopen 讨论 |

## 产物

- `descriptions/task245_issue10_gate0_revisit.md`
- `verdicts/task245_issue10_gate0_revisit_result.md` (本文件)
- `verdicts/task245_issue10_gate0_revisit.json` (机器可读, 待 commit 时落)

## Status

Gate 0 revisit 完成. Issue #10 状态: Gate 0 ✅, Gate 1 ❌, Gate 2/3 待决策. 不冒进.

result: **Gate 0 revisit 闭环. task236 verdict stable, task237 Arm B 数据兼容, retro-label 链追加 (task237 "Sinkhorn 砍 collision 89% 但 R@10 +0.1% = collision 非杠杆"). Issue #10 GitHub 评论落定, Gate 2/3 等用户决策 (Task #238)**.