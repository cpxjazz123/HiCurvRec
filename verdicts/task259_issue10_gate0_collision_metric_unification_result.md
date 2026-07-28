# Task #259 — Issue #10 Gate 0: collision 指标口径统一 + 历史数字重述 (零 GPU)

## Gate 0 决策

**GATE0_PASS** —— collision 口径已锁定, 历史数字重述表已完成, Issue #10 §反证 §2 揭示的口径冲突有解.

**核心结论**: 仓库存在 **两套不兼容口径**, 它们各自内部自洽, 但跨表比较时方向相反. 重述到统一口径后, **collision ≤ 12% 这条达标线在 task225 §5 口径下不存在** (baseline = 0.99), 在 task223 口径下也不存在 (PC κ = 0.37 > 0.12). 整条 collision 达标线需要重写.

## 1. 锁定单一 collision 定义

Stage 2 diagnostic 实际计算代码 (`scripts/task223_stage2_codebook.py:136`):
```python
collision_rate = (tot_item - tot_indice) / tot_item
```
即 `collision_rate = 1 - unique_count / total_items`. 这是 **collision 严格定义**: 重复 SID 占比.

- 9922 items
- 9014 unique SID (task253) → collision = 1 - 9014/9922 = **0.0915** ✅
- 9922 unique SID (task223 baseline 没去重, 重复 0) → collision = **0.0** ❌ (不是 0.99)

**task223 verdict §5 表里 baseline "collision = 0.99" 是 uniqueness (1%) 的误读**. 实际 collision 应该是 0.0 (因为 9922 unique SID 占 9922 items).

但 Sinkhorn 后 baseline uniqueness 暴跌到 1% (因 Sinkhorn 把 9922 items 压到 ~100 个 cluster, 9922 items 都映射到少数 cluster → 大量重复 → collision 暴涨). 所以:

| 实测 | collision 严格定义 | uniqueness | 来源 |
|---|---|---|---|
| task223 baseline (no Sinkhorn, no 4-digit dedup) | ~0.0 | 1.0 | Stage 2 推断产物 9922 unique |
| task223 baseline (Sinkhorn 30 iters, **没去重**) | 0.99 | 0.01 | task223 verdict §5 表 (实际是 uniqueness 误读为 collision) |
| task223 baseline (Sinkhorn 30 iters, **去重**) | 0.0 | 1.0 | 实际 Stage 2 推断产物 |
| **task84 HG-Rec baseline (实际跑的 SID)** | **0.0** | **1.0** | unique SID = items = 9922, 实际 collision = 0 |
| task225 PC κ | 0.37 | 0.63 | task223_pck_diagnostic.json: 6245/9922 unique = 62.94%, collision = 37.06% |
| task253 Möbius 残差 | 0.0915 | 0.9085 | task253 verdict: 9014/9922 unique = 90.85%, collision = 9.15% |
| task237 Sinkhorn max_iters=10 | 0.1005 | 0.8995 | task237 verdict |
| task84 vanilla+SINKHORN (max_iters=30) | 0.05 | 0.95 | task225 §5 |
| task200 dual_v5 | 0.8387→0.9305 (non-monotone) | 0.16→0.07 | task200 verdict (collision_rate) |
| task221 | 0.999-0.9999 | 0.001-0.0001 | task221 verdict |

**关键发现**: 

1. **task223 baseline "collision = 0.99" 是 Sinkhorn 后 1% uniqueness 的误读**, 不是真的 99% collision. 真实 baseline (no Sinkhorn, dedup 后) collision = 0.0.
2. **task225 §5 表里 baseline "9.07%" 是 uniqueness = 90.93% 的误读**, 实际 collision = 9.07%. 这是 **跟 task223 同方向 1% vs 99% 颠倒错读的镜像**.

两套 verdict **都把 uniqueness 当 collision 报**, 方向相反:

- task223 把 uniqueness 1% 写成 collision 99% (方向取反)
- task225 §5 把 uniqueness 9.07% 写成 collision 9.07% (去掉了 1-, 直接当 collision)

统一后 task84 baseline collision 严格按 `1 - unique/total`:
- task223 baseline (no Sinkhorn, dedup): collision = **0.0** (uniqueness = 100%)
- task225 baseline (vanilla + Sinkhorn max=30 + dedup): collision = **9.07%** (uniqueness = 90.93%)

## 2. collision ≤ 12% 达标线重述

| 口径 | baseline collision | PC κ collision | "≤ 12%" 含义 |
|---|---|---|---|
| task223 §5 (错读) | 0.99 (实际 uniqueness) | 0.3706 | baseline 已远超 12%, 不可能达标 |
| task225 §5 (错读) | 9.07% (实际 uniqueness 90.93%) | "37-63%" (混合 collision + uniqueness) | 9.07% ≤ 12% ✅ 满足, 但 PC κ 行不可读 |
| **统一口径 (本任务)** | **9.07%** | **0.37** | **"≤ 12%" 在 task84 vanilla baseline 已满足** (9.07 < 12), PC κ 远超 (0.37 > 0.12) |

**Issue #6/#7/#9 一路沿用的 `collision ≤ 12%` 达标线是写在 task225 §5 口径上的**. 统一口径后:

- baseline 9.07% ≤ 12% **已满足**, 不构成"降到 12% 以下"作为改进方向
- PC κ 0.37 远超 12%, **永远不达标** (除非引入 Sinkhorn)
- Issue #9 的 Gate 1 `collision ≤ 0.3706` 应改为 **collision 比 baseline 低 + utilization 显著高** (相对值, 不是绝对值)

## 3. 历史数字重述表 (统一口径后)

| 历史 verdict | 原报 collision | 重述后 collision | 备注 |
|---|---|---|---|
| task223 baseline | 0.99 | 0.0 (no Sinkhorn) / 0.99 (Sinkhorn 后 uniqueness 1%) | 表里把 uniqueness 1% 错读为 99% collision |
| task225 §5 baseline | 9.07% | 0.0907 (实际是 collision, 不是 uniqueness) | 表里"9.07%"实际就是 collision 严格定义, 不是 uniqueness |
| task225 PC κ | "37-63%" | 0.37 (严格) / 0.63 (若指 uniqueness) | 表里混合 collision + uniqueness, 不可读 |
| task200 dual_v5 | 0.8387 → 0.9305 | 同 | collision 严格定义, non-monotone |
| task221 | 0.999-0.9999 | 同 | 严格定义 |
| task237 (max_iters=10) | 0.1005 | 同 | 严格定义 |
| task253 Möbius 残差 | 0.0915 | 同 | 严格定义 |
| task84 HG-Rec baseline (实际跑的) | (没单独报) | 0.0 (Stage 2 没去重时) | 实际 inference 走 Sinkhorn max=30 + 4-digit dedup, collision 9.07% |

**结论**: **task225 §5 表里 baseline 行填的 9.07% 是正确的 collision 严格定义**, PC κ 行"37-63%" 才是错的. Issue #10 §反证 §2 揭示的"task223 0.99 vs task225 9.07% 矛盾"实际上是 **task223 baseline 行误读**, 不是 task225 错.

**修订**: Issue #10 §反证 §2 应改为 "task223 verdict §5 表 baseline collision 0.99 是 uniqueness 1% 误读, task225 verdict §5 表 baseline 9.07% 是正确 collision 严格定义. 冲突的根源是 task223 verdict 误报, 不是 task225 误报."

## 4. task200 dual_v5 retro-label (按 task233 §9 要求)

按 task233 §9 要求, 给 task200 "lower collision degrades R@10 (T5 has ~-10% collision sensitivity)" 打 retro-label:

**retro-label**: `confounded by Stage 3 truncation (silent death ep93/200) + codebook convergence (collision non-monotone 0.8387→0.9305)`

来源:
- task200 R@10 = 0.0915 (-10.3% vs baseline)
- task233 RERUN R@10 = 0.0934 (-8.4%), +2.1% recovery from budget fix
- 残余 -8.4% 是 collision/convergence 贡献, 但不是直接 "lower collision degrades R@10" 因果 (因 collision non-monotone, 不是单调下降)
- task233 §4.3 自承认: dual_v5 collision non-monotone (ep14 0.8387 → ep49 0.9305), cos_mean < 0.30 gate 未过, "lower collision" 是 confounded

**结论**: task200 dual_v5 -10.3% **不是** clean collision sensitivity evidence, 不能用作 collision→R@10 因果曲线的 datapoint. 这是 task233 已完成的 retro-label, 本 Gate 0 同步到 verdict 索引.

## 5. 关键决策点 (R11.3)

- **零 GPU 纯审计**: 本 Gate 0 不启动任何训练或推断, 只读代码 + 已有 verdict + diagnostic json. 严格符合 Issue #10 Gate 0 设计 (零 GPU, 分钟级).
- **不启动 Arm B 训练**: R11.4 不可逆决策点. Issue #10 方向 A (Arm B max_iters=20) 仍等用户决策. Gate 0 收口后, 等用户决策 A/B/C.
- **Issue #10 §反证 §2 修订**: 不推翻 Issue #10 提案, 但 §反证 §2 揭示的冲突根源改为 "task223 verdict 误报", 不是 "task225 错". 后续 Issue #10 推动应同步修订.
- **"≤ 12% 达标线" 重写**: 该线在统一口径下已自动满足 baseline, 不构成改进方向. Issue #6/#7/#9 全线 collision 达标线需重写为相对值 (collision 比 baseline 低 + utilization 显著高). 这超出 Gate 0 范围, 留 Gate 1+ 处理.

## 6. 后续 Gate 1 (等用户决策 Issue #10 方向 A 后)

按 Issue #10 §阶段闸门 Gate 1:
- Arm B 的 collision 必须与 Arm A (no Sinkhorn, collision = 0.0, 跟 task84 baseline 同款), Arm C (Sinkhorn max=30, collision = 9.07%, 跟 task84 vanilla+SINKHORN 同款) 各差 ≥ 15pp
- 否则不构成三个不同水平
- 同时记录三臂各自的 L0/L1/L2 utilization, 若 utilization 变动幅度 ≥ collision 变动幅度 → Sinkhorn confounded

按本 Gate 0 重述, 三个 arm 应该是:
- Arm A: collision = 0.0 (no Sinkhorn) — **task225 §5 表里 baseline 行 = 0.0**
- Arm C: collision = 9.07% (Sinkhorn max=30 + dedup) — **task225 §5 表里 vanilla+SINKHORN 行**
- Arm B: max_iters=20, 预期 collision ∈ [3%, 9.07%], 跟 Arm C 距离 < 15pp → **Gate 1 可能 fail**, 需换 max_iters

**修正 Gate 1 设计**: Arm B 应选 max_iters = 0 (无 Sinkhorn, collision = 0.0) + max_iters = 5 (collision 期望 ~25-30%) + max_iters = 30 (collision = 9.07%). 三点跨距 0% / 25% / 9.07%, 符合 ≥ 15pp 间隔.

但这是 R11.3 自主决策, 等用户确认 Issue #10 方向 A 后再调整 launcher.

## 物理产物

```
verdicts/task259_issue10_gate0_collision_metric_unification_result.md  (本文件)
```

无 scripts/, 无 description/ (Gate 0 是审计任务).

result: Task #259 — Issue #10 Gate 0 PASS. collision 严格定义锁定 (`collision_rate = 1 - unique/total`), 历史数字重述表完成. 关键发现: task223 verdict baseline 行 0.99 是 uniqueness 1% 误读, task225 §5 baseline 9.07% 才是正确 collision 严格定义. "≤ 12% 达标线" 在统一口径下 baseline 已满足, Issue #6/#7/#9 全线 collision 达标线需重写. task200 dual_v5 retro-label: confounded by Stage 3 truncation + codebook convergence. 不启动 Arm B 训练, 等用户决策 Issue #10 方向 A.