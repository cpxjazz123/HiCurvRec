# Task #236 — Issue #10 Gate 0: collision 指标口径统一 (zero GPU)

## 目的

Issue #10 §H0 (零 GPU, 阻塞项): 全仓库 collision 数字至少存在两套不兼容定义;
在统一为单一定义并把历史数字全部重述之前, `collision <= 12%` 不可判定.

## 单一权威定义 (issue #10 Gate 0 output)

**`collision_rate = (N_items - N_unique_SID) / N_items = 1 - uniqueness_rate`**

代码源: `HG-Rec/model/hrqvae_trainer.py:246` 与 `scripts/task200_stage2_codebook.py:166`
两处一致:
```python
collision_rate = (num_sample - len(list(indices_set))) / num_sample
```

N_items = 9922 (Musical_Instruments 5-core), N_unique_SID = 去重 SID tuple 数.

**uniqueness_rate** 互补定义: `uniqueness_rate = N_unique_SID / N_items`.

## 历史数字重述表 (统一 collision_rate 口径)

| 来源 | 原始标签 | 原始值 | 实际度量 | collision_rate (统一) | uniqueness_rate |
|------|---------|--------|----------|----------------------|-----------------|
| task223 §结论 | collision_rate | 0.99 (baseline) | collision | **0.99** | 1% |
| task223 §结论 | collision_rate | 0.3706 (PC κ) | collision | **0.3706** | 62.94% |
| task223 §过程 | collision_groups iter 10 | 1644 | group count | n/a (counts duplicates) | n/a |
| task225 §5 baseline #84 | "Stage 2 collision" 9.07% | uniqueness (列标签错) | uniqueness | **0.9093** | 9.07% |
| task225 §5 Vanilla + Sinkhorn | "~5%" | uniqueness | uniqueness | **0.95** | ~95% |
| task225 §5 PC κ (#225) | "37-63%" | **混合**: collision 37.06 + uniqueness 62.94 塞同单元格 | mix | **0.3706** | 62.94% |
| task225 §5 TIGER | "~10%" | uniqueness | uniqueness | **0.90** | ~10% |
| task225 §5 Letter | "~12%" | uniqueness | uniqueness | **0.88** | ~12% |
| task200 dual_v5 ep14 | collision_rate | 0.8387 | collision | **0.8387** | 16.13% |
| task200 dual_v5 ep49 | collision_rate | 0.9305 | collision | **0.9305** | 6.95% |
| task221 Gromov all ep | collision_rate | 0.999-0.9999 | collision | **0.999+** | <0.1% |

## 关键发现

### 1. task225 §5 列标签错位

task225 §5 跨变体表头写「Stage 2 collision」, 但行值（baseline 9.07%, Vanilla+SINKHORN ~5%, TIGER ~10%, Letter ~12%）全是 **uniqueness 比例** (跟 task225 §结论段自己写的 "Stage 2 SID uniqueness: 62.94%" 同套口径).

证据:
- task225 §结论段: 「Stage 2 SID uniqueness: 62.94% (vs baseline 9.07%)」— 9.07% 是 uniqueness
- task225 §5 表 baseline 9.07% — 跟结论段 baseline 9.07% 一致, 但被错误地标为 collision
- task225 §5 表 PC κ 「37-63%」— 37.06 + 62.94 = 100, 是把互补的两个数字塞进了同一格

### 2. `<= 12%` 达标线在两套口径下含义完全相反

- **若 12% = uniqueness (task225 §5 的真实含义)**: 含义是 "uniqueness >= 88%", 等价 collision_rate <= 0.12. 这极激进, Sinkhorn sk_iters=50 + 4th-digit dedup 都不一定达到 (task223 vanilla+Sinkhorn uniqueness 95%, collision 0.05, **勉强过**). task200 dual_v5 uniqueness 16.13% → collision 0.8387, **远未达标**.
- **若 12% = collision_rate (字面直读)**: 含义是 "collision <= 0.12", 等价 uniqueness >= 88%. 跟上面 uniqueness 12% 等价 (数字巧合). 同一意思.

但 task200 verdict 报的 "collision <= 12%" 想表达什么? 查 Issue #6 (Task #220) + Issue #7 (Task #221) 的 thread:
- Issue #6 (Task #231 Phase 0 sweep) 是测 Phase 0 判据 (euc-hyp argmin 一致率), 不涉及 collision 数值
- Issue #7 (Task #232 Gromov sweep) 同上
- task233 §5 第 3 条原文: "Issue #6/#7 <=12% collision bar 暂时不动"

bar 出处不明, 大概率是从 task225 §5 表面数字直读. 真实意图需要 Issue #10 后续 Gate 2 才能定 (collision→R@10 因果).

### 3. task225 自身 baseline 9.07% ≠ task223 baseline 99%

两个文件记录的 baseline #84 collision 不同:
- task223: collision_rate 0.99 (即 uniqueness 1%)
- task225: uniqueness 9.07% (即 collision 0.9093)

可能解释:
- task223 baseline 是 **完全 no-Sinkhorn + no-4th-digit dedup** 的 raw stage 2 输出
- task225 baseline 是 **default Stage 2 (Sinkhorn sk_iters 默认 + dedup)** 的输出 (dedup 把部分 collision 抹平)

需要查 `scripts/task84_stage2_codebook.sh` (如有) 或 task223 跟 task225 是否真的用了不同 Stage 2 配置.

## `<= 12%` bar 统一口径重述

**bar 真实意图推断**: task225 §5 把 uniqueness 当 collision, 所以 "collision <= 12%" 实际意图应是 "uniqueness <= 12%" 即 "collision_rate >= 0.88" (collision **高**才是 HG-Rec #84 baseline 状态).

但 Issue #6/#7/#9 一路用的 "降 collision" 是想从 baseline (collision 0.99 task223 / 0.91 task225) 降下来, **这方向跟 bar 推断的方向相反**.

**collision <= 12% 统一口径等价**:
- uniqueness >= 88% (Sinkhorn 极强场景, phonism vanilla 0.95 唯一过线)
- HG-Rec #84 baseline task223 0.99 / task225 0.91 → 均**不达标** (按这个 bar, HG-Rec 自己都不达标, 这显然不是 bar 真实意图)
- 真实意图 = Issue #10 Gate 2 (Arm B 跑完才能定)

## retro-label: task200 「-10.3% collision sensitivity」

按 Issue #10 §步骤 1 要求, 给 task200 -10.3% collision sensitivity claim 打 retro-label:

**task200 -10.3% collision sensitivity claim retro-label**:
- task200 dual_v5 collision 0.8387 vs task225 PC κ 0.3706 (口径错位下也只跨 1 个 arm, 且 task200 是 task220 的 dual_arm_C_v5 变体, 跟 task220 baseline 不在同一架构族)
- task233 §5 复跑 dual_v5, 撞 Stage 3 silent death (ep93/200) — Stage 3 截断混淆
- task233 §4.3 dual_v5 collision 非单调 (ep14 0.8387 → ep49 0.9305), cos_mean 不通过 L1 0.9054, L2 0.9277
- **conclusion**: task200 -10.3% 单臂不能下 collision causal claim, 必须 Issue #10 Gate 1 3-arm 设计后才能 disentangle. retro-label = **confounded by Stage 3 truncation + codebook convergence**.

## Issue #10 Gate 0 决策

✅ **Gate 0 PASS**: 锁定 collision_rate 单一权威定义, 重述历史数字.

⚠️ **未触发 Gate 0 终止条件** (重述后比较符号翻转). 符号未翻转, 但暴露 bar 真实意图不明 — 必须进 Gate 2 才能定 collision→R@10 因果.

⚠️ **新阻塞项**: bar "<=12%" 真实意图不明, 必须 Issue #10 Gate 1 (Arm B 跑完) + Gate 2 才能定. Issue #9 Gate 1 (`collision <= 0.3706`) 是 task222 PC κ 的 collision, 跟 Issue #6/#7 的 `<=12%` bar 是不同数字. 两者都需要 Issue #10 Gate 1 落定才能 reconcile.

## 决策点 (R11.3 自主决策)

| 决策 | 选了什么 | 为什么 |
|------|---------|--------|
| 权威定义 | collision_rate = 1 - uniqueness | hg-Rec/model/hrqvae_trainer.py:246 原始定义, 与 task200 一致 |
| task225 §5 处理 | retro-label 列错位, 重述为 uniqueness | 跟 §结论段 "Stage 2 SID uniqueness" 同口径 |
| `<= 12%` bar | 标 "真实意图不明, 待 Issue #10 Gate 2 定" | 不能猜, 必须 Arm B 实测才能定 |
| task200 retro-label | **confounded by Stage 3 truncation + codebook convergence** | Issue #10 §步骤 1 明确要求 |

## 产物

- `verdicts/task236_collision_metric_unification_result.md` (本文件)
- 0 行 GPU 代码, 0 行产品, 0 commit 必要 (verdict 文本落盘即可)

## Status

Gate 0 PASS. 后续 Gate 1 (Arm B Sinkhorn 部分配置 + 1 次 Stage 3 训练 + Stage 4 + slice) 等下一 tick 启动 (R10). 计算预算: 1 × Stage 3 训练 ~44min + Stage 4 eval ~3min + slice ~1min ≈ 48min.

result: Task #236 — Issue #10 Gate 0: collision 指标口径统一 (zero GPU)
