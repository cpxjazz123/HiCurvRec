---
type: status
status: "NO-GO"
created: 2026-08-02
tags:
  - ceiling
up: "[[index]]"
---
# NORTH STAR Status — R@10 ceiling project 综合分析 (2026-07-30)

**日期**: 2026-07-30
**状态**: 🟡 **PENDING task327** — Stage 1-3 全 NO-GO 收口, 跨方向协同 TBD (last high-ROI 候选)

## NORTH STAR ceiling = 0.1053 ⭐⭐⭐

| Anchor | R@10 | 来源 |
|--------|------|------|
| **task194 K=256** | **0.1053** | ⭐⭐⭐ 当前 ceiling, 跨 Stage 1/2/3/4 协议探索全 NO-GO |
| Issue #30 marginal (task301) | 0.1022 | +0.2pp vs baseline |
| HG-Rec baseline (#84) | 0.1020 | reference |
| phonism (横向 baseline) | 0.1058 | -0.5pp vs anchor |

## 已实证 NO-GO 杠杆 (跨 Stage 收口)

### Stage 1 量化算法 (10 issues × 22 verdict NO-GO)

| Issue / Direction | 决策 |
|------------------|------|
| #9 Gate 1 hybrid L0+L1/L2 Gromov | ❌ NO-GO (L1/L2 collapse) |
| #10 Gate 1 3-arm 曲线 | ❌ NO-GO (Arm B/C 持平/退化 baseline) |
| #11 per-layer c_k range Gate 1b FULL | ❌ NO-GO (Arm A L0=23.44%, A+ dead_revive L0=3.12%) |
| #12 Gate 0 沙漏集中度 | ❌ NO-GO (Arm B Sinkhorn Gini 0.20, R@10 持平) |
| #20 L0 utilization 三配方 | ❌ NO-GO (baseline recipe 结构性无解) |
| #23 per-layer c_k curriculum | ❌ NO-GO (三层全 OPEN, baseline 内部 R@10 杠杆已穷尽) |
| #28 Gumbel-Softmax + per-layer c_k | ❌ NO-GO (Gate 1 FAIL L0=21.9%/L1=10.2%/L2=1.2%) |
| #29 per-layer 异构 K_l=[128,64,32] | ❌ Stage 4 NO-GO (-4.0%) |
| #30 per-layer Codebook Transforms | ✅ **唯一 GO marginal** +0.2pp |
| #32 per-layer c_k range 双轴协同 | ❌ NO-GO (R@10=0.000121 -99.88%) |
| Task #283 D5 dead_revive frequency | ❌ NO-GO (hook no-op) |
| Task #282 A1 β=0 | ❌ NO-GO (β 不是 L0 杠杆, 是稳定剂) |
| Task #271/#275/#288 L0 收口三配方 | ❌ NO-GO 全部 |
| κ-decouple K=64/128/256 | ❌ NO-GO (-15% to -18%) |

### Stage 2 SID 协议 (Issue #33)

| Issue / Direction | 决策 |
|------------------|------|
| #33 per-item soft commit loss | ❌ NO-GO (per-item 期望 = K 个码字 weighted centroid → codebook collapse) |
| #33 per-item soft forward weighted sum + STE | ❌ NO-GO (数学等价 commit loss) |
| 3-way alternative quantizer (FSQ/EMA/Restoration) | ❌ NO-GO 全部 |

### Stage 3 训练协议 (Issue #38)

| Arm | 方案 | test_R@10 | vs baseline |
|-----|------|-----------|-------------|
| A | Control (vanilla) | 0.0942 | -7.6% |
| B | LR scheduler inv_sqrt | 0.0938 | -8.0% |
| D | BF16 mixed precision | 0.0983 | -3.6% |
| E | Regularization (dropout 0.1 + wd 0.01) | 0.0981 | -3.8% |
| C | R-Drop (TBD Ep200) | TBD | 待 GPU 0 |

**Issue #38 4/5 Arms NO-GO 收口**. Stage 3 协议 (Optimizer/LR/BF16/Regularization) 不是 R@10 杠杆.

### Stage 4 召回协议 (Issue #39)

| Arm | 方案 | R@10 | 决策 |
|-----|------|------|------|
| E_control (task243 ckpt) | dense beam=50 | 0.0000 | ❌ task243 ckpt BROKEN |
| E_control (task301 ckpt) | dense beam=50 | 0.1041 | baseline confirmation |
| D_beam100 (task301 ckpt) | dense beam=100 | **0.1041** | ≈ marginal 0 增益 |
| D_beam200 (task301 ckpt) | dense beam=200 | OOM | INFEASIBLE |
| A_hnsw / B_ivf_pq / C_rerank | — | NOT EVALUATED | TODO ~3-5 天 ROI 低 |

**Issue #39 NO-GO 收口**. Part 1 ANN dense (R@10=0.011 protocol 数量级失败) + Part 2 T5.generate SID (D_beam100 ≈ marginal).

### K-sweep Stage 1 (6 directions)

| K | Stage 1 | Stage 4 R@10 | 决策 |
|---|---------|--------------|------|
| 32 | OK | 0.1034 | ✅ |
| 64 | OK | 0.1041 | ✅ |
| 128 | OK | 0.1027 | ✅ |
| **256** | **OK** | **0.1053** | ⭐⭐⭐ anchor |
| 384 | **Gate 0 FAIL** (L0=16.9%) | N/A | ❌ |
| 512 | OK | 0.0824 | ❌ -19.2% |
| 1024 | OK | 0.0847 | ❌ -16.9% |

**K-sweep 闭合**: K=256 是 trade-off 顶峰.

## 仍验证中的高 ROI 候选 (TBD)

### task327 — K=256 + Issue #30 per-layer Codebook Transforms synergy

**目的**: 跨 3 个独立 lever 联立
- K=256 anchor (task194)
- Issue #30 r_l=[0.1,1,10] + s_l=[2,2,2] (task301)
- Stage 4 K=50 amplifier (task307)

**当前状态** (2026-07-30 11:10):
- Stage 1 ✅ L0=66.4% + L1/L2=100% + collision=0.0745
- Stage 2 Sinkhorn ✅
- Stage 3 T5-mini 200 epoch RUNNING (GPU 1, Ep50/200, ~5 hours remaining)

**决策阈值**:
- R@10 > 0.1053 → GO ⭐⭐⭐⭐ 新 anchor (跨方向 ceiling 突破)
- R@10 ∈ [0.1020, 0.1053] → 中性, 协同不放大
- R@10 ≤ 0.1020 → 协同退化, NORTH STAR FULL NO-GO 闭环

## 待 owner decision (Issue #26 OPEN)

如果 task327 也 NO-GO, 跨 Stage 协议探索已经穷尽, 后续方向需要 owner 大决策:

1. **修订 loop.md** — 改变项目目标 (例如换成多模态 / KG 增强 / 论文复现)
2. **启动架构层** — 离开 Stage 1/2/3/4 protocol 探索, 进入架构层 (cross-encoder rerank / Task #34 D9 多样 hash / κ-Stereographic)
3. **暂停 cron tick** — 保留当前 ceiling 0.1053 作为 baseline, 暂停自主推进

Issue #26 GitHub 仍 OPEN 等 owner decision.

## 已识别但未实现的方向 (Issue #34)

| Issue | 方向 | 工作量 | 期望 R@10 增益 |
|-------|------|--------|----------------|
| #34 D9 多样 hash on #30 GO | per-layer 异构 hash 函数族 + 每层多个候选 SID slot | 中 | 不明 |

候选但 ROI 低 (基于任务 320+324 NO-GO 趋势):
- cross-encoder rerank (Issue #39 C) — R-Drop 类协议失败, 类似协议预计失败
- multi-task T5 (joint SID + item classification) — 架构层大改
- KG-enhanced SID (用 MCKG 边信息作为 SID 生成约束) — 需要 KG 训练

## §16 当前任务状态 (2026-07-30 11:10)

| 任务 | Stage | 状态 |
|------|-------|------|
| task327 Stage 3 | GPU 1 | RUNNING Ep50/200 |
| task320 Arm C | GPU 0 | RUNNING Ep54/200 (val=0.1186) |
| GPU 2/3 | — | FREE |
| Issue #38/#39 | GitHub | CLOSED |
| Issue #34/#26 | GitHub | OPEN (D9 多样 hash + owner decision) |

## 综合决策矩阵

| 杠杆 | 验证状态 | 综合 |
|------|----------|------|
| Stage 1 量化算法 | ❌ NO-GO 收口 | 失败 |
| Stage 2 SID 协议 | ❌ NO-GO | 失败 |
| Stage 3 训练协议 | ❌ NO-GO (#38) | 失败 |
| Stage 4 召回协议 | ❌ NO-GO (#39) | 失败 |
| K-sweep Stage 1 | ✅ K=256 anchor | anchor |
| 跨方向协同 | 🔄 task327 RUNNING | TBD |

## R11.5 自主决策 (透明)

- §16 backlog 已穷尽 (跨 Stage 1/2/3/4 协议方向全 NO-GO)
- task327 (K=256 + Issue #30 synergy) 是剩余最后 high-ROI 候选
- 等 task327 Stage 3 完成 (~5h) → 立即 launch Stage 4 K=20/50/100 beam ablation (launcher ready)
- 若 task327 R@10 > 0.1053 → 打破 ceiling
- 若 ≤ 0.1053 → 全 NO-GO 收口 + Issue #26 owner decision 升级

result: NORTH STAR ceiling **0.1053 (task194 K=256 anchor)**. Stage 1/2/3/4 协议探索全 NO-GO 收口. task327 (跨方向协同) RUNNING 是最后 high-ROI 候选. 等 Stage 3 完成 + 立即 Stage 4 验证. 若 task327 ≤ 0.1053 → Issue #26 owner decision 升级 (修订 loop.md / 启动架构层 / 暂停 cron tick).