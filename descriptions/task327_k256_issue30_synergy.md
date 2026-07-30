# Task #327 — K=256 anchor + Issue #30 per-layer Codebook Transforms synergy probe

**日期**: 2026-07-30
**状态**: 🔄 READY (R11.5 自主决策启动 — 取代 task326 K=384 NO-GO, 最高 ROI 协同实验)
**Stage**: Stage 1 (100 epoch) + Stage 2 Sinkhorn + Stage 3 T5-mini 200 epoch + Stage 4 R@10 eval @ K=50 amplifier
**Anchor**: task194 K=256 R@10=0.1053 (⭐⭐⭐ trade-off 顶峰) + task301 Issue #30 R@10=0.1022 (Stage 1 GO marginal) + Stage 4 K=50 amplifier (+2.3% task307)

## 背景

跨方向已实证 3 个独立 lever:
1. **K-sweep K=256 anchor** (task194): R@10=0.1053 — Stage 1 K=256 vanilla 最优
2. **Issue #30 Stage 1 marginal GO** (task301): R@10=0.1022 — K=64/128/256 vanilla + per-layer r_l=[0.1,1,10]+s_l=[2,2,2]
3. **Stage 4 K=50 amplifier** (task307): R@10=0.1022 → 0.1045 (+2.3%) — Issue #30 特定 K=50 amplifier

**K=256 + Issue #30 (Stage 1 K=256 + per-layer transforms) 协同 尚未测试**.

Hypothesis: 三个 lever 联立 → R@10 > 0.1053 (新 anchor ⭐⭐⭐⭐)
- 若协同成立 → R@10 跨方向 ceiling 突破 0.1053
- 若协同不成立 (≤ 0.1053) → 验证 K-sweep 跟 Issue #30 不可叠加, 锚定 K=256 vanilla

## 设计 (单臂 K=256 + Issue #30)

| 参数 | 值 | 来源 |
|------|----|------|
| num_emb_list | [256, 128, 256] | task194 K=256 anchor |
| radius_list | [0.1, 1.0, 10.0] | Issue #30 task301 |
| scale_list | [2.0, 2.0, 2.0] | Issue #30 task301 |
| c_k_range_list | "1.0:5.0,0.5:20.0,0.5:20.0" | Issue #30 task301 |
| epochs (Stage 1) | 100 | Issue #30 task301 |
| batch_size | 512 | K=256 ≥ batch_size (n_samples 512 ≥ 256 OK) |
| Stage 4 beam_size | 50 | Issue #30 K=50 amplifier task307 |

## 决策阈值

| K=256 + Issue #30 R@10 | 解读 |
|------------------------|------|
| > 0.1053 | GO ⭐⭐⭐⭐ 新 anchor (跨方向 ceiling 突破) |
| 0.1020 ~ 0.1053 | 中性, 跟 K=256 anchor 持平 |
| ≤ 0.1020 | 协同退化 (Issue #30 marginal 在 K=256 不放大) |

## 关联

- task194 (K-sweep K=256 anchor ⭐⭐⭐ R@10=0.1053)
- task279 (K-sweep K=512/1024 NO-GO)
- task301 (Issue #30 Stage 1 GO marginal R@10=0.1022)
- task307 (Stage 4 K=50 amplifier +2.3%, Issue #30 特定)
- task326 (K=384 Gate 0 FAIL, USAGE-KILL)
- verdicts/task327_*_verdict.md (本任务输出)

## R10 推进

- task320 还在 GPU 0/2/3 跑 (Arm A/D/E @ Ep80-88/200), GPU 1 已释放
- GPU 1 启动 task327 Stage 1 (Issue #30 K=256, 100 epoch, ~30 min)
- Stage 2 Sinkhorn → Stage 3 T5-mini 200 epoch → Stage 4 R@10 @ K=50
- 总耗时 ~2h

## 关键决策点 (R11.5 自主决策)

1. **task327 取代 task326**: K=384 NO-GO 后, 跨方向协同是最高 ROI (K=256 anchor + Issue #30 GO marginal + K=50 amplifier)
2. **batch_size=512**: K=256 ≥ 256, 留 margin 512
3. **Stage 4 K=50 amplifier**: 跟 task307 一致 protocol, 才能跟 task194 比较
4. **GPU 1 启动**: task320 释放 GPU 1 后错峰

result: Task #327 — K=256 + Issue #30 per-layer Codebook Transforms synergy probe. Stage 1 100 epoch + Stage 2 Sinkhorn + Stage 3 T5-mini 200 epoch + Stage 4 K=50 amplifier. 决策阈值 R@10 > 0.1053 (vs task194 K=256 anchor) 验证协同效应