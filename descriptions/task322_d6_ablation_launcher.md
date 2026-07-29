# Task #322 — Issue #30 r_l + s_l ablation (D6) launcher prep

**日期**: 2026-07-30
**状态**: 🔄 READY (D6 ablation Arm A launcher 已写, 等 task320 释放 GPU)
**Stage**: Stage 1 + Stage 2 + Stage 3 + Stage 4 (3-arm: r_l only / s_l only / r_l+s_l control)
**Anchor**: Issue #30 GO marginal R@10=0.1022 (+0.2pp vs baseline 0.1020)

## 背景

Task #321 description 已写. 现在 task322 准备 3-arm 串行 launcher.

## 3-arm 设计

| Arm | r_l | s_l | 决策 |
|-----|-----|-----|------|
| **A** | [0.1, 1.0, 10.0] | [1.0, 1.0, 1.0] | r_l 单独效果 |
| **B** | [1.0, 1.0, 1.0] | [2.0, 2.0, 2.0] | s_l 单独效果 |
| **C** | [0.1, 1.0, 10.0] | [2.0, 2.0, 2.0] | Issue #30 GO baseline (control) |

## 已写脚本

- `scripts/task322_d6_ablation_armA_rl_only.sh` — Arm A Stage 1 launcher (r_l only)
- (待写) Arm B (s_l only) + Arm C (r_l+s_l control) launcher
- (待写) Arm A/B/C Stage 2 Sinkhorn inference + Stage 3 T5-mini 200ep + Stage 4 K=20 eval

## 启动时机

- GPU 1 等待 task320 Arm A/D/E 完成 (ETA ~110 min, 200 epoch × 3 arms)
- Arm B GPU 1 重启后 (R7 严令不抢卡, 等 task320 全部完成)
- 串行执行 3 arm 总耗时估计 ~4-5 hr (Stage 1 100ep + Stage 3 200ep 是 GPU 密集段)

## 决策阈值

- Arm A R@10 ≈ Arm B R@10: r_l / s_l 都不是真杠杆, Issue #30 marginal 是 noise
- Arm A R@10 > Arm B R@10 + ≥ 0.005: r_l 是真杠杆
- Arm B R@10 > Arm A R@10 + ≥ 0.005: s_l 是真杠杆
- Arm C > Arm A + Arm B ≥ 0.005: r_l + s_l 协同效应

## 关联

- Task #321 description (D6 ablation design)
- Issue #30 (per-layer Codebook Transforms r_l + s_l)
- Task #301 (Issue #30 完整 Gate 0-4 pipeline)
- Task #318 (Issue #38 Arm 1 optimizer, 证明 K=100 amplifier 非 universal)
- Task #312/313 (Issue #35 r_l/s_l 隔离 NO-GO -17%, 用 [0.5,1,2]+[1,1,1] 测试 → Issue #30 design [0.1,1,10]+[2,2,2] 更温和)
result: Task #322 — D6 Issue #30 ablation Arm A launcher 已写. 等 task320 完成后启动 3-arm 串行. 决策 Issue #30 marginal +0.2pp 是 r_l / s_l / 协同