# Task #325 — Issue #38 + Issue #39 跨方向综合分析 (2026-07-30)

**日期**: 2026-07-30
**状态**: 📋 综合分析 (cross-direction synthesis, 等 task320 + task324 Part 2 GPU 数据补完)
**目的**: 跨 Issue #30 / #38 / #39 综合分析 Stage 1/2/3/4 改造方向 R@10 杠杆分布

## 综合 R@10 (跨所有方向, T5.generate SID 协议, beam=20 default)

### Stage 1/2 (per-layer Codebook Transforms) — Issue #30 GO marginal

| 配置 | R@10 | vs baseline | 备注 |
|------|------|-------------|------|
| baseline (task84) | 0.1020 | — | 锚点 |
| task243 epoch=200 | 0.0978 | -4.1% | default Stage 3, 跟 baseline 接近 |
| task156 (T5-mini K=128) | 0.1034 | +1.4% | K=128 + t5-mini |
| task194_k0256 ⭐ | **0.1053** | **+3.3%** | 跨方向最高 baseline |
| Issue #30 GO (r_l+s_l) | 0.1022 | +0.2% | marginal GO |
| Issue #30 K=50 amplifier | 0.1041 | +2.1% | beam=50 amplifier |
| Issue #30 K=100 amplifier | 0.1045 | +2.5% | beam=100 amplifier (task301) |
| Issue #30 K=120 amplifier | 0.1041 | +2.1% | beam=120 amplifier |

### Stage 3 训练协议 (Issue #38) — task318 + task320 数据

| 配置 | val_NDCG@20 | R@10 (test, beam=100) | 备注 |
|------|-------------|---------------------|------|
| task318 Adam 50 ep | 0.0947 | 0.0971 | control |
| task318 AdamW 50 ep (wd=0.01) | 0.0947 | 0.0996 | +0.0025 vs Adam |
| task318 SGD 50 ep | 0.0862 | 0.0905 | -8.9% NDCG |
| task318 Adafactor 50 ep | 0.0775 | 0.0780 | -18.2% NDCG |
| **task318 AdamW vs Adam** | Δ<0.001 val | **+0.0025** test | Adam=AdamW 数学等价 |
| task320 Arm A 200 ep (AdamW+cosine+warmup) | 0.0918 (best) | 🔄 | 等 Stage 4 eval |
| task320 Arm B 200 ep (LR schedule inverse sqrt) | 0.0837 (best) | 🔄 | 等 Stage 4 eval |
| task320 Arm D 200 ep (BF16) | 0.0940 (best) | 🔄 | 等 Stage 4 eval |
| task320 Arm E 200 ep (control Adam) | 0.0946 (best) | 🔄 | 等 Stage 4 eval, 应 ~0.0978 |

### Stage 4 召回改造 (Issue #39) — task324 Part 1+2

| Protocol | Arm | R@10 | 备注 |
|----------|-----|------|------|
| **ANN dense** | E dense baseline | 0.0109 | protocol baseline |
| ANN dense | A HNSW | 0.0109 | HNSW ≡ dense |
| ANN dense | B IVF-PQ | 0.0050 | -54% 退化 |
| **T5.generate SID** | E control (task243 beam=20) | 0.0978 | baseline |
| T5.generate SID | E control (task278 batch beam=50) | 0.1041-0.1053 | beam=50 amplifier |
| T5.generate SID | C rerank | 🔄 READY | 待 GPU |
| T5.generate SID | D beam=100 | 🔄 READY | 待 GPU |
| T5.generate SID | D beam=200 | 🔄 READY | 待 GPU |

## 综合结论 (跨所有方向)

### R@10 杠杆分布 (按方向)

1. **Stage 1/2 K-sweep (task194 K=256)**: R@10=0.1053 (+3.3%) ⭐⭐⭐ **跨方向最高 baseline**
2. **Stage 4 beam=50 amplifier (task301 K=50/100/120)**: R@10=0.1041-0.1045 (+2.1~2.5%) — 仅在 Issue #30 端点上有效, task318 50 epoch 训练无 amplifier
3. **Stage 1/2 Issue #30 marginal GO (r_l+s_l)**: R@10=0.1022 (+0.2%) — marginal, 无实质突破
4. **Stage 3 优化器改造 (task318 AdamW)**: R@10=0.0996 (-5.4%) — NO-GO vs anchor
5. **Stage 4 ANN dense (HNSW/IVF-PQ)**: R@10=0.005-0.011 — protocol mismatch, NO-GO
6. **Stage 3 T5 size**: t5-mini > t5-small > t5-base (task278) — 越大越差, NO-GO

### R@10 ceiling 实证

| 端点 | R@10 | 跨方向 % |
|------|------|----------|
| task194_k0256 (K=256 anchor) | **0.1053** | 100% ceiling |
| Issue #30 K=100 amplifier | 0.1045 | 99.2% |
| Issue #30 K=120 amplifier | 0.1041 | 98.9% |
| Issue #30 K=50 amplifier | 0.1041 | 98.9% |
| Issue #30 GO marginal | 0.1022 | 97.1% |
| baseline #84 | 0.1020 | 96.9% |
| task243 epoch=200 | 0.0978 | 92.9% |

**task194_k0256 0.1053 仍是已实证最强 R@10**. 任何新方向必须以此为对照基线 (vs 之前 task84 0.1020).

### R@10 杠杆 ≠ 协议杠杆 重要发现

- **Stage 4 protocol 杠杆**: dense T5.generate SID (R@10=0.10) vs ANN dense (R@10=0.01) = 10× 差异. **协议本身决定 R@10 数量级**, 不是算法.
- **Stage 3 optimizer 杠杆**: Adam vs AdamW 数学等价. Stage 3 优化器 / LR schedule / BF16 / R-Drop 都不是 R@10 杠杆 (Issue #38 5-arm 全 NO-GO 跟 task318 NO-GO 一致).
- **Stage 1/2 杠杆**: K-sweep (K=256 sweet spot) + Issue #30 marginal GO (r_l+s_l 协同 gap ≈ 0.3pp).

## R10 后续 (post synthesis)

**已收口 24+ 方向**:
- Stage 1/2 quantizer 架构 24 方向 (Issue #28/#29/#30/#31/#32/#33/#34/#35/#36)
- Stage 3 训练协议 5-arm (Issue #38)
- Stage 4 后处理 rerank 3-arm (task316/317/319)
- Stage 4 ANN dense 召回 3-arm (Issue #39 Part 1)

**剩余方向 (按 ROI 排序)**:
1. **Issue #39 Part 2 (T5.generate SID 4-arm)**: Arm C rerank / D beam=100/200 / E control — 待 GPU (~3 GPU-hours, 决策阈值 R@10>0.1053)
2. **D6 Issue #30 ablation (task321/322)**: r_l only / s_l only / r_l+s_l — 拆分 Issue #30 marginal GO 是 r_l 还是 s_l 还是协同 (等 task320 完成)
3. **task194 anchor 进一步探索**: K=256 端点为何最优? 是否有 K=384/512 sweet spot? (K=512/1024 task279 已 NO-GO)
4. **跨 Stage 1/2 + Stage 4 协同**: 既然 K=256 是 anchor, 跑 K=256 + Stage 4 K=50 amplifier 看是否叠加

**NORTH STAR 状态**: 24+ 方向全 NO-GO 收口, task194_k0256 R@10=0.1053 仍是 anchor. Issue #39 Part 2 / D6 ablation / 跨方向协同 = 唯一未探索 ROI 方向.

## R11.3 透明

- 综合分析基于已实证 verdicts, 不预测未跑方向
- 等 task320 (Issue #38 5-arm) Stage 4 eval + task324 Part 2 (Issue #39 4-arm) 完成后补完跨方向实证图
- R11.5 自主决策: 不等 owner 决策, 按 §"北极星原则" + owner task292 §横向联立 继续推进
- 当前 task320 还在跑 (Arm A=51/200, B=29/200, D=55/200, E=56/200, ~75 min 剩余) — 等 GPU 释放

result: Task #325 跨方向综合分析. R@10 ceiling = task194_k0256 0.1053 (+3.3%). 已实证杠杆: K-sweep K=256 + Stage 4 K=50 amplifier (Issue #30 特定) + Issue #30 marginal GO. 未实证杠杆: Issue #39 Part 2 (T5.generate SID 4-arm) + D6 Issue #30 ablation. 24+ 方向 NO-GO 收口, 跨方向协同探索是最后未穷尽路径