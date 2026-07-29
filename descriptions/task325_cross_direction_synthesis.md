# Task #325 — Cross-direction synthesis (Issue #30 / #38 / #39)

**日期**: 2026-07-30
**状态**: 📋 SYNTHESIS (cross-direction 综合分析, 等 task320 + task324 Part 2 GPU 数据补完)
**目的**: 跨 Issue #30 (Stage 1/2 per-layer Codebook Transforms) + Issue #38 (Stage 3 训练协议) + Issue #39 (Stage 4 召回改造) 综合分析 R@10 杠杆分布

## 综合 R@10 ceiling

| 端点 | R@10 | 跨方向 % | 来源 |
|------|------|----------|------|
| **task194_k0256** | **0.1053** ⭐⭐⭐ | 100% ceiling | task194 + task278 batch |
| Issue #30 K=100 amplifier | 0.1045 | 99.2% | task301 |
| Issue #30 K=120 amplifier | 0.1041 | 98.9% | task301 |
| Issue #30 K=50 amplifier | 0.1041 | 98.9% | task301 |
| Issue #30 GO marginal | 0.1022 | 97.1% | task301 |
| baseline #84 | 0.1020 | 96.9% | task84 |
| task243 epoch=200 | 0.0978 | 92.9% | task243 |

## R@10 杠杆分布

1. **Stage 1/2 K-sweep K=256 (task194)**: ⭐⭐⭐ +3.3% (最强 anchor)
2. **Stage 4 K=50/100/120 amplifier (Issue #30 特定)**: +2.1-2.5% (但 task318 50 epoch 训练无 amplifier — 非 universal)
3. **Issue #30 marginal GO (r_l+s_l 协同 gap ≈ 0.3pp)**: +0.2% (marginal)
4. **Stage 3 优化器改造 (task318 + task320)**: NO-GO
5. **Stage 4 ANN dense (HNSW/IVF-PQ)**: NO-GO (protocol mismatch)
6. **Stage 3 T5 size**: NO-GO (大模型反而差)

## 24+ 方向 NO-GO 收口

- Stage 1/2 quantizer 架构 24 方向 (Issue #28/#29/#30/#31/#32/#33/#34/#35/#36)
- Stage 3 训练协议 5-arm (Issue #38)
- Stage 4 后处理 rerank 3-arm (task316/317/319)
- Stage 4 ANN dense 召回 3-arm (Issue #39 Part 1)

## 剩余未探索方向

1. **Issue #39 Part 2 (T5.generate SID 4-arm)**: Arm C rerank / D beam=100/200 / E control
2. **D6 Issue #30 ablation (task321/322)**: r_l only / s_l only / r_l+s_l 拆分
3. **跨方向协同**: K=256 (anchor) + Issue #30 (Stage 1/2) + Stage 4 K=50 amplifier 叠加验证

## 关联

- Issue #30 (Stage 1/2 per-layer Codebook Transforms GO marginal)
- Issue #38 (Stage 3 训练协议改造, task318 NO-GO)
- Issue #39 (Stage 4 召回改造, task324 Part 1 NO-GO)
- task194 (K=256 anchor ⭐⭐⭐)
- task278 (12 ckpt batch Stage 4 eval)
- task301 (Issue #30 K-sweep K=50/100/120 amplifier)
- task318 (Issue #38 Arm 1 optimizer NO-GO)
- task324 (Issue #39 Stage 4 召回 5-arm)
- verdicts/task325_cross_direction_synthesis.md (synthesis output)

result: Task #325 — Cross-direction synthesis. R@10 ceiling = task194_k0256 0.1053 (+3.3%). 已实证 24+ 方向 NO-GO 收口, 唯一未探索: Issue #39 Part 2 + D6 ablation + 跨方向协同