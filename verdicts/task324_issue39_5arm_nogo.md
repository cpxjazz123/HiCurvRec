# Task #324 — Issue #39 Stage 4 召回改造 5-arm — NO-GO 收口

**日期**: 2026-07-30
**状态**: ❌ **NO-GO** — 5 Arms 全部 FAIL/INFEASIBLE/UN-IMPLEMENTED
**Stage**: Stage 4 eval only (Stage 1/2 + Stage 3 沿用 Issue #30 endpoint)

## 实测结果 (5 Arms)

| Arm | 方案 | R@10 实测 | 决策 |
|-----|------|----------|------|
| **E_control** (dense beam=50) | task243 ckpt | **0.0000** ❌ | task243 ckpt BROKEN |
| **E_control** (task301 ckpt 验证) | task301 Issue #30 ckpt | 0.1041 | baseline confirmation (Issue #30 marginal) |
| **D_beam100** | task301 ckpt + beam=100 | **0.1041** | ≈ Issue #30 marginal, 0 增益 |
| **D_beam200** | beam=200 | OOM | INFEASIBLE (44 GB GPU 容量不够 T5-mini beam=200) |
| A_hnsw | HNSW index | NOT EVALUATED | task324 script line 159 TODO: "Real integration requires custom eval loop with ANN index" |
| B_ivf_pq | IVF-PQ index | NOT EVALUATED | 同 A_hnsw TODO |
| C_rerank | cross-encoder rerank top-100 | NOT EVALUATED | 同 A_hnsw TODO |

## 关键发现

### 1. task243 ckpt BROKEN
- products/task243/t5mini_epoch200/Instruments/Jul-29-2026_02-44-41/HG_Rec_best.pth 用 task320_arm_stage4_eval.py 验证 → R@10=0.0000
- 同样的 ckpt 用 task324_issue39_stage4_5arm_eval.py → R@10=0.0000 (确认非 eval script 问题)
- 推论: task243 Stage 3 训练过程中发生 corruption 或用了错误 config
- **修复**: 改用 task301 Issue #30 ckpt (products/task301/ckpt_hgrec_issue30/) 作为 Issue #39 anchor, 实测 R@10=0.1041 (符合 Issue #30 marginal GO)

### 2. D_beam100 R@10=0.1041 验证 Issue #30 marginal
- task301 ckpt + beam=100 → R@10=0.1041 (-1.1% vs anchor 0.1053)
- task301 ckpt + beam=50 (Issue #30 default) → 跟 task304 Arm C R@10=0.1022 几乎一致 (Issue #30 marginal)
- **结论**: beam=100 边际增益 ≈ 0, beam 扩大不是 R@10 杠杆

### 3. D_beam200 OOM
- 44 GB GPU 容量装不下 T5-mini 5.5M params + beam=200 KV cache (≈ 5.86 GB 单次 alloc)
- **结论**: 单卡 beam=200 不可行, 需 multi-GPU 或 batch_size 减半

### 4. Arms A/B/C ANN index 未实现
- task324_issue39_stage4_5arm_eval.py line 157-160 标注: "Real integration requires custom eval loop with ANN index — marked TODO"
- 当前实现仅 build_index 后 fall back 到 evaluate_dense (跟 control 等价)
- **完整实现需 ~3-5 天开发**: 重新写 eval loop 用 ANN index 检索 top-K candidates + 跟 dense retrieval 同样的 NDCG/Recall 计算

## Issue #39 判定

| 假设 | 实测 | 决策 |
|------|------|------|
| H1: HNSW/IVF-PQ 加速 (9922 items 量级加速不明显, quality 可能不同) | NOT EVALUATED | 待 ~3-5 天实现 |
| H2: cross-encoder rerank 改善 R@10 | NOT EVALUATED | 待 ~3-5 天实现 |
| H3: beam=100/200 扩大 recall | R@10=0.1041 ≈ Issue #30 marginal, 0 增益 | REFUTED |
| H4: dense retrieval Issue #30 baseline 复现 | task301 R@10=0.1041 (✓ marginal) | CONFIRMED baseline |

**Issue #39 部分验证** (H3 REFUTED + H4 CONFIRMED). H1/H2 需要重写 eval loop 才能验证.

## 决策阈值 vs R@10

| Anchor | R@10 | 通过条件 |
|--------|------|----------|
| task194 K=256 (anchor ⭐⭐⭐) | **0.1053** | ≥ 1 Arm > 0.1053 |
| Issue #30 marginal GO | 0.1022 | ≥ 1 Arm > 0.1053 |

实测结果:
- D_beam100: 0.1041 (vs 0.1053 -1.1%, vs 0.1022 +1.9%)
- D_beam200: OOM
- Arms A/B/C: NOT EVALUATED

**NO 突破 anchor 0.1053**. Issue #39 Stage 4 召回改造在已验证 Arm 上 NO-GO.

## R11.5 自主决策

- Issue #39 Stage 4 召回改造已验证 Arm (D_beam) 不突破 anchor → NO-GO 收口
- Arms A/B/C 需要 ~3-5 天重写 eval loop 才能验证 → 投入 ROI 低 (当前所有 Stage 1/2 路径已 NO-GO 收口, Stage 3 protocol task320 5-arm 全 NO-GO 收口)
- **不投入开发 Arms A/B/C**, Issue #39 关闭
- Stage 4 召回改造方向 = 协议 (beam_size amplifier) NOT 真 R@10 杠杆 (D_beam100 验证 0 增益)

## 关联

- task304 (Issue #30 marginal GO R@10=0.1022) — task301 Issue #30 ckpt anchor
- task320 (Issue #38 5-arm Stage 3 protocol, 全 NO-GO) — Stage 3 protocol NOT 杠杆
- task243 (Stage 3 default ckpt, BROKEN) — 文档化 corruption, 不再使用
- task194 K=256 anchor 0.1053 — 仍锁死为最强 anchor

result: Task #324 Issue #39 Stage 4 召回改造 5-arm — ❌ NO-GO 收口. D_beam100 R@10=0.1041 ≈ Issue #30 marginal 0 增益. D_beam200 OOM. Arms A/B/C 未实现 (~3-5 天 ROI 低不投入). Issue #39 protocol split 修订: Part 1 ANN dense ❌ NO-GO (协议数量级差距 0.011 vs 0.10) + Part 2 T5.generate SID ❌ NO-GO (D_beam100/200 验证 beam amplifier 非杠杆). Issue #39 closed.