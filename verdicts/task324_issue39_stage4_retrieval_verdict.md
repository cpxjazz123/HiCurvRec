# Task #324 — Issue #39 Stage 4 召回改造 5-arm verdict (PART 1: ANN dense retrieval)

**日期**: 2026-07-30
**状态**: ❌ **NO-GO** (Issue #39 协议差异导致 Arm A/B NO-GO + Arm E baseline R@10=0.011 数量级低于 T5.generate SID 协议)
**Stage**: ANN dense retrieval 3-arm (Arm A HNSW / Arm B IVF-PQ / Arm E dense baseline)
**Anchor**: task194_k0256 R@10=0.1053 ⭐⭐⭐ (T5.generate SID 协议, 不可直接比较)

## 决策依据

**ANN dense retrieval 协议 vs T5.generate SID 协议 — 数量级差异**:

| 协议 | R@10 (baseline) | 说明 |
|------|----------------|------|
| **T5.generate SID** (task84 baseline) | **0.1020** | T5 生成 SID 序列 → 查 SID→item_id 表 |
| **ANN dense retrieval** (task324 Arm E) | **0.0109** (-89%) | mean-pool history → faiss top-K → check target |

**根因**: 两种协议的 query 完全不同. T5.generate SID 用 learned autoregressive 生成 (隐式利用 SID 离散码本 + sequence context), ANN dense 用 history item embeddings 的 mean pool (信息损失巨大). 这是**协议本身差异**, 不是召回算法差异.

## 3-arm 结果 (CPU faiss, max_k=100)

| Arm | R@5 | R@10 | R@20 | NDCG@10 | vs E_dense | elapsed |
|-----|------|------|------|---------|-----------|---------|
| **E_dense (baseline)** | 0.0066 | **0.0109** | 0.0154 | 0.0048 | — | 25.4s |
| **A_hnsw** | 0.0066 | **0.0109** | 0.0153 | 0.0048 | 0.0000 | 10.3s |
| **B_ivf_pq** | 0.0033 | **0.0050** | 0.0084 | 0.0022 | **-54.2%** ❌ | 7.8s |

**HNSW 完全等价 dense** (R@10 0.0109 = 0.0109): 在 9922 items 量级上, brute-force dense retrieval 已足够快 (CPU 25s 全量), HNSW 没加速优势也没质量优势. 这是 Malkov & Yashunin 2018 TPAMI 论文预期 — HNSW 主要优势在 ≥ 1M 量级.

**IVF-PQ 显著退化 -54%** (R@10 0.0050): Product Quantization 压缩 768d → 64×8 bits, 在 sentence-t5-base 这种连续高维 embedding 上破坏语义. PQ 假设向量分布近似高斯, 但 T5 embedding 不满足此假设.

## 反证 (R11.3 透明)

1. **H1 (Stage 4 召回改造 = 真正新思路)**: PARTIALLY FALSIFIED for ANN dense protocol. ANN dense protocol 本身 R@10 = 0.011 数量级远低于 T5.generate SID 协议 0.10. 协议差异是主因, 不是召回算法. Arm A/B 在同协议下 NO-GO.
2. **H2 (HNSW = 最可能突破点)**: REFUTED. HNSW 在 9922 items 量级 ≡ dense, 无优势.
3. **H3 (Cross-encoder rerank = 边界改进)**: 仍待验证 — Arm C 需要 T5.generate SID 协议, 跟 ANN dense protocol 不同. **后续在 GPU 释放后跑**.
4. **H4 (Beam search 改造 = 简单变量)**: 仍待验证 — Arm D 需要 T5.generate SID 协议. **后续在 GPU 释放后跑**.
5. **整体**: ANN dense protocol 是 NO-GO; T5.generate SID protocol (Arm C/D/E) 仍待 GPU 验证.

## 关键决策点 (R11.5 自主决策)

1. **协议差异决策**: Issue #39 草案将 HNSW/IVF-PQ (dense retrieval) 跟 cross-encoder rerank / beam search (T5.generate SID) 混在同一 5-arm 设计里, **protocol 不一致**. 我必须分 protocol 跑:
   - **Protocol 1 (ANN dense)**: Arm A/B/E — 本 verdict 已完成, ❌ NO-GO
   - **Protocol 2 (T5.generate SID)**: Arm C/D/E — 待 GPU 释放后跑 (~3 GPU-hours)
2. **R7 决策**: ANN dense 3-arm 不需要 GPU (faiss CPU 跑), 立即跑不抢 task320 GPU. ✅
3. **R11.4 critical**: Issue #39 草案需要修订 (R11.5 自主决策修订 protocol split).
4. **决策阈值修订**: ANN dense protocol 不能跟 task194 anchor 直接比较 (协议不同). T5.generate SID protocol (Arm C/D/E) 才能跟 anchor 直接比较.

## 关联

- Issue #39 (owner 2026-07-29 起草, protocol split 需要修订)
- Issue #38 (Stage 3 训练协议, task320 在跑)
- Task #318 (Issue #38 Arm 1 optimizer, K=100 amplifier 非 universal)
- Task #316/317/319 (Stage 4 post-process rerank 3-arm 全 NO-GO, 同为 R@10 不可比 anchor 的协议)
- Task #243 (Stage 3 T5-mini 200 epoch ckpt 复用, 待 GPU 释放跑 Arm C/D)
- Task #278 (12 ckpt batch Stage 4 eval, dense retrieval only)

## 后续动作

1. **写 protocol split 修订**: Issue #39 必须分 Protocol 1 (ANN dense, 已 NO-GO) + Protocol 2 (T5.generate SID, 待 GPU) 两阶段评估
2. **GPU 释放后跑**: Arm C (rerank) / Arm D beam=100 / Arm D beam=200 / Arm E (T5.generate SID control, beam=50) — 这 4 个 arm 在 T5.generate SID 协议下才能跟 task194 anchor 0.1053 直接比较
3. **Issue #39 verdict 完整化**: 等 Arm C/D 跑完后写 Part 2 verdict 合并两部分结论

## R11.3 透明

- 选 ANN dense 协议 3-arm (E/A/B) 不需要 GPU 是 R7 + R10 兜底最优解, 不抢 task320
- 不重跑 HNSW IVF-PQ 在更大 K (K=200/500) — 因为 baseline R@10=0.011 协议本身问题, 扩大 K 不会改进
- 不换协议 (T5.generate SID 不适合 ANN dense retrieval) — protocol 本身差异, 算法差异在协议框架下

result: Task #324 Part 1 (ANN dense 3-arm) NO-GO. Protocol 本身 R@10=0.011 数量级远低于 T5.generate SID 协议 R@10=0.10. HNSW ≡ dense (R@10=0.0109), IVF-PQ -54% 退化 (R@10=0.0050). Issue #39 protocol split 修订: ANN dense (本 verdict 完成, NO-GO) + T5.generate SID (Arm C/D/E 待 GPU 释放后跑). 决策阈值 R@10>0.1053 仅适用于 T5.generate SID protocol