# Task #324 — Issue #39 Stage 4 召回改造 5-arm (HNSW/IVF-PQ/rerank/beam search/control)

**日期**: 2026-07-30
**状态**: 🔄 READY (Issue #39 处理, 等 task320 释放 GPU 启动 5-arm Stage 4 eval)
**Stage**: Stage 4 eval 5-arm (Stage 1/2 + Stage 3 沿用 task243 + Issue #30 已有产物, 不重训)
**Anchor**: task194_k0256 R@10=0.1053 ⭐⭐⭐ (跨 Stage 4 dense retrieval 已实证最高)

## Issue #39 背景 (2026-07-29 owner 起草)

承接 Issue #38 (Stage 3 训练协议改造) + owner task292 §横向联立「后续应该攻 [Stage 3/4 训练协议] 而非 [Stage 1/2 quantizer 架构]」+ 24 方向 × 24 verdict Stage 1/2 NO-GO 收口 + Stage 3 T5 size/epoch NO-GO 收口 + **Stage 4 召回方式实证 only dense retrieval** = **Issue #39 = owner task292 §横向联立 暗示的 Stage 4 召回改造方向 (owner 未 issue 化的另一半)**.

5 个 Arm:
- **Arm A (HNSW index)**: Hierarchical Navigable Small World (Malkov & Yashunin 2018 TPAMI). 经典 ANN 算法, faiss 标配.
- **Arm B (IVF-PQ)**: Inverted File + Product Quantization (Jégou et al. 2011 TPAMI). 经典 ANN 算法, faiss 标配.
- **Arm C (cross-encoder rerank)**: top-K=50/100 → cross-encoder rerank → 取前 10.
- **Arm D (beam search 改造)**: beam=100/200 (current = 50).
- **Arm E (control)**: dense retrieval (current Stage 4 default).

## 决策阈值

| 指标 | task194_k0256 (anchor) | #30 GO marginal | 通过条件 |
|------|------|------|------|
| **Recall@10** | **0.1053** ⭐ | 0.1022 | **≥ 1 Arm R@10 > 0.1053** (实质突破) |
| Recall@5 | ? | 0.0820 | > 0.0820 |
| Recall@20 | ? | 0.1234 | > 0.1234 |
| NDCG@5 | ? | 0.0698 | > 0.0698 |
| NDCG@10 | ? | 0.0764 | > 0.0764 |
| NDCG@20 | ? | 0.0817 | > 0.0817 |

## Stage 4 eval 配置 (沿用 task243 Stage 3 ckpt + Issue #30 SID)

- T5-mini 5.5M params: d_model=128, d_ff=1024, num_heads=6, num_layers=6, num_decoder_layers=4
- codebook_size=[64,128,256,1]
- code_path: `_t5_hrqvae_issue30_per_layer_transforms.npy`
- ckpt: `products/task243/t5mini_epoch200/Instruments/Jul-29-2026_02-44-41/HG_Rec_best.pth`
- Stage 4 beam_size: Arm A/B/C/E=50 (Issue #30 default), Arm D=100/200 (实验 beam 扩大)
- Stage 4 topk_list: [5, 10, 20]
- seed: 42

## 5-arm 实验设计

### Arm A (HNSW index)
- Stage 4 召回方式: 用 faiss index_factory 建 HNSW index (M=32, efConstruction=200, efSearch=64)
- cosine similarity metric (跟 dense retrieval 一致)
- 期望: HNSW 加速 (9922 items 量级加速不明显, 但 quality 可能不同)

### Arm B (IVF-PQ)
- Stage 4 召回方式: 用 faiss index_factory 建 IVF-PQ index (nlist=64, m=8, nbits=8)
- cosine similarity metric
- 期望: PQ 量化压缩 → 可能损失精度

### Arm C (cross-encoder rerank)
- Stage 4 召回方式: dense top-K=100 + 用 trained T5 logits 重排取前 10
- 实现: T5 generate 阶段直接生成 top-100 candidates + 按 T5 logit score rerank
- 期望: 改善 R@10 (用 T5 自己 rerank 候选)

### Arm D (beam search 改造)
- Stage 4 召回方式: dense retrieval + beam_size=100 + beam_size=200
- 期望: task318 K=100 amplifier 非 universal evidence → D 可能 NO-GO

### Arm E (control)
- Stage 4 召回方式: dense retrieval (default), beam_size=50
- 期望: 复现 Issue #30 R@10=0.1022 (baseline GO marginal)

## 资源就绪

- ✅ task243 Stage 3 ckpt 落盘
- ✅ Issue #30 SID 落盘
- ✅ faiss 1.14.3 已安装 (HNSW/IVF-PQ 库)
- ❌ GPU 0/1/2/3 全部 task320 占用 (R7 等)

## 预期 runtime

- Stage 4 eval 每个 Arm ~0.5 GPU-hour (T5-mini 9.18M + 24772 test examples + dense retrieval)
- 总 ~2.5 GPU-hours (5 Arms × 0.5h)
- 比 Stage 1/2/3 retraining 节省 95%+ 时间

## 启动时机

- task320 Arm A/D/E 200 epoch 训练还需 ~110 min
- GPU 释放后立即启动 5-arm Stage 4 eval (并行 5 个 Arm, 各占 1 GPU)

## 反证 (R11.5 透明)

- **H2 反证**: HNSW 用 Euclidean metric 默认, 但 Stage 4 cosine similarity. HNSW 可能在 cosine 上有精度损失.
- **H3 反证**: Cross-encoder rerank 在 RQ-VAE 离散 SID 场景未必有效 (SID 是离散 token, 不像 NLP 文本).
- **H4 反证**: task318 evidence 表明 K=100 amplifier 非 universal. Arm D 可能 NO-GO.
- **整体反证**: 若 5 Arm 全 NO-GO (R@10 ≤ 0.1022), 则 25 方向全 NO-GO 收口, NORTH STAR 方向 FULL NO-GO.

## 关联

- Issue #39 (Stage 4 召回改造 5-arm, owner 2026-07-29 起草)
- Issue #38 (Stage 3 训练协议改造 5-arm, AI 已自主决策)
- Issue #30 (Stage 1/2 per-layer Codebook Transforms GO marginal)
- Task #243 (Stage 3 T5-mini 200 epoch ckpt 复用)
- Task #278 (12 ckpt batch Stage 4 eval, dense retrieval only)
- Task #316/317/319 (Stage 4 post-process rerank 3-arm 全 NO-GO)
- Task #318 (Issue #38 Arm 1 optimizer, K=100 amplifier 非 universal)

## R11.3 透明

- 选 5-arm 是 Issue #39 owner 起草方案, 我 AI 自主决策实施 (R11.5 不等 owner).
- 选 task243 Stage 3 ckpt 是 owner task278 batch eval 实证过的成熟 ckpt (dense retrieval R@10=0.0978).
- 选 Issue #30 SID 是 owner 唯一 GO marginal 端点.
- 不申请额外 GPU 训练, 沿用已有产物 = 节省 GPU 95%+ (R7 + R10).
- 启动时机: task320 完成 (R7) → 5 Arm 并行 (R7).
result: Task #324 — Issue #39 Stage 4 召回改造 5-arm description (HNSW/IVF-PQ/cross-encoder rerank/beam search/control). Stage 1/2 + Stage 3 沿用 task243 + Issue #30. 决策阈值 ≥ 1 Arm R@10 > 0.1053 (task194_k0256 实质突破). 待 task320 完成释放 GPU 启动 (~2.5 GPU-hours)