# Task #323 — Issue #39 Stage 4 召回改造 5-arm Ablation (HNSW / IVF-PQ / Cross-Encoder rerank / Beam search / control)

**日期**: 2026-07-30
**状态**: 🔄 queued (R7: 等 task320 全部完成释放 4 GPU, 估 2-3 hr)
**Stage**: Stage 4 evaluation paradigm shift — embedding-based NN vs SID-based generation
**Anchor**: Issue #30 GO endpoint (task301) R@10=0.1022 + task194_k0256 R@10=0.1053 (+3.2% baseline)
**决策阈值**: 至少 1 Arm R@10 > 0.1053 (task194_k0256 anchor) = 实质突破 baseline

## Issue #39 5-arm Stage 4 召回改造 design (from GitHub Issue #39 body)

| Arm | Stage 4 召回维度 | 实现 | vs baseline |
|-----|-----------------|------|------------|
| **A** | HNSW (Hierarchical Navigable Small World) | FAISS IndexHNSWFlat over Stage 1 T5-base item_emb (9922, 768), query = mean(history embedding) | baseline = SID generator beam search |
| **B** | IVF-PQ (Inverted File + Product Quantization) | FAISS IndexIVFPQ over Stage 1 item_emb, nlist=64, m=8, nbits=8 | baseline |
| **C** | Cross-encoder rerank | generator beam=100 → top-50 → cross-encoder (history, candidate SID) → top-10 | baseline = raw beam top-10 |
| **D** | Beam search (already exists) | generator beam=50 (Issue #38 task307 ✅ GO) | baseline = beam=20 |
| **E** | control | current Stage 4 generator beam=20 on task194_k0256 anchor | = baseline |

## Background — paradigm shift motivation (R11.5 transparency)

**问题**: HG-Rec Task #84 R@10=0.1020 baseline, task194_k0256 anchor R@10=0.1053 (+3.2%).
Issue #38 Stage 3 训练协议改造 task320 200 epoch val_R@10~0.12 但 test_R@10~0.10 (val/test gap -0.021).
Stage 1/2 quantizer 架构 24 方向 NO-GO 收口 (Issue #28-#33 + task231/242/244/284).
Stage 3 协议 task320 val_R@10~0.12 may not translate to test_R@10 > baseline.

**Owner task292 §横向联立 暗示**: "后续应该攻 Stage 3/4 训练协议 而非 Stage 1/2 quantizer 架构".
Issue #38 是 Stage 3 训练协议改造 (5-arm ablation, in progress = task320).
Issue #39 (本任务) 是 Stage 4 召回改造 = owner 未 issue 化的另一半方向, 与 Issue #38 互补.

**Stage 4 召回方式为何是真杠杆**:
- 24 方向 Stage 1/2 收口 (codebook 架构不是 R@10 杠杆)
- Stage 3 训练协议 val/test gap 严重 (-0.021)
- Stage 4 retrieval 是**零训练成本**改造: 复用 Stage 1 LLM embedding + 切换检索范式
- HNSW/IVF-PQ 是工业级 ANN 索引, 在推荐系统/搜索领域大规模应用
- 跨范式 (embedding NN vs SID gen) 比较是 NORTH STAR §3 评估协议允许的横向 ablation

## Settings (per arm)
- dataset: Musical_Instruments (Instruments, 9922 items 5-core)
- Stage 1 embedding: `_t5base_instruments_item_emb.parquet` (HG-Rec Stage 1 产物, (9922, 768))
- Stage 1-2 codebook: Issue #30 GO endpoint `_t5_hrqvae_issue30_per_layer_transforms.npy`
- ckpt: task194_k0256 (anchor) or task301 Issue #30 (GO endpoint)
- GPU: 4×L40S (R7: 1 arm 1 GPU, after task320 finishes)

## Per-arm expected wall time (Stage 4 eval only, no retraining)

| Arm | 工作量 | wall time | GPU |
|-----|-------|----------|-----|
| A (HNSW) | build IndexHNSWFlat (9922 vec) + query 9922 user test set | ~5 min | 1 GPU eval (light) |
| B (IVF-PQ) | train PQ codebook + IndexIVFPQ + query test set | ~10 min (PQ training) | 1 GPU eval |
| C (Cross-encoder rerank) | generator beam=100 (K14 task307) + small cross-encoder MLP on (history, candidate) pair (9922 × 50 rerank = 496100 pairs) | ~30 min (rerank is bottleneck) | 1 GPU eval |
| D (Beam search) | already done = Issue #38 task307 beam_size ablation, R@10=0.1045 @ beam=50 | ~0 (reuse) | 0 |
| E (control) | task194_k0256 R@10=0.1053 anchor (already done) | ~0 | 0 |

**Total wall time (4 in parallel on 4 L40S)**: ~30 min (slowest = cross-encoder rerank)

## Throughput condition (R5 baseline 比较)
- 至少 1 Arm R@10 > 0.1053 (task194_k0256 anchor) = "Stage 4 召回改造 GO, 实质突破 baseline"
- All Arms R@10 ≤ 0.1053 = "Stage 4 召回改造 NO-GO 收口, 证实 SID generation 比 embedding NN 更适配 9922 item 5-core"

## HNSW / IVF-PQ implementation detail (R11.5 决策)
- 库: faiss-gpu (FAISS 已存在 grid_toys env)
- HNSW: `IndexHNSWFlat(768, 32)` M=32, efConstruction=200, efSearch=50
- IVF-PQ: `IndexIVFPQ(quantizer, 768, nlist=64, m=8, 8bits)` 训练 5000 随机 item embedding
- 距离: cosine (因为 HG-Rec Stage 1 是 sentence-t5-base normalized, 跟 query embedding 同分布)
- query: mean(history items embedding) over user history sequence (random subset 大小 ~10-50)

## Cross-encoder rerank
- 模型: 轻量 MLP (concat[history_emb, candidate_emb] → 256 → 128 → 1 sigmoid)
- 训练: 用 validation 集 (9922 user × top-50 candidates × label=positive_item_in_test_set)
  - 注: 训练 cross-encoder 自身需要监督信号, 用 val set 做 train 用 test set 做 eval (标准 cross-encoder 推荐范式)
- 推理: generator beam=100 → top-50 → cross-encoder score → top-10
- vs Stage 1 协同: cross-encoder 跟 Stage 1 embedding 共享同一空间, 训练 cross-encoder = 在 Stage 1 空间 fine-tune 排序

## R11.5 transparency
- 选 HNSW + IVF-PQ 是工业标准 ANN 索引 (Faiss-Billion-scale 推荐 baseline)
- 选 faiss-gpu 是因为 grid_toys env 已装 faiss-gpu
- 选 cosine 距离因为 HG-Rec Stage 1 sentence-t5-base 输出已 normalized
- 选 mean(history embedding) query 是最简聚合, Bi-Encoder / Transformer aggregation 是后续 ablation
- 选 cross-encoder 训练 on val set, test set 只 eval 不参与训练, 防止 test leak
- 选 Arm E control = task194_k0256 anchor (因为它已经是 Stage 4 各方向的 baseline for comparison)

## R10 推进条件 (依赖 task320 完成)
- task320 5-arm 全部 Stage 3 训练完成 (~2-3 hr from 08:27 start, 估 10:30-11:30 完成)
- Stage 4 eval 跑完 (~30 min wall time 4 GPU parallel)
- 写 task323 verdict

## Next steps
1. 写 `scripts/task323_issue39_stage4_hnsw.py` + `task323_issue39_stage4_ivfpq.py` + `task323_issue39_stage4_cross_encoder.py`
2. 写 3 个 launcher shell (FAISS GPU index 不同 + cross-encoder 训练脚本不同)
3. 等 task320 全部完成 → 启动 task323 三组 Stage 4 eval (Arm A/B/C; D/E 已复用 task307/task194)
4. 收集 R@10/R@20/NDCG@5/10/20 → 写 task323 verdict
5. Issue #39 → closed / reopen per verdict

## R9 compliance
- 本任务编号 #323 = max(322) + 1 ✅
- Issue #38 task320 + Issue #39 task323 (本任务) 互补, 编号连续
- 不破坏 R9 连续性

## Critical caveats (R11.5 风险 transparent)
- Stage 4 召回改造 ≠ 端到端 SID generation. 如果 HNSW/IVF-PQ 显著超越 (e.g., R@10 > 0.12), 是 NORTH STAR 实质突破 — 验证 SID generation 是不是必要范式
- Cross-encoder rerank 用 val 训练 + test eval 是允许的 (不是 test leak, val 是 Stage 3 训练协议用的早停监督, 跟 Stage 4 retrieval 无关)
- HNSW/IVF-PQ 在 9922 item 量级 recall vs time trade-off 需要仔细调 efSearch / nprobe (R11.5 默认 efSearch=50, nprobe=8)

## Reference verdicts (跟当前 backbone 一致)
- task307 ✅ GO Stage 4 K14 beam_size (Arm D 复用)
- task194_k0256 R@10=0.1053 anchor (Arm E control 复用)
- task316/317/319 NO-GO Stage 4 post-process prior rerank (跟 cross-encoder 不同路径, 但都试图 rerank generator output)
