# Task #73 — ETEGRec 全 Table 3 复现 (含 5 个 Generative Baselines)

> **任务目的**: 复现 ETEGRec paper Table 3 Instrument 列的 6 个缺项 — ETEGRec 自身 + TIGER + TIGER-SAS + LETTER + SID(P5-SID) + CID(P5-CID) — 让我们的 13-baseline 复现表完整闭环, 对齐 ETEGRec 论文 R@10 阈值.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 0. 数据集上下文 (重要 — 与 R5 偏差说明)

⚠️ **本次复现沿用 Task #72 已授权的 Musical_Instruments (而非 R5 默认 Toys)**, 因为:
- ETEGRec 论文 Table 3 报告的就是 Instrument 列
- Task #72 (DECOR 主结果复现) 已决策使用 Musical_Instruments, 理由: DECOR / ETEGRec 是 sequential recommendation 任务, Toys 数据集语义噪声大 (DECOR 论文 Section 4.3 明确说 Instruments 是该方法的"演示用数据集")
- 维持数据集一致性便于和 Task #72 verdict 对照

数据集来源 (Task #72 验证): McAuley 5core/rating_only, **57439 users / 24587 items / 511837 interactions** — 与 paper Table 2 完全一致.

---

## 1. 背景

承接 Task #72 verdict 中的已知缺口:
- ✅ 已复现: Caser, GRU4Rec, SASRec, BERT4Rec, FDSA(skip), S³Rec(skip), HGN(未跑), FMLP-Rec(未跑) + DECOR 自身
- ❌ **未复现** (本次任务目标):
  1. **ETEGRec** (论文主方法) — Phase 4 v4 训练在 Task #72 中断, 无 test result
  2. **TIGER** (RQ-VAE + T5 + flash attention)
  3. **TIGER-SAS** (TIGER variant: 用 SASRec embedding 替代 text embedding)
  4. **LETTER** (LETTER repo, contrastive RQ-VAE + T5)
  5. **SID** (P5 framework, ID prefix tokens)
  6. **CID** (P5 framework, spectral clustering)

每个 baseline 都有 paper Table 3 阈值, 完成度 = 6/6 R@10 ≥ paper 报告值 - 6% (我们 RecBole 经验偏差).

---

## 2. 实验设计 (子任务分阶段)

### 子任务 A — 基础设施搭建 (前置依赖)

**步骤 1**: Clone 官方仓库到 `/home/wlia0047/ar57/wenyu/GeneRec/external/` 目录
```bash
mkdir -p external && cd external
# ETEGRec 官方实现
git clone https://github.com/RUCAIBox/ETEGRec.git ETEGRec
# LETTER (CIKM 2024) — 用于 LETTER baseline
git clone https://github.com/wxshallcheng/LETTER.git LETTER
# P5 / P5-SID / P5-CID 框架 (用于 SID/CID baselines)
git clone https://github.com/jeyao/P5.git P5
# TIGER / TIGER-SAS — DECOR repo 已经引用, 不需额外 clone (使用 src/ 既有 pipeline)
```

**步骤 1 (实际克隆结果, 2026-07-22)**:
- ✅ ETEGRec → `external/ETEGRec/` (github.com/RUCAIBox/ETEGRec)
- ✅ LETTER → `external/LETTER/` (github.com/HonghuiBao2000/LETTER, 原 wxshallcheng/LETTER 404)
- ✅ P5 → `external/P5/` (github.com/jeykigung/P5, 原 jeyao/P5 404)
- ✅ LLM-RecSys-ID → `external/LLM-RecSys-ID/` (github.com/Wenyueh/LLM-RecSys-ID) — 用于 SID/CID baselines (paper [9])
- ⚠️ 注意: P5 自身不实现 SID/CID, 需要 LLM-RecSys-ID 配套

**步骤 2**: 检查现有 Musical_Instruments 数据格式与外部 repo 是否兼容 (字段: user_id / item_id / timestamp / aspect / etc.)
**步骤 3**: 安装依赖 (CUDA 12.x torch, recbole==1.2.1, transformers, accelerate, docling skip)

### 子任务 B — ETEGRec 自身 (论文主方法) ⭐ 最高优先级

**Stage 2.1 — Item Tokenizer (RQ-VAE 预训练)**
- **Variable**: 用 SASRec 256-dim collaborative embedding 作为 RQ-VAE 输入 (per ETEGRec 4.1.4)
- **Stage 配置**: 3 codebooks × 256 codes × 128 dim, 3-layer MLP encoder/decoder
- **训练数据来源**: 我们 Task #72 训练好的 SASRec embedding (saved products/task72/sasrec_emb.pt)

**Stage 2.2 — RQ-VAE 推断**
- 生成 SID tensor (3 + 1 dedup = 4 tokens per item)

**Stage 3 — Generative Recommender (T5 + Alternating Training)**
- **变量**: T5 backbone (6 encoder + 6 decoder layers, hidden=128, FFN=512, heads=4)
- **保持不变**: weight_decay=0.05, AdamW, μ=3e-4, λ=1e-4 (Instrument 最优)
- **Alternating**: 1 epoch tokenizer + 1 epoch recommender per cycle (C=2)
- **LR**: tokenizer ∈ {5e-4, 1e-4, 5e-5}, recommender ∈ {5e-3, 3e-3, 1e-3}
- **种子**: 2025 (paper Section 4.1.2)
- **预算**: ~5-7 GPU 小时 (含 alternating optimization)

**Stage 4 — 推断 + 评估**
- Beam size = 20
- 全 test set 评估 (57439 users × 24587 items full ranking)

**Decision threshold (vs paper)**:
- ETEGRec R@10 ≥ 0.0624 ✅ 主决策达成
- ETEGRec R@10 < 0.0586 (paper -6%) → 接受 (-6% framework 偏差)
- ETEGRec R@10 < 0.05 → ⚠️ 调查根因

### 子任务 C — TIGER + TIGER-SAS (2 baselines)

**Stage 2.1**: 复用 Task #72 训练好的 RQ-VAE + sentence-t5 嵌入 (TIGER 用 text, TIGER-SAS 用 SASRec embedding)
**Stage 3**: T5 seq2seq 训练 (每个 ~2-3 GPU 小时)
- TIGER: text embedding 作为 SID input
- TIGER-SAS: SASRec collaborative embedding 替换

**Decision thresholds**:
- TIGER R@10 ≥ 0.0574 (paper)
- TIGER-SAS R@10 ≥ 0.0576 (paper)

### 子任务 D — LETTER (1 baseline)

**Stage 1**: Clone LETTER repo, 跑它自带的 RQ-VAE + contrastive alignment
**Stage 3**: 训练
**Decision threshold**: LETTER R@10 ≥ 0.0581

### 子任务 E — SID + CID (2 baselines, P5 framework)

**Stage 1**: Clone P5 repo, 安装依赖
**Stage 2**: SID 用 prefix token (按数字 item ID 拆分); CID 用 spectral clustering on co-occurrence graph
**Stage 3**: T5 seq2seq 训练 (P5 框架)
**Decision thresholds**:
- SID R@10 ≥ 0.0438
- CID R@10 ≥ 0.0507

---

## 3. 决策触发 (汇总)

| Baseline | Paper R@10 | 接受阈值 (paper -6%) | 不达标调查方向 |
|----------|-----------|---------------------|---------------|
| **ETEGRec** ⭐ | 0.0624 | ≥ 0.0586 | 调 alternating cycle / μ/λ |
| TIGER | 0.0574 | ≥ 0.0539 | 验证 SID tensor, 调 T5 hidden |
| TIGER-SAS | 0.0576 | ≥ 0.0541 | 验证 SASRec embs 256-d |
| LETTER | 0.0581 | ≥ 0.0546 | 调 contrastive loss coeffs |
| SID | 0.0438 | ≥ 0.0412 | 调 prefix length |
| CID | 0.0507 | ≥ 0.0477 | 调 cluster 数 |

总决策触发: **至少 4/6 R@10 ≥ 接受阈值** 视为子任务组成功.

---

## 4. 预算 (累计 GPU 小时)

| 阶段 | 估算 |
|------|------|
| 子任务 A — 仓库 clone + 依赖 | 1-2 hours (人工) |
| 子任务 B — ETEGRec 训练 (Stage 2 + 3 + 4) | **5-7 GPU hours** |
| 子任务 C — TIGER + TIGER-SAS (Stage 2-4 × 2) | **4-6 GPU hours** |
| 子任务 D — LETTER (Stage 1-4) | **3-4 GPU hours** |
| 子任务 E — SID + CID (P5) | **3-5 GPU hours** |
| Stage 4 评估 (recsys eval) | ~30 min total |
| **总计** | **~16-22 GPU hours** (单 GPU) 或 **~4-6 hours** (4-GPU 并行) |

4 张 L40S 并行: 主路径 ~5 hours. 串行: ~22 hours.

---

## 5. 风险与缓解

**R1**: P5 框架代码可能不支持 Amazon 2023 数据格式 (P5 原版基于 Amazon 2014/2018 5-core)
→ **缓解**: 把 Musical_Instruments 转为 P5 期望的 `*.inter` + `*.feat` 格式, 字段映射: `uid`/`iid`/`time`

**R2**: ETEGRec alternating training 收敛时间长 (paper 没给 epochs, 全 grid search)
→ **缓解**: 先用 μ=3e-4, λ=1e-4, LR=3e-3/5e-4 一组超参, 监控 val R@10 单调性

**R3**: LETTER 仓库依赖冲突 (graphein/prodigy 等)
→ **缓解**: 先读 LETTER repo requirements.txt, 在新 conda env 装

**R4**: 单 GPU 显存 OOM (T5 + ETEGRec 双 encoder/decoder + alternating)
→ **缓解**: batch_size=64, gradient_accumulation=4, 48GB L40S 充裕

**R5**: ETEGRec 仓库 README 没给完整超参 (只给 range) → grid search cost 大
→ **缓解**: 用 task #72 验证过的 SASRec embedding + paper 4.1.5 报告的 grid 中心点

**R6**: SID/CID P5 框架 sequential recommendation 与 generative recommendation 表述不同 (P5 是 multi-task, ETEGRec Table 3 只用 P5 推荐任务)
→ **缓解**: 取 P5 单任务版本 `rec`, 不跑 P5 的 `sequential` multi-task

---

## 6. 完成度跟踪

### 子任务 A — 基础设施
- [x] clone ETEGRec repo (`external/ETEGRec/`) → github.com/RUCAIBox/ETEGRec ✅
- [x] clone LETTER repo (`external/LETTER/`) → github.com/HonghuiBao2000/LETTER ✅ (原 wxshallcheng/LETTER 404, 已用 HonghuiBao2000 替代)
- [x] clone P5 repo (`external/P5/`) → github.com/jeykigung/P5 ✅ (原 jeyao/P5 404, 已用 jeykigung 替代)
- [x] 验证依赖: 3 repo 均无 requirements.txt, README 列依赖. ETEGRec (torch 2.4.0+cu121), LETTER (torch 1.13.1+cu117 + bitsandbytes), P5 (torch 1.10.1 + transformers 4.2.1) — 版本差异大, 需要 3 个独立 conda env 避免冲突. P5 `preprocess/data_preprocess_amazon.ipynb` 验证支持 Amazon 5-core ✅

### 子任务 B — ETEGRec 自身
- [x] **Stage 2.1 (Prior run, Jul-21-2026)**: SASRec 128-dim embedding (`ETEGRec/dataset/Musical_Instruments/Musical_Instruments_emb_128.npy`) → 3-layer MLP RQ-VAE 训练 (256-256-256, 128-dim, beta=0.25)
- [x] **Stage 2.2 (Prior run)**: RQ-VAE 推断 → SID tensor (3 + 1 dedup = 4 tokens)
- [x] **Stage 3 (Prior run)**: Alternating Training (T5 + tokenizer, C=2, 9.7h, epoch 27 early stop)
- [x] **Stage 4 (Prior run)**: 推断 + 评估 (beam=20). R@10=0.025993, R@5=0.015512 ❌ **(-58% vs paper 0.0624)**
- [x] **Stage 2.1+2.2+3+4 (Paper-aligned rerun v1, Jul-22-2026 13:50 启动, PID 1173805)**: lr_rec=0.003, μ=1e-4, λ=1e-4, **batch_size=512 (default yaml) OOM** at epoch 0, 5 min后崩 (CUDA OOM at 41/44 GB). 修复: 改用 batch_size=128 + grad_accum=4 (effective 512, 与 prior run 一致).
- [x] **Stage 2.1+2.2+3+4 (Paper-aligned rerun v2, Jul-22-2026 13:55 启动, PID 1196934, GPU 0)**: 复用 prior 预训练 RQ-VAE, lr_rec=0.003, μ=1e-4, λ=1e-4, batch_size=128, grad_accum=4. 新 run: `myckpt/Musical_Instruments/Jul-22-2026_13-55-XXXXXX/`. 7 GB / 44 GB, 25 it/s, 预计 ~3-5 min/epoch, 27 epochs ≈ 2-3 hours. 预计 16:00 完成.
- [ ] R@10 result ≥ 0.0586 (paper -6%) **TBD (训练中)**

### 子任务 C — TIGER + TIGER-SAS
- [ ] RQ-VAE 训练 (text emb) → TIGER SID
- [ ] RQ-VAE 训练 (SASRec emb) → TIGER-SAS SID
- [ ] T5 训练 × 2
- [ ] Stage 4 评估 × 2
- [ ] 2/2 R@10 ≥ 接受阈值

### 子任务 D — LETTER
- [ ] LETTER repo RQ-VAE + contrastive alignment
- [ ] T5 训练
- [ ] Stage 4 评估
- [ ] R@10 ≥ 0.0546

### 子任务 E — SID + CID (P5)
- [ ] P5 数据格式转换 (Musical_Instruments → P5 .inter/.feat)
- [ ] SID prefix tokenizer + T5 训练
- [ ] CID spectral clustering + T5 训练
- [ ] Stage 4 评估 × 2
- [ ] 1/2 R@10 ≥ 接受阈值

### 收尾
- [ ] 整合 6 baselines 进 Task #72 verdict 表格
- [ ] 写 `verdicts/task73_etegrec_generative_baselines_result.md` (含 6 baseline 数值 + paper 对比 + 偏差分布)
- [ ] 更新 `verdicts/task72_decor_main_repro_instrument_result.md` 加入新 baseline 行 (覆盖"漏跑: TIGER/LETTER/CoST/P5-SID/P5-CID")

---

## 7. 子任务论文核验 (强制, per loop.md §8)

每个 Stage 完成后必须核验:
1. **shape**: SID tensor `(N, 4)`, predictions `(N_users, 20, 4)` (beam=20)
2. **codebook coverage**: SID 每层使用码本比例 ≥ 80%
3. **eval metric**: R@10 ≥ 接受阈值 (per 子任务)

不通过则需修复, 不能直接进下一 stage.

---

## 8. 关联 / 引用

- 前置: Task #72 (DECOR 主结果 + 20 RecBole baselines + Musical_Instruments 数据准备)
- 关联: ETEGRec paper §4.1.4 (Stage 2 配置), §4.1.5 (T5 hyperparams), §4.2 (Table 3 主结果)
- 数据: `data/recbole/Musical_Instruments.inter` (Task #72 准备)
- 外部代码: `external/ETEGRec/` (待 clone), `external/LETTER/` (待 clone), `external/P5/` (待 clone)
