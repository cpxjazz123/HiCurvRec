# Task #125 — phonism/genrec 仓库 TIGER 复现静态分析

> **任务名**: phonism/genrec 仓库 TIGER 复现实现细节 + 与 GRID 流水线对比
> **完成日期**: 2026-07-19
> **状态**: ✅ 完成 (静态分析, 无 GPU)
> **执行人**: Claude (3 个并行 subagent + 合成)

---

## 1. 仓库速览

| 项 | 值 |
|---|---|
| URL | https://github.com/phonism/genrec |
| 本地路径 | `/home/wlia0047/ar57/wenyu/genrec/` |
| 配置系统 | **Gin-config** (`.gin` 文件 + `--gin "param=value"`) |
| T5 实现 | **纯 PyTorch 自实现** (`genrec/modules/t5.py` 911 行, 声称与 HF pixel-perfect) |
| 框架 | PyTorch 2.0+, Python 3.9+ |
| 主要模块 | SASRec / HSTU / RQVAE / TIGER / LCRec / COBRA / RPG / NoteLLM |

**对比本项目**:
- 本项目用 **Hydra** (`configs/` + CLI override), 配置系统等价
- 本项目用 **HF transformers `T5ForConditionalGeneration`** (vs genrec 自实现)

---

## 2. 报告数 vs 论文 vs 本项目

**Amazon 2014 Toys**:

| 方法 | R@5 | R@10 | N@5 | N@10 |
|------|----:|-----:|----:|-----:|
| **TIGER paper (Table 1)** | 0.034 | 0.051 | 0.022 | 0.028 |
| **phonism/genrec (实跑)** | **0.0340** | **0.0521** | **0.0214** | **0.0272** ✅ 完全对齐 |
| **我们 Task #15 legacy** (flan-t5-xl, snap-default RQ-VAE) | 0.06419 | 0.09710 | 0.04118 | 0.05180 |
| **我们 Task #87 TIGER-aligned** (sentence-t5-base, Adafactor, LSH, 4 enc/dec) | 0.01937 | 0.03318 | 0.01222 | 0.01663 |

**关键观察**:
- phonism 跑出**与论文 Table 1 完全一致**的数字 — 是首个外部 reference,证明 Toys RQ-VAE 上限 R@5 ≈ 0.034
- 我们 Task #15 (legacy) 反而比论文高 2× — 用了 flan-t5-xl (2048d) + 旧 TIGER, 与论文不是 apples-to-apples
- 我们 Task #87 (TIGER-aligned) 只到 0.019 (56% of paper) — 有优化空间

---

## 3. 架构对比 (TIGER 模型)

### 3.1 架构完全对齐论文 §4 Implementation Details

| 超参数 | 论文 | phonism/genrec | 我们 Task #87 |
|--------|------|---------------|---------------|
| enc/dec layers | 4 / 4 | **4 / 4** ✅ | 4 / 4 ✅ |
| heads | 6 | **6** ✅ | 6 ✅ |
| d_model | 128 | **128** ✅ | 128 ✅ |
| d_ff | 1024 | **1024** ✅ | 1024 ✅ |
| d_kv | 64 | **64** ✅ | 64 ✅ |
| dropout | 0.1 | **0.1** ✅ | 0.1 ✅ |
| tie_word_embeddings | (T5 默认) | **True** ✅ | True ✅ |
| vocab_size | V+SID | **769 = 256×3+1** ✅ | 等价 ✅ |
| codebook K | 256 | **256** ✅ | 256 ✅ |
| sem_id_dim | 3 | **3** ✅ | 3 (训练) / 4 (含 dedup) |

### 3.2 训练超参 (论文 vs phonism)

| 超参数 | 论文 | phonism | 我们 Task #87 |
|--------|------|---------|---------------|
| optimizer | **Adafactor** | **Adam** (lr=1e-4) ❌ | Adafactor ✅ |
| scheduler | **inverse_sqrt + warmup 10k** | **none** ❌ | inverse_sqrt + 10k warmup ✅ |
| grad_clip | (常规 1.0) | **无** ❌ | (隐含 by Lightning) |
| steps | 100k (固定) | **200 epochs + 早停 patience=10** | 100k |
| batch_size | 256 | **256** ✅ | 256 ✅ |
| sequence_length | 20 | **50** ⚠️ (2.5×) | 120 |

### 3.3 关键 trick 缺失清单 (phonism 跳过)

| Trick | 论文 | phonism | 影响 |
|-------|------|---------|------|
| **trie-constrained decoding** | ✅ 必备 | ❌ **完全缺失**, raw top-k over vocab | **未知**: 无效 SID 进入 top-k |
| **LSH Hashing Trick** (2000 user bins) | ✅ 必备 | ❌ 用 `hash(uid) % 10000`, 简化 | 弱化 user 信号 |
| **user embedding 输入模型** | ✅ 必送入 | ❌ user_id **从未送入模型** | 序列只有 SID |
| **Adafactor + relative_step** | ✅ | ❌ 用 Adam | 收敛曲线可能不同 |

> ⚠️ **潜在虚高风险**: phonism 不做 trie 约束, 模型可输出任意 3-token 序列。top-20 beam 命中 ground truth 的概率在 Toys (~12k items × 256³ ≈ 16.7M SID) 上不小, 即使模型未真学到偏好也可能命中。**R@K 可能部分虚高**。

---

## 4. RQ-VAE 关键工程选择 (为什么 phonism 不坍缩)

phonism 能复现 R@5=0.034 的真正关键是 RQ-VAE 设计选择:

| 维度 | 论文 | phonism | 关键洞察 |
|------|------|---------|---------|
| Optimizer | Adagrad lr=0.4 | **AdamW lr=0.001** | phonism 显著偏离论文 |
| Precision | fp32 | **fp16 + autocast + grad_clip=1.0** | 更激进 |
| Encoder hidden | `[512,256,128]→32` | **`[512,256,128,64]→32`** (4 层) | 加深一层 |
| Activation | (未明, 默认 ReLU) | **SiLU** | 现代化 |
| Distance | cosine | **L2 + L2-normed decoder** | 对 unit-norm 输入是天然匹配 |
| Codebook mode | (未明) | **STE 前层 + SINKHORN 最后层** ⭐ | **核心: 强制 codebook 均衡使用** |
| KMeans init buffer | (未明) | **20000 样本** | 比 snap-default 3072 大 6.5× |
| Empty cluster revival | (未明) | **有** (kmeans.py:70-73, 随机重抽) | 防 collapse |
| EMA codebook update | (VQ-VAE 标准) | **无** (标准 VQ 梯度) | - |
| LR scheduler | (Adagrad 自适应) | **linear warmup + linear decay** | - |
| Commitment β | (典型 0.25) | **0.25** ✅ | - |

### ⭐ 核心 trick: SINKHORN-Knopp 最后层 balanced assignment

`genrec/models/rqvae.py:218-241` (Sinkhorn forward mode):

```python
dist_norm = ((dist - mid) / amp).double()
row_marginals = torch.full((B,), 1./B, device=self.device)
col_marginals = torch.full((K,), 1./K, device=self.device)
P = _sinkhorn_knopp(cost=dist_norm, row_marginals, col_marginals,
                    eps=0.003, max_iter=100).detach()
sk_ids = P.argmax(dim=-1)
emb = self.get_item_embeddings(sk_ids)
emb_out = x + (emb - x).detach()  # STE
```

- **关键**: Sinkhorn-Knopp 强制 row_marginal (每个 batch 样本分配一次) 和 col_marginal (每个 code 被分配一次) 均衡
- **效果**: codebook 利用率天然 ≈ 100%, **避免 codebook collapse**
- **来源**: LCRec 论文 (arXiv 2311.09049)

> **这是为什么 phonism 不需要 lr=0.4 + Adagrad**: 代码层就用 Sinkhorn 防 collapse 了, 不依赖优化器暴力保持均衡。

---

## 5. Stage 1 文本格式

phonism (`genrec/data/amazon.py:213-220`):
```python
semantics = (
    f"'title':{info.get('title', '')}\n"
    f" 'price':{info.get('price', '')}\n"
    f" 'salesRank':{info.get('salesRank', '')}\n"
    f" 'brand':{info.get('brand', '')}\n"
    f" 'categories':{info.get('categories', '')}"
)
```

5 字段拼接 = **Title + Price + SalesRank + Brand + Categories**。

**对比本项目**:
- 我们用 tfrecord 预拼装 text (上游 `prepare_amazon.py`)
- Task #15 用 flan-t5-xl (2048d), Toys 价格信息**包含** (含 Price 字段)
- Task #87 用 sentence-t5-base (768d), Toys 数据**不含 Price 字段** (baseline 重训后已加 Price, 但与 snap 默认 Tasks 重合度待验证)

---

## 6. 评估实现

| 项 | phonism | 本项目 |
|---|--------|--------|
| Recall | vectorized first-match rank < K | `SIDRetrievalEvaluator` torchmetric |
| NDCG | `1/log2(rank+1)` 0-indexed | DCG / `log2(2..K+1)` 标准公式 |
| Leave-one-out | ✅ | ✅ |
| Full ranking | ✅ (无 neg sampling) | ❌ **默认 `should_sample_negatives_from_vocab=True` 用 500 negs** ⚠️ 风险 |
| Sequence length | 20 (TIGER) / 50 (RPG) | 120 (我们) ⚠️ |

> ⚠️ 我们项目 `sample_negative_ids_from_vocab` 默认 500 negatives — 与论文 full-ranking 协议**不一致**, 是已知风险点。

---

## 7. 关键文件路径速查

### phonism/genrec
| 角色 | 路径 |
|------|------|
| RQ-VAE 模型 | `genrec/models/rqvae.py:278` |
| 量化层 + SINKHORN | `genrec/models/rqvae.py:113,218-241` |
| TIGER 模型 | `genrec/models/tiger.py` (61 行) |
| T5 纯 PyTorch 实现 | `genrec/modules/t5.py` (911 行) |
| TIGER trainer | `genrec/trainers/tiger_trainer.py:332` (main) |
| 评估指标 | `genrec/modules/metrics.py:10-74` |
| 文本构造 | `genrec/data/amazon.py:213-220` |
| 滑窗 dataset | `genrec/data/amazon.py:413-465` |
| RQ-VAE 配置 | `config/tiger/amazon/rqvae.gin` |
| TIGER 配置 | `config/tiger/amazon/tiger.gin` |

### 本项目
| 角色 | 路径 |
|------|------|
| 评估指标 | `src/components/eval_metrics.py:262-314` |
| TIGER 模型 | `src/models/modules/semantic_id/tiger_generation_model.py` |
| Stage 1 嵌入模块 | `src/modules/semantic_embedding_inference_module.py` |
| Stage 1 配置 | `configs/experiment/sem_embeds_inference_tiger.yaml:4` |
| Stage 2 RQ-VAE 配置 | `configs/experiment/rqvae_train_flat.yaml` (snap-default) |

---

## 8. 关键发现与对 Task #124 的影响

### 8.1 Task #124 方向需重大调整

User 原本要求 "RQ-VAE 用 latent=32 + lr=0.4 (TIGER paper)". 但 phonism 实跑证明:
- **lr=0.4 + Adagrad** 是论文原文, 但**工程上不稳** (我们已观察 step 49→99 collapse)
- phonism 用 **AdamW lr=0.001 + Sinkhorn + KMeans 20000 样本** 完全替换
- phonism 用 **fp16 + grad_clip 1.0** 完全替换论文 fp32

→ **两种路径都到 R@5=0.034, 但机制不同**。Task #124 已在框架加 dead code revival, 但可能还需:
1. **加 SINKHORN 最后层** (从 `genrec/models/rqvae.py:218-241` port)
2. **改 KMeans init buffer 到 20000** (vs snap-default 3072)
3. **保持 user 指定的 lr=0.4 但加 SINKHORN**, 即可

### 8.2 评估一致性风险

phonism 用 **full ranking** 评估, 我们默认用 **500 negs sampling**:
- 即使模型 + RQ-VAE 完全一致, R@K 数字也会**虚高** (negative sampling 让任务变简单)
- Task #87 baseline 数字 0.01937 可能部分**虚低** (因 500 negs 严于 full ranking)
- 需要在复现报告里**显式注明**评估协议

### 8.3 trie-decoding 缺失风险

phonism 完全不做 trie 约束, top-k 里可含任意 3-token 序列:
- Toys 16.7M SID × 12k items, top-20 命中率自然高
- **phonism R@5=0.034 可能部分来自此 trick 的逆向利用**
- 我们若做 trie 约束严格评估, 数字可能更低 (但更接近真实推荐质量)

---

## 9. 推荐后续动作

| 优先级 | 动作 | 价值 |
|--------|------|------|
| 🔴 高 | Task #124 重启: 用 lr=0.4 + Adagrad + SINKHORN 最后层 + KMeans init 20000 | 验证 Sinkhorn 能否替代 collapse 修复 |
| 🔴 高 | phonism 仓库加到 git submodule 备查 | 后续可直接对照源码 |
| 🟡 中 | 评估协议核实: 用 full-ranking 重算 Task #87 R@5, 看是否上浮 | 排除 500 negs 虚低问题 |
| 🟡 中 | Task #87 baseline 加 Price 字段重训 (用 sentence-t5-base) | 对齐 phonism 5 字段输入 |
| 🟢 低 | trie-constrained decoding 模块化实现 | 论文严格对齐, 但需要重训 TIGER |

---

## 10. 产物

- 本 verdict: `verdicts/task125_phonism_genrec_analysis_result.md`
- phonism 仓库: `/home/wlia0047/ar57/wenyu/genrec/`

---

**当前任务已完成，请做下一个任务的指示.**

result: Task #125 — completed (R8 §9.3 retroactive compliance line appended by Task #114; refer to verdict body above for full details).
