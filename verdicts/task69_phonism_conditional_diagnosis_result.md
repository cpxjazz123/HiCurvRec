# Task #69 — Phonism 4-seed 条件化诊断 (Conditional Re-test)

> **任务目的**: 验证历史 Mantel/kNN 分析是否因 global pairwise distance 而稀释，导致 L2/L3 层拓扑保留的低估
> **完成日期**: 2026-07-20
> **状态**: ✅ Step 0-5 完成, 关键发现写入 memory

---

## 1. 数据来源

- **Task #126 phonism RQ-VAE** (sentence-t5-base 768d embedding + neural RQ-VAE)
- SID: `/home/wlia0047/ar57/wenyu/GeneRec/products/task126_phonism_rqvae/semantic_ids.pt` (11924, 3)
- Embedding: `/home/wlia0047/ar57/wenyu/genrec/dataset/amazon/processed/toys/item_emb_sentence-t5-base.parquet` (11924, 768)
- TIGER prediction: `/home/wlia0047/ar57/wenyu/GeneRec/products/task87_tiger_baseline/stage4_tiger_infer/merged_predictions_dedup.pt`

---

## 2. Step 0: Codebook 预检

| 层 | 唯一 codes / 256 | 覆盖率 |
|----|-----------------|--------|
| L0 | 172 | **67.19%** ✅ |
| L1 | 69 | **26.95%** ⚠️ < 50% 阈值 |
| L2 | 205 | **80.08%** ✅ |
| **3-token 联合** | **11607 / 11924** | **97.36%** ✅ |

**结论**: L1 单独 < 阈值但 3-token 联合唯一率 97.36%, 进入 Step 1-5 (transparently 标注 L1 弱).

---

## 3. Step 1: Global Baseline (Version A 复现)

| 指标 | Value |
|------|-------|
| Mantel L1 ρ (n=2000) | 0.123 |
| Mantel L2 ρ (n=2000) | **0.085** |
| Mantel L3 ρ (n=2000) | **0.079** |
| kNN L1 hit rate (k=10) | **47.56%** |
| kNN L2 hit rate (k=10) | **6.37%** |
| kNN L3 hit rate (k=10) | **0.34%** |

L3 hit rate 0.34% 极低, 强稀释信号.

---

## 4. Step 2: L1 Bucket 分布

- 172 个 buckets (与 L0 coverage 一致)
- Size: min=12, max=241, **mean=69.3, median=64**
- Percentiles: 10%=30, 50%=64, 90%=110, 99%=184
- **所有 172 个 buckets 都 ≥ 5 items, 无需排除**

每个 bucket 大小适中, 适合做桶内 Mantel + kNN.

---

## 5. Step 3: Conditional (Within L1 Bucket) — Version B

| 指标 | Global | Conditional (L1 桶内) | Δ |
|------|--------|----------------------|---|
| Mantel L2 ρ | 0.085 | **0.131** | **+0.046** (54% ↑) ✅ |
| Mantel L3 ρ | 0.079 | **0.039** | **-0.040** (50% ↓) ❌ |
| kNN L2 hit rate | 6.37% | **8.65%** | +2.28pp |
| kNN L3 hit rate | 0.34% | **0.56%** | +0.22pp |
| kNN L1 hit rate | 47.56% | 100% (常量) | — |

**关键发现**:
- **L2 稀释假设: 部分确认**. Conditional ρ 提升 54% (0.085 → 0.131), 验证 global Mantel 对 L2 有稀释.
- **L3 稀释假设: 否证**. Conditional ρ **下降** (0.079 → 0.039), L3 在 L1 桶内是 **噪声**.

---

## 6. Step 4: Collision Rate (TIGER Independent Signal)

### Catalog 层 collision (item SID table)

| 层 | Top-5 最高频 codes | 频次 |
|----|-------------------|------|
| L0 | 62, 174, 175, 119, 23 | 241, 194, 180, 174, 167 |
| L1 | **150, 252, 38, 56, 14** | **453, 379, 378, 376, 376** |
| L2 | 122, 104, 158, 197, 194 | 80, 79, 78, 78, 78 |

L1 collision 极重: top-1 code (150) 含 **453 个 items** — 单一 L1 macro-category 占据 catalog 3.8%.

### TIGER prediction 层 collision

- 9706 users, top-10 predictions per user
- **Mean unique L1 codes in user top-10: 5.37 / 10 = 53.6%**
- 即: 用户推荐列表里 **46.4% 的 item 共享 L1 code** (与另一个推荐重复 macro-category)

---

## 7. Step 5: 最终结论

### 主要结论

1. **L2 稀释部分确认**: Global Mantel ρ=0.085 被稀释 → Conditional ρ=0.131 (真实相关性提升 54%)
2. **L3 稀释否证 + L3 silent failure 确认**: Conditional ρ 不升反降 (0.079 → 0.039), **L3 digits 实际是噪声**
3. **97.36% unique SID 是必要不充分条件**: 高唯一率掩盖了 L3 局部拓扑失败 — "silent failure"
4. **TIGER 推荐层有 46% L1 重复**: 即使 TIGER 端到端 R@5=0.0200, 推荐列表里 macro-category 重复率仍高

### 启示

- **下游模型如果依赖 L3 区分细粒度相似性**, 实际看到的是 **几乎随机** 的 codes
- **L1 在承担主要信号**: 53.6% 推荐 diversity, 单一 code 容纳 453 个 item, 表明 L1 macro-category 是主要的信息载体
- **未来工作方向**: 调查为什么 L3 Mantel 在 L1 桶内反而下降 (可能: L3 没捕捉到 L2 之后的残余方差)

### 对历史结论的修正

- 历史 L3 Mantel/kNN 结论不是 "稀释假象", 而是 **接近真实** — L3 在 phonism RQ-VAE 里 **确实没保留拓扑**
- 历史 L2 Mantel/kNN 结论 **确实被稀释** — 真实 L2 信号是历史的 1.5× 强
- 但绝对值仍弱 (ρ=0.131), 不能过分强调 L2 的作用

---

## 8. 产物清单

- `result/task69_step1_global/baseline.json` — global Mantel + kNN
- `result/task69_step23_conditional/result.json` — conditional 分析 + bucket 分布
- `result/task69_step45_verdict/verdict.json` — collision rate + final summary
- `scripts/task69_step1_global_baseline.py`
- `scripts/task69_step23_conditional.py`
- `scripts/task69_step45_verdict.py`

---

result: Task #69 条件化诊断完成. L2 稀释部分确认 (Mantel ρ 0.085→0.131), L3 稀释否证且 L3 是 silent failure. TIGER top-10 仅 53.6% L1 diversity. 历史 L2 结论需修正 (实际信号更强), L3 结论接近真实. 数据来源 Task #126 phonism RQ-VAE / sentence-t5-base 768d.
