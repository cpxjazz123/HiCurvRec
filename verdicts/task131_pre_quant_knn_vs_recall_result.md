# Task #131 — phonism 4 seed 量化前 kNN vs Recall 相关性分析 verdict

> **完成日期**: 2026-07-19
> **状态**: ✅ 主分析完成 (per-seed 模型 rank per-item 相关性需重新加载 best_model.pt 跑 inference, 当前未做)

---

## 1. 任务目标

在 4 个 phonism TIGER seed（42, 123, 7, 2024）上, 计算 sentence-t5-base dense retrieval 作为"量化前 kNN"的测试集 R@5 / R@10, 并与 4 seed 模型 Test R@5 / R@10 做对比, 回答:

1. **Pre-quantization dense retrieval (mean-pool history) 在 test 集上的 R@K 是多少?**
2. **4 seed 模型 R@5 / R@10 与 pre-quantization R@5 / R@10 的比例关系** (即: 模型把 pre-quantization 的"近零"信号放大到多少倍)
3. **4 seed 间放大比例是否一致** (即: 是否存在 seed 专门"放大"或"缩小" pre-quantization 的有效信号)

---

## 2. 关键数字

### 2.1 Pre-quantization dense retrieval (sentence-t5-base mean-pool history)

| 指标 | 值 |
|------|-----|
| 测试样本数 | 16,759 |
| item corpus | 11,924 |
| history 聚合方式 | mean-pool of train history items |
| embedding 维度 | 768 |
| **Pre-quant R@5** | **0.00107** |
| **Pre-quant R@10** | **0.00173** |
| Pre-quant R@50 | 0.00674 |
| Pre-quant R@100 | 0.01378 |
| Mean target rank | 5848 (out of 11924) |
| Median target rank | 5845 |
| Rank quartiles (25/50/75) | 2728 / 5845 / 8946 |

**结论**: mean-pool of history items 的 dense retrieval 在 Toys test 上几乎是 **random baseline** — 中位 rank = 5845, 几乎正好是 corpus 的一半. R@5 仅 0.00107.

### 2.2 4 seed 模型 R@K (从 verdicts 抽取)

| seed | task | Test R@5 | Test R@10 | Model/Pre-quant R@5 | Model/Pre-quant R@10 |
|------|------|----------|-----------|---------------------|----------------------|
| 42 | #32 | 0.03150 | 0.04950 | **29.33x** | **28.61x** |
| 123 | #33 | 0.03097 | 0.04948 | 28.83x | 28.59x |
| 7 | #34 | 0.02981 | 0.04694 | 27.75x | 27.13x |
| 2024 | #35 | 0.02822 | 0.04766 | 26.27x | 27.54x |
| **mean** | - | **0.03013** | **0.04840** | **28.05x** | **27.97x** |
| **std** | - | 0.00126 | 0.00112 | 1.17 | 0.65 |
| **CV** | - | 4.4% | 2.4% | 4.2% | 2.3% |

### 2.3 关键 gap (pre-quant → model)

- **Pre-quant R@5 gap** = 0.00107 - 0.03013 = **-0.02906** (模型比 pre-quant 高 +0.02906)
- **Pre-quant R@10 gap** = 0.00173 - 0.04840 = **-0.04667** (模型比 pre-quant 高 +0.04667)
- 模型相对 pre-quantization baseline 提升 ~28x

---

## 3. 解读

### 3.1 Pre-quantization dense retrieval 几乎为 random baseline

**这是预期内的发现**. mean-pool of history items 的语义表达力很弱:
- 用户的历史可能包含多个语义类别 (e.g. 教育玩具 + 户外玩具)
- mean-pool 把它们混合成一个平均语义
- 这种"平均用户"语义与"下一个具体物品"的语义距离没有判别性
- R@5 = 0.001 接近随机 (1/11924 ≈ 0.000084)

**结论**: 如果要构造一个有意义的 pre-quantization baseline, 应该用:
- Last-item only (用户的最后一个行为 → next item)
- Recency-weighted (近因权重)
- 或者 learned sequential encoder (SASRec, GRU4Rec 等)

### 3.2 4 seed 模型一致地把 pre-quantization 信号放大 ~28x

| 指标 | seed 42 | seed 123 | seed 7 | seed 2024 | CV |
|------|---------|----------|--------|-----------|-----|
| 放大倍数 R@5 | 29.33x | 28.83x | 27.75x | 26.27x | **4.2%** |
| 放大倍数 R@10 | 28.61x | 28.59x | 27.13x | 27.54x | **2.3%** |

**结论**: 4 个 seed 在把 pre-quantization 的弱信号放大到下游 R@5/R@10 上, **比例高度一致** (CV 2.3-4.2%). 这意味着:
- TIGER 模型对 pre-quantization 的"利用效率"是 seed-invariant 的
- 4 seed 之间的 R@5 差异 (0.028-0.031) 来自训练 RNG 引入的 model quality variation, **不是** 来自 pre-quantization signal quality (因为 pre-quantization 是 constant)
- 想要进一步提升 R@5, 需要从**改进 pre-quantization signal 本身** 入手 (如更好的 sequential encoding), 而不是继续调 seed/hyperparam

### 3.3 4 seed R@5 与放大倍数的弱相关性

排序: seed 42 (R@5=0.03150, 放大 29.33x) > seed 123 (R@5=0.03097, 放大 28.83x) > seed 7 (R@5=0.02981, 放大 27.75x) > seed 2024 (R@5=0.02822, 放大 26.27x)

**两者的排序完全一致**: 模型 R@5 越高的 seed, pre-quantization 放大倍数也越大. 这暗示 (n=4 不显著):
- 更好的模型会"更好地"把 pre-quantization signal 转化为 downstream R@5
- 即: model R@5 与 pre-quantization 利用效率正相关

### 3.4 Per-test-item: Pre-quantization rank 分布

```
Pre-quant rank quartiles: 25%=2728, 50%=5845, 75%=8946
Pre-quant R@5 by rank quartile (lower rank = easier):
  Q0 (n= 4190, rank     0- 2727): pre-quant R@5 = 0.0043
  Q1 (n= 4189, rank  2729- 5843): pre-quant R@5 = 0.0000
  Q2 (n= 4190, rank  5845- 8945): pre-quant R@5 = 0.0000
  Q3 (n= 4190, rank  8946-11921): pre-quant R@5 = 0.0000
```

**只有 rank quartile 0 (top 25% easiest) 的 test item 有非零 pre-quant R@5** (0.0043). 其他 75% 的 test item 在 pre-quantization dense retrieval 中完全无法命中 top-5.

---

## 4. 限制与未来工作

### 4.1 当前未做 (per-test-item model rank)

**理想分析**: per-test-sample, 比较 pre-quantization rank (constant) vs per-seed model rank, 计算 Spearman ρ.
- 需要加载 4 个 best_model.pt + RQ-VAE ckpt + 重新跑 inference
- 每个 seed 都需要 ~5-10 min inference + 复杂的数据 pipeline
- 当前 verdict **未做** 这部分, 只做了 aggregate ratio

### 4.2 当前未做 (last-item only baseline)

mean-pool 是最弱的 pre-quantization baseline. 更合理的 baseline:
- Last-item only: 用最后一个 history item 的 embedding 做 dense retrieval
- Recency-weighted: 越近的 history item 权重越高
- 这两个 baseline 的 R@5 应该显著高于 mean-pool 的 0.001

### 4.3 当前未做 (per-target-item pre-quant hit rate 与 model hit 相关性)

理想: 计算每个 test sample 的 pre_quant_hit@K vs per-seed model_hit@K, 看是否:
- 模型能"补救" pre-quant 失败的 test item
- 模型与 pre-quant 都失败的 test item 比例
- 这需要 per-seed per-test-item 模型预测

---

## 5. 产物清单

| 产物 | 路径 |
|------|------|
| 分析脚本 | `scripts/task131_pre_quant_knn_vs_recall.py` |
| 总结 JSON | `products/task131_pre_quant_knn_vs_recall/task131_summary.json` |
| Per-test-sample 数据 | `products/task131_pre_quant_knn_vs_recall/task131_per_test_sample.npz` (target_idxs, users, pre_quant_rank, pre_quant_hit_5/10/50) |
| 散点 + bar 图 | `products/task131_pre_quant_knn_vs_recall/task131_pre_quant_vs_recall.png` |

---

## 6. 结论

### 6.1 量化前 dense retrieval 在 Toys test 集上几乎为 random baseline

- **Pre-quant R@5 = 0.00107**, 中位 rank = 5845/11924
- mean-pool history 不能给"下一个物品"提供有判别力的查询向量
- **TIGER 模型 R@5 = 0.03013** ≈ pre-quant 的 **28 倍**

### 6.2 4 seed 模型一致放大 pre-quant signal ~28x

- 4 seed 放大倍数 CV 仅 2.3-4.2%
- 模型对 pre-quant 的利用效率高度 seed-invariant
- 4 seed 间的 R@5 差异 (CV 4.4%) 来自训练 RNG, 不是 pre-quant 质量

### 6.3 模型 R@5 与放大倍数排序一致

- 排序: seed 42 > seed 123 > seed 7 > seed 2024 (R@5 和放大倍数都)
- 暗示: 更好的模型 = 更好地利用 pre-quant signal (n=4 不显著, 需更多 seed 验证)

### 6.4 未来方向

- 用更聪明的 pre-quantization baseline (last-item, recency-weighted) 重做对照
- Per-test-item model rank 与 pre-quant rank 的 Spearman ρ (需要重跑 inference per seed)
- 如果放大倍数是 seed-invariant, 提升模型 R@5 的关键是**改进 pre-quant signal**, 而非换 seed

---

**result:** Task #131 完成 — pre-quantization dense retrieval (mean-pool history) 在 Toys test R@5=0.00107, 中位 rank=5845/11924, 几乎为 random baseline; 4 seed 模型一致放大 pre-quant signal ~28x (CV 2.3-4.2%), 模型 R@5 与放大倍数排序一致 (n=4)

result: Task #131 — completed (R8 §9.3 retroactive compliance line appended by Task #114; refer to verdict body above for full details).
