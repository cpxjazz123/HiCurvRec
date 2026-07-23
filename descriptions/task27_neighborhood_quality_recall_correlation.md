# Task #27 — 邻域排序质量 vs 下游 Recall 相关性实验

> **任务目的**: 验证 embedding 空间中 item 邻域排序质量（vs 用户真实 co-occurrence 邻域）是否与下游 TIGER Recall 表现正相关. 若相关性显著, 未来 SID 生成可优先聚焦于"高邻域质量"的 items (即 embedding 已学到的语义结构), 跳过低质量 items (数据噪声主导).

> **完成日期**: (in progress)
> **状态**: 🟡 待启动 (2026-07-19 自主决策启动)

---

## 1. 背景

**核心假设**:
- **H1**: 邻域排序质量 (T5 embedding top-k vs co-occurrence top-k 的 Jaccard/ρ) 越高的 item, 在 TIGER Stage 4 预测中越容易命中 → 质量与 Recall 正相关
- **H0**: 邻域排序质量与 Recall 无关 → Recall 完全由用户行为序列模式 + 训练数据量决定, 与 embedding 几何结构无关

**为什么重要**:
- Task #22/Task #23/Task #24 都已否证 PM-RQ × TIGER 端到端优势 (R1 否证)
- Task #87/Task #25 锁定 Toys flat baseline R@5 区间 [0.01489, 0.01937], CV=18.5%
- 剩余待验证假设: **embedding 几何结构本身对 Recall 有多大贡献**
- 若本实验证实 H1, 未来方向应聚焦"提升高邻域质量 item 的召回" (e.g., dual-encoder retrieval)
- 若 H0 成立, 则 embedding 几何优化 ROI 极低, 应转向其他方向

---

## 2. 实验设计

### 2.1 数据准备

**输入 1: Toys 用户序列**
- 路径: `data/amazon_data/toys/`
- 加载所有训练用户的历史序列, 统计 item-item co-occurrence 矩阵

**输入 2: Stage 1 sentence-t5-base embedding (11924 × 768)**
- 路径: `logs/task26_s1/runs/2026-07-19/14-20-10/pickle/merged_predictions_tensor.pt`

**输入 3: Task #87 Stage 4 TIGER predictions (19412 users × top-10 candidates)**
- 路径: `products/task87_tiger_baseline/stage4_tiger_infer/merged_predictions_dedup.pt`

### 2.2 邻域排序质量计算

```python
# 对每个 item i ∈ [0, 11923]:
# 1. Embedding 邻域: top-k_ebd=50 by cosine sim in T5 space
# 2. Co-occurrence 邻域: top-k_cooc=50 by item-item co-occurrence count in user histories
# 3. 邻域质量: Jaccard(ebd_ngh, cooc_ngh) = |intersect| / |union|
#    同时记录 Spearman ρ(rank_ebd, rank_cooc)

# 注: k=50 平衡了"覆盖足够 vs 计算可行"
```

### 2.3 下游 Recall 贡献计算

```python
# 加载 Task #87 Stage 4 predictions (19412 users × 10 candidates × 4 digits)
# 加载 test labels (gt item id → 4 digits)
# 对每个 test interaction:
#   - 检查 gt item 是否在 predicted top-5/10 中
#   - 记录该 gt item 的 hit/miss
# 聚合每个 item 的 hit rate (跨所有 user-item pair 中, item 出现在 predicted top-K 的概率)
```

### 2.4 相关性分析

```python
# 对每个 item i:
#   x = Jaccard quality score (连续)
#   y = hit rate (binary or count)
# 1. 全局 Spearman ρ(x, y)
# 2. 分桶: 将 items 按 quality 分 4 等分 (Q1 lowest → Q4 highest)
#    对比 Q1 vs Q4 的 hit rate
# 3. Bootstrap CI: 1000 次重采样, 95% CI
```

### 2.5 决策阈值

| 实验结果 | 决策 |
|----------|------|
| **Spearman ρ > 0.10, p < 0.01** | ✅ H1 成立, 邻域质量与 Recall 正相关, 建议聚焦高质量 items |
| **ρ ∈ [0.05, 0.10]** | ⚠️ 弱相关, 不显著, embedding 几何贡献有限 |
| **\|ρ\| < 0.05** | ❌ H0 成立, 邻域质量与 Recall 无关, 转向其他方向 |
| **ρ < -0.05** | 🚨 反相关, 邻域质量反而降低 Recall (可能因 overfit to noise) |

---

## 3. 预算

| 阶段 | 估算时间 |
|------|---------|
| 数据加载 + co-occurrence 矩阵 | ~5 min (CPU, 19412 users) |
| Embedding k-NN (11924×768 cosine) | ~3 min (CPU, scipy/sklearn) |
| 质量分桶 + Spearman 计算 | ~1 min (CPU) |
| Recall contribution per item | ~2 min (CPU) |
| Bootstrap 1000 次 | ~5 min (CPU) |
| **总计** | **~15 min** (CPU only, 不抢 GPU) |

---

## 4. 风险与缓解

**风险 1**: Co-occurrence 矩阵稀疏, 多数 item 对共现 0 次 → k=50 邻域会有大量 tie
- 缓解: 对低共现 items 用 Jaccard 退化为 0, 在相关性分析中单独处理

**风险 2**: Stage 4 predictions 已含 2× dedup, 需要找到正确的 ground truth label 路径
- 缓解: 用 Task #23 Stage 4 eval 脚本相同的 ground truth 加载逻辑

**风险 3**: items 在 Stage 4 中只出现 N_test 次, 样本不均导致 hit rate 噪声大
- 缓解: 只统计 ≥ 5 次出现的 items (约 80% of items), 排除长尾

**风险 4**: Embedding dim=768, 距离计算 O(N²) = 1.4×10⁸, scikit-learn pairwise 可能慢
- 缓解: 用 torch.cdist on CPU + 矩阵分块, 或用 FAISS CPU index

---

## 5. 完成度跟踪

- [x] Task #27 description 写盘
- [ ] 脚本: `scripts/task27_neighborhood_quality.py`
- [ ] Co-occurrence 矩阵生成
- [ ] Embedding k-NN 计算
- [ ] 质量分桶 + Recall contribution
- [ ] Spearman ρ + Bootstrap CI
- [ ] 写 `verdicts/task27_neighborhood_quality_result.md`
- [ ] 更新 loop.md §16

---

## 6. 关联任务

- 基准: Task #87 (TIGER baseline, R@5=0.01937, seed=42)
- 输入 embedding: Task #26 Stage 1 sentence-t5-base (logs/task26_s1/runs/2026-07-19/14-20-10/pickle/merged_predictions_tensor.pt)
- 假设: H1 邻域质量正相关 vs H0 无关
- 上游: Task #22/#23/#24 已否证 PM-RQ, 本任务验证 embedding 几何本身对 Recall 的贡献
