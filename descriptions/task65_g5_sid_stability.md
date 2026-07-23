# Task #65 — G5: SID 稳定性几何 — 边界间隙场与码漂移 (实用性冷门)

> **任务目的**: 验证 G5: 边界间隙场 γ̃_i 预测 SID 翻码 (AUC > 0.7), 即 embedding 微小漂移下 item 的 SID 翻码概率可由"距量化边界多近"的几何度量预测
> **执行日期**: (待启动, 优先级 4/5)
> **状态**: 🟡 待启动

---

## 1. 背景

承接 Task #58-#61 Campaign + G1/G2. 生产环境中 embedding 会周期性重训 (新数据, 新 seed, 新超参). SID 是 embedding 的确定性函数 → embedding 微小漂移可导致 item 的 SID 大规模改变 (码漂移, SID churn) → 部署的生成模型词表语义瞬间失效, 必须全量重训.

**据我所知没有任何 SID 论文量化过这个问题** (P0 必须核查). 这是纯几何问题: item 离量化边界多近, 决定它在扰动下翻码的概率.

**核心数学**:
- 边界间隙场: $\gamma_i^{(l)} = d_{(2)}(r_i^{(l)}) - d_{(1)}(r_i^{(l)})$, 归一化 $\tilde{\gamma}_i^{(l)} = \gamma_i^{(l)} / \|r_i^{(l)}\|$
- 一阶翻码概率: $P(\mathrm{flip}_i^{(l)}) \approx \Phi(-\gamma_i^{(l)} / (2\sigma_{\mathrm{proj}}))$
- **级联翻码放大率**: $\mathrm{Cascade}(l) = P(\mathrm{flip}^{(>l)} \mid \mathrm{flip}^{(l)}) / P(\mathrm{flip}^{(>l)} \mid \neg\mathrm{flip}^{(l)})$

---

## 2. 实验设计

**变量**: 量化方式 (标准 VQ → 大间隔量化器 / 锚定重量化)
**保持不变**:
- Stage 1 embedding (复用 Task #59/#60/#61 多源)
- seed=42, K=256

**启动命令**:
```bash
# D0: 边界间隙分布 + 翻码 AUC (1 天, 纯 GPU KMeans + numpy)
python3 scripts/task65_margin_auc.py \
    --embeddings logs/task59_s1/.../merged_predictions_tensor.pt \
                 logs/task61_s1/merged_predictions_2816d.pt \
    --rq_outputs logs/task59_s2_infer/pickle \
                 logs/task61_s2_infer/pickle \
    --seed_pair [42, 1234] \
    --out_json verdicts/task65_margin_auc.json

# P1: 实测码漂移矩阵 (per layer, per item, 2-3 天)
python3 scripts/task65_churn_matrix.py \
    --embeddings [Task #59/#60/#61 路径] \
    --seeds [42, 1234, 5678] \
    --out_csv verdicts/task65_churn_matrix.csv

# P2: 大间隔量化器原型 (3 天)
python3 scripts/task65_large_margin_kmeans.py \
    --input_pt logs/task59_s1/.../merged_predictions_tensor.pt \
    --output_dir logs/task65_s2/pickle \
    --K 256 --num_hierarchies 3 --beta 0.1 --tau 0.05 --seed 42

# P3: 端到端 TIGER (沿用战线四协议)
CUDA_VISIBLE_DEVICES=1 python3 -m src.train experiment=tiger_train_flat \
    data_dir=data/amazon_data/toys \
    semantic_id_path=logs/task65_s2/pickle/cluster_ids.pt \
    sequence_length=120 num_hierarchies=4 task_name=task65_s3_train
```

---

## 3. 决策触发 (vs 提案 D0 GO 条件)

| D0 AUC | 决策 |
|--------|------|
| γ̃ 对翻码的 AUC > 0.7 | ✅ GO → P1 实测码漂移 + P2 大间隔量化器 + P3 TIGER |
| AUC 在 0.6-0.7 之间 | ⚠️ PARTIAL → 调归一化方式 (per-layer / global) + 改用 γ 而非 γ̃ |
| AUC < 0.6 | ❌ NO-GO → 翻码不由边界间隙主导, 切换调查方向 (e.g., 与 embedding drift 模式相关) |
| P3 R@5 ≥ 0.0977 + 翻码率 < 10% | ✅✅ 大间隔量化器端到端胜利 (双指标) |
| P3 R@5 ≥ 0.0977 但翻码率 ≥ 10% | ⚠️ 仅 Recall 持平, 稳定性无收益 |

---

## 4. 预算

| 阶段 | 估算时间 | GPU |
|------|---------|-----|
| P0 文献核查 (1 周) | 半天 | 无 |
| **D0** 边界间隙分布 + 翻码 AUC | **1 天** | cuda:0 (~10 min 跑多 seed KMeans) |
| P1 实测码漂移矩阵 (3 seed × 5 embedding × 4 layer) | 2 天 | cuda:0 (~30 min total) |
| P2 大间隔量化器原型 + 锚定重量化 | 3 天 | cuda:0 (~1 hour) |
| P3 Stage 3 TIGER | 3 天 | cuda:1 (~120 min) |
| P3 Stage 4 + 评估 | 5 min + 1 min | cuda:1 |
| **总计** | **~10 天** | **~5 GPU-hour** |

---

## 5. 风险与缓解

**风险 1**: 多 seed checkpoint 素材依赖 → Task #53 时期 MCKG 训练记录已存档 (备查), KMeans 本身只需换 seed 重跑 (~10 min/seed)
**风险 2**: 大间隔量化器 hinge 截断 τ 与失真权衡敏感 → P2 阶段扫 β ∈ {0.01, 0.05, 0.1, 0.5} × τ ∈ {0.01, 0.05, 0.1}, 选 Pareto 最优
**风险 3**: 锚定重量化是工程化方案, 需保留旧 SID 作为先验 → 与 P3 TIGER 端到端改动较大, 先做"大间隔量化器"路线
**风险 4**: 翻码 AUC 是经验指标, 可能与下游生成模型零样本退化不直接对应 → P3 同步测"用旧 SID 训练 TIGER + 新 SID 词表"零样本退化曲线

---

## 6. 完成度跟踪

- [ ] P0 文献核查 (检索词 `semantic ID stability retraining`, `codebook drift generative retrieval`, `quantization margin robustness`)
- [ ] D0 写 task65_margin_auc.py + 跑 γ̃ + 翻码 AUC
- [ ] D0 GO/NO-GO 判定 (AUC > 0.7)
- [ ] P1 实测码漂移矩阵 (如 D0 GO)
- [ ] P2 大间隔量化器原型
- [ ] P3 Stage 3 TIGER 训练
- [ ] P3 Stage 4 推断 + Recall/NDCG 评估
- [ ] 写 verdict + 更新 loop.md §16

---

## 7. 重叠审计

- ✅ 与幸存 idea "confusion channel/index assignment" 正交 — 那个关心解码端生成错误的 index 邻近性, G5 关心编码端时间维度码稳定性
- ✅ 与幸存 idea "SID self-fulfilling prophecy/path-dependence" 叙事互补 (都关于 SID 时间动力学), 可同议程并列
- ✅ 与已杀清单零重叠
- ✅ 与四战线零重叠 (战线三架构公平性 ≠ G5 编码几何)