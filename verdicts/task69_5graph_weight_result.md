# Task #69 — 5-graph MCKG 权重诊断 (Front 4 集中实验)

> **任务目的**: 通过五种从信息最少到最多的图构造 (G0→G4), 训练 5 个 MCKG 模型, 验证 "几何异质性 vs 图结构信息量" 的关系.
> **完成日期**: (in progress)
> **状态**: 🟡 训练完成后判读

---

## 1. 背景

KGAT/MCKG 论文 (M2GNN Table 7) 显示 κ + per-item subspace weights 与图结构丰富度强相关. Toys 上从未系统性验证.
本任务填补这一空白, 验证 R1-A/B/C/D 四个假设.

## 2. 实验设计

### 2.1 五种图构造

| 图 | 信息 | kg_final edges |
|----|------|----------------|
| G0 (attribute) | 73,152 (4 relations, no co_purchase) |
| G1 (interaction) | 128,773 (user-item) |
| G2 (cooccur) | 587,680 (item-item, window=5) |
| G3 (copurchase) | 3,084 (relation 3 only) |
| G4 (full KG) | 76,236 (5 relations) |

### 2.2 统一训练配置

- M=3, dim=64 (per subspace), n_hops=2, n_neighbors=8
- init_kappas=[1.0, 0.0, -1.0], kappa_clamp=2.0
- num_epochs=50, batch_size=1024, lr=1e-3, c=1.0
- eval_every=5, patience=4 (paper-strict 20 epochs early stop)
- seed=42

### 2.3 5 层权重诊断

| 层 | 指标 |
|----|------|
| 1 | κ1, κ2, κ3 (curvature) |
| 2 | 全体 item 平均权重 (w1, w2, w3) — proxy via embedding norm softmax |
| 3 | 集中度 (mean_max_weight, std_max_weight) |
| 4 | 异质性 (mean_entropy, std_entropy) |
| 5 | quantizability (norm_CV, SCR) — 复用 Front 1 |

## 3. 关键指标 (5 组完整对照)

| 图 | κ1 | κ2 | κ3 | mean_max_w | mean_ent | std_ent | norm_CV | SCR | test HR@10 | test HR@20 |
|----|----|----|----|------------|----------|---------|---------|-----|------------|------------|
| G0 (attribute) | 0.646 | -0.088 | -0.991 | 0.5150 | 0.9777 | 0.1846 | 3.666 | 3.666 | 0.25783020811868945 | 0.4078405110241088 |
| G1 (interaction) | 0.121 | -0.064 | -1.059 | 0.5262 | 0.9495 | 0.1912 | 2.466 | 2.466 | 0.4661034411704101 | 0.5779929940243148 |
| G2 (cooccur) | -0.054 | -0.098 | -0.667 | 0.4899 | 1.0081 | 0.1355 | 0.908 | 0.908 | 0.45574902122398514 | 0.5687203791469194 |
| G3 (copurchase) | 1.166 | -0.085 | -1.007 | 0.8968 | 0.2515 | 0.3703 | 7.221 | 7.221 | 0.24680609932000824 | 0.3653925406964764 |
| G4 (full KG) | 0.713 | -0.109 | -0.984 | 0.5063 | 0.9850 | 0.1850 | 10.651 | 10.651 | 0.24696064290129816 | 0.3965073150628477 |

## 4. R1-A/B/C/D 假设判读

### R1-A: ⚠️ fail_or_partial

**证据**: 非单调: G0_attribute_only=0.5150, G1_interaction_only=0.5262, G2_cooccurrence=0.4899, G3_copurchase=0.8968, G4_full_kg=0.5063

### R1-B: ✅ pass

**证据**: G0 集中度=0.5150, 熵=0.9777 (低集中度或高熵)

### R1-C: ⚠️ partial

**证据**: 4/5 组 norm_CV > 2.0

### R1-D: ✅ pass

**证据**: G1 norm_CV=2.466, G4 norm_CV=10.651, 差=8.185 (> 0.5)

## 5. 解读

### 5.1 Test HR@20 排名 (核心)

| 排名 | 图 | HR@20 | norm_CV | 集中度 |
|------|----|-------|---------|--------|
| 1 | **G1 (interaction)** | **0.5780** | 2.466 | 0.5262 |
| 2 | G2 (cooccurrence) | 0.5687 | 0.908 | 0.4899 |
| 3 | G0 (attribute) | 0.4078 | 3.666 | 0.5150 |
| 4 | G4 (full KG) | 0.3965 | 10.651 | 0.5063 |
| 5 | G3 (copurchase) | 0.3654 | 7.221 | 0.8968 |

**核心发现 (反直觉)**:
- **G1 (纯用户-商品交互) > G4 (完整 KG)**: HR@20 0.5780 vs 0.3965 (+46%)
- **更多图信息不总是更好**: G4 是 5 种图里 norm_CV 最高的 (10.65), 说明多关系 KG 信息放大了 norm 病态
- **G2 (cooccurrence) 反常低 norm_CV (0.908)**: item-item 共现图让 norm 分布最健康, 接近随机均匀

### 5.2 R1-A: 集中度非单调, G3 是异常值

```
G0=0.515  G1=0.526  G2=0.490  G3=0.897  G4=0.506
```

- G0/G1/G2/G4 都在 0.49-0.53 范围 (低集中度, 几何混合使用)
- **G3 (copurchase only) 极端高集中度 0.897**: 3084 条 co_purchase 边强制所有 item 集中在同一子空间
- 推测: co_purchase 关系过于稀疏, 训练时 GCN 聚合把所有 item 推到同一个 manifold

### 5.3 R1-B: G0 零行为也有几何异质性 ✅

- G0 (attribute only) 集中度 0.515, 熵 0.978
- 与 G1/G2/G4 (0.49-0.53) 几乎一样, 远低于 G3
- 含义: **属性/类别结构本身就能催生几何异质性**, 不依赖协同过滤信号
- 这是论文 Front 4 章节可强调的新发现

### 5.4 R1-C: 4/5 高 norm_CV, G2 异常低

```
G0=3.67  G1=2.47  G2=0.91  G3=7.22  G4=10.65
```

- G2 (cooccurrence) 的 norm_CV=0.91 < 2.0 阈值, **不符合 R1-C 假设**
- 含义: cooccurrence 边天然平衡 (无向 + 重复 item 对), 让 norm 分布最接近均匀
- Front 1 结论 (margin ranking loss 致 norm 病态) 仍部分成立: 4/5 组高 CV

### 5.5 R1-D: G1 << G4 (norm_CV) ✅ 强支持

- G1 norm_CV=2.47, G4 norm_CV=10.65, 差=8.19
- 含义: **KG 关系 (尤其是 brand/category/same_sub_cat) 显著加重 norm 病态**
- 推测机制: 多个 item-brand 边把所有 item 推向 brand embedding 方向, 拉大 norm 差异

### 5.6 对下游 RQ-VAE 训练的启示

- **G1 (interaction) embedding 是最佳 MCKG 训练输入**: HR@20 0.578, norm_CV 2.47
- **避免 G4 (full KG)**: norm_CV 10.65 会直接放大 Stage 2 RQ-VAE 的 L0 coverage 问题 (参 Task #67)
- **G3 (copurchase only) 是最差选择**: 高集中度 0.90 + 高 norm_CV 7.22, 信息量小但病态重
- **G2 (cooccurrence) 最有意思**: norm_CV 0.91 < 1.0, 接近完美均匀 norm, 但 HR@20 0.569 仅次 G1

### 5.7 与论文 M2GNN Table 7 的对比

- M2GNN 论文在 book/lastfm 上报告: 集中度随图信息量增加而下降
- **Toys 上未观察到该模式**: G0/G1/G2/G4 集中度几乎相同 (~0.5), G3 异常
- 可能原因: Toys 数据集规模小 (11924 items vs book 30K+), GCN 聚合未能形成明显子空间分化

### 5.8 训练稳定性观察

| 图 | 训练稳定性 | 备注 |
|----|----------|------|
| G0 | ✅ 稳定 (loss 0.05) | HR@20 0.41 |
| G1 | ✅ 稳定 (loss 0.27) | HR@20 **0.58** 最佳 |
| G2 | ⚠️ 后期 loss 爆炸 (0.27→8.6) | best_state epoch 20 抢救, HR@20 0.57 |
| G3 | ⚠️ 数值尖峰 (loss=9927) | 1 次爆炸, 1 次恢复, HR@20 0.37 |
| G4 | ✅ 稳定 (loss 0.04) | HR@20 0.40 |

- 早期 stop 实际未触发 (mckg.py 嵌套 bug: `if no_improve >= patience` 在 `if scheduler is not None` 内)
- 所有 5 组都跑满 50 epochs, 但 best_state 在更早 epoch (基于 val HR@20)

## 6. 产物清单

## 6. 产物清单

- `products/task69_5graph_weight/{group}/entity_embedding.pt` — 5 组 checkpoint
- `products/task69_5graph_weight/diagnose_results.json` — 5 层诊断结果
- `products/task69_5graph_weight/task69_summary.csv` — 汇总表
- `products/task69_5graph_weight/task69_weight_vs_richness.png` — 可视化
- `logs/task69_5graph_*.log` — 训练日志

## 7. 完成度跟踪

- [x] Phase 0: 5 种图构造 (73152 / 128773 / 587680 / 3084 / 76236 edges)
- [x] Phase 1: G1 + G4 训练完成
- [x] Phase 1.5: G0 + G2 + G3 训练完成
- [x] Phase 2: 5 层权重诊断脚本
- [x] Phase 3: 汇总 + 可视化
- [x] Phase 4: R1-A/B/C/D 判读

---

result: Task #69 完成 5 组训练 + 5 层诊断. R1-A=fail_or_partial, R1-B=pass, R1-C=partial, R1-D=pass. (详见上文表格 + 假设判读).