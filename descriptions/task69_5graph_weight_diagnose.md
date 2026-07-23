# Task #69 — 5 种图构造 + MCKG 权重诊断 (Front 4 集中实验)

> **任务目的**: 通过五种从"信息最少"到"信息最多"的图构造 (G0 attribute / G1 interaction / G2 cooccurrence / G3 copurchase / G4 full KG), 训练 5 个独立 MCKG 模型, 记录每组的 κ 值、per-item subspace 权重、权重集中度、权重异质性、quantizability 指标, 验证"几何异质性 vs 图结构信息量"是否单调变化.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

### 1.1 已有数据点 (Front 系列)

| 前置任务 | 结论 |
|---------|------|
| Front 1 (Task #43 / #49) | MCKG 64d norm_CV 高 (病态), margin_ranking_loss 本身是根因 |
| Front 2 (Task #67 v5 vs #68) | 192d raw 比 norm fix + Revival 优 (cov 0.651 vs 0.374) |
| Front 3 (Task #36) | MCKG 64d 训练 collapse (L0 cov 0.11) |
| Task #100 | MCKG 64d raw collapse |

### 1.2 关键空白

当前所有 MCKG 实验都是**单图结构** (G4 = 完整 KG). 但 KGAT/MCKG 论文 (M2GNN Table 7) 显示权重分布 (κ + per-item weights) 与图结构丰富度**强相关**. 本任务第一次在 Toys 上系统性验证 5 种图变体.

### 1.3 核心假设 (R1)

| 假设 | 内容 |
|------|------|
| **R1-A** | 集中度 (mean_max_weight) 随 G0→G4 信息量**单调下降** |
| **R1-B** | G0 (零行为) 也表现出明显异质性 → 内容/属性结构本身就能催生几何异质性 |
| **R1-C** | 所有组的 norm_CV 都很高 → 病根在 margin_ranking_loss, 与图结构无关 (坐实 Front 1 结论) |
| **R1-D** | G1 (纯交互) 的 norm_CV 明显低于 G4 (完整 KG) → KG 关系加重 norm 病态 |

---

## 2. 实验设计

### 2.1 Phase 0 — 5 种图构造 (从信息最少到最多)

| 图 | 边类型 | 信息来源 | 是否依赖特殊字段 |
|----|--------|---------|----------------|
| **G0 attribute_only** | 商品-类别, 商品-品牌 | items 属性 | ✅ 用 toys.items |
| **G1 interaction** | 用户-商品 (评分/购买) | interactions | ✅ 用 toys.interactions |
| **G2 cooccurrence** | 同用户序列窗口内前后商品 | user_sequences | ✅ 用 toys.user_sequences |
| **G3 copurchase** | also_bought, also_viewed | item.related | ⚠️ **需先确认 toys.items[0] 是否有 .related 字段, 没有则降级 4 组** |
| **G4 full_kg** | 现成的 KG 三元组 | load_existing_mckg_kg_graph() | ✅ 复用现有 MCKG KG 加载 |

**降级规则**: 若 `check_related_field_exists(toys_dataset) == False`, G3 自动跳过, 实验变 4 组 (G0/G1/G2/G4).

### 2.2 Phase 1 — 统一训练配置

```python
shared_config = {
    'n_subspaces': 3,
    'aggregator': 'GCN',              # Table 4 验证效果最好
    'embedding_dim': 64,
    'n_hops': 2,
    'loss': 'margin_ranking_geometry_aware',  # 公式 20+21, 完全不动
    'seed': 42,                       # 固定随机种子
}
```

**5 组训练** (按推荐顺序): 先跑 G1 + G4 核心对照 → 验证诊断脚本 → 再补 G0/G2/G3 (并行).

### 2.3 Phase 2 — 权重分布诊断 (5 层记录)

| 层 | 指标 | 来源 |
|----|------|------|
| **第 1 层** | κ1, κ2, κ3 (curvature) | `model.kappa_*` |
| **第 2 层** | 全体商品平均权重 (w1, w2, w3) | `model.get_per_item_subspace_weights()` |
| **第 3 层** | 集中度 (mean_max_weight, std_max_weight) | per_item_w.max(dim=1) |
| **第 4 层** | 异质性 (mean_entropy, std_entropy) | per_item_w 熵 |
| **第 5 层** | quantizability (norm_CV, SCR) | 复用 Front 1 现成脚本 |

### 2.4 Phase 3 — 汇总表格 + 可视化

- `summary_df` 表 (列: 图 / κ1/κ2/κ3 / 集中度 / 熵 / 熵std / norm_CV / SCR)
- 散点图: weight_concentration vs graph_richness (G0→G4)

### 2.5 Phase 4 — 判读

| 观察 | 结论 |
|------|------|
| 集中度 G0→G4 单调下降 | R1-A 成立, 直接写论文 |
| 某一步突变 (如 G2→G3) | 某类特定信息触发异质性 |
| G0 也有明显异质性 | R1-B 成立 (新发现) |
| 全部 norm_CV 都高 | R1-C 成立 (坐实 Front 1) |
| G1 norm_CV 显著低于 G4 | R1-D 成立 (KG 关系加重病态) |

---

## 3. 决策触发 (vs baseline)

| 集中度趋势 | 决策 |
|------------|------|
| **G0→G4 单调下降** | ✅ R1-A 成立, 直接写论文 (Front 4 主要结论) |
| **集中度无明显趋势** | ⚠️ R1-A 否证, 改为报告"图结构对权重分配无显著影响" |
| **某步突变** | ⚠️ 单独报告该图结构的新发现, 深挖是哪类信息 |

| norm_CV 模式 | 决策 |
|--------------|------|
| **5 组都高 (CV > 2.0)** | ✅ R1-C 成立 (Front 1 坐实) |
| **G1 < G4 (差 > 0.5)** | ✅ R1-D 成立 (KG 关系加重病态) |
| **无明显差异** | ⚠️ R1-C/D 部分成立, 但仍支持 margin_ranking_loss 是根因 |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Phase 0: 5 种图构造 | ~30 min |
| Phase 1: MCKG 训练 (5 组, 每组 ~2 小时) | **~10 hours GPU** |
| Phase 2: 5 层诊断脚本 | ~30 min |
| Phase 3: 汇总 + 可视化 | ~15 min |
| Phase 4: 判读 + 写 verdict | ~15 min |
| **总计** | **~11 hours GPU + 1.5 hours CPU** |

> 训练成本最高, 建议 5 组按 2+3 拆分: 先 G1+G4 验证流程, 再并行补 G0/G2/G3.

---

## 5. 风险与缓解

### 风险 1: G3 缺少 `related` 字段

**描述**: Toys 数据集可能没有保留 `related` 字段 (also_bought/also_viewed).
**缓解**: 在 Phase 0 第一步就跑 `check_related_field_exists()`, 不存在则降级 4 组.

### 风险 2: KGAT/MCKG 训练环境未就绪

**描述**: 需要 `kgat_tf216` env (TF 2.16.2 + cu12 + cudnn 8.9.7.29) + 4 GPU.
**缓解**: v6 (PID 3665558) 在 GPU 0 跑, 还剩 GPU 1/2/3. 第一批 G1+G4 可用 GPU 1+2, 第二批 G0/G2/G3 用 GPU 3 (轮流).

### 风险 3: 训练时间超过预算

**描述**: 5 组 MCKG 各 2h 共 10h, 可能被其他任务打断.
**缓解**: 每组都保存 checkpoint (`checkpoints/{name}.pt`), 断点可重训.

### 风险 4: 种子敏感性

**描述**: seed=42 可能不是最优, 5 组共享一个种子保证可比, 但单组内可能不收敛.
**缓解**: 监控 train loss + recall@20, 异常时单独重跑该组 (用 seed=123/7/2024 验证稳定性).

---

## 6. 完成度跟踪

- [ ] Phase 0.1: 确认 toys.items 是否有 `related` 字段 (G3 降级判断)
- [ ] Phase 0.2: 写 `scripts/task69_build_graphs.py` (5 种图构造)
- [ ] Phase 0.3: 验证 5 (或 4) 种图连通性 + 边数合理性
- [ ] Phase 1.1: 写 `scripts/task69_train_mckg.py` (统一 config + 5 组训练)
- [ ] Phase 1.2: 跑 G1 + G4 (核心对照, GPU 1+2)
- [ ] Phase 1.3: 验证诊断脚本对 G1+G4 输出合理, 再并行 G0/G2/G3
- [ ] Phase 2: 写 `scripts/task69_diagnose_weights.py` (5 层记录)
- [ ] Phase 3: 写 `scripts/task69_summarize.py` (汇总表 + 可视化)
- [ ] Phase 4: 写 `verdicts/task69_5graph_weight_result.md` (含 result: 行 + R1-A/B/C/D 判读)
- [ ] 清理 loop.md §16 (从活跃列表删除)
- [ ] 更新 memory `front4_5graph_weight_diagnose.md`

---

**设计依据**: 整合之前分散讨论的"五种图 + 权重诊断 + κ 追踪 + Front 1 复用", 第一次在 Toys 上系统性测试图结构丰富度 vs 几何异质性的关系. 输出将作为论文 Front 4 章节的核心数据.
