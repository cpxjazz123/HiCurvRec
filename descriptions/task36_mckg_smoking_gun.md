# Task #36 — MCKG embedding 重复 Task #132 完整诊断

> **任务目的**: 把 Task #132 的 7 种 history 聚合 + R@5/R@10 + smoking gun cosine gap 检验搬移到 MCKG embedding 空间 (3 个子空间 + fused),与 T5 空间 0.0002 gap 对比,判断 MCKG 是否真携"下一步买什么"的强信号.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动 (CPU-only 分析, ~15 min)

---

## 1. 背景

承接 Task #120 / #123 (已确认 MCKG 空间几何结构存在 + 3 子空间 (κ=[+5.05, -0.08, -5.04]) 在 Toys 数据有效).

承接 Task #131 / #132 (T5 空间 dense retrieval pre-quant R@5=0.001, gap=0.0002 ≈ 随机).

**关键问题**: MCKG Toys 训练目标是 KG-based CF,user history 是全部交互物品;而 phonism/TIGER Toys 有"next-item"标注. 两者语义不同. **本任务用 leave-one-out 协议近似 next-item**(train_history = test_history[:-1], target = test_history[-1]),保持与 Task #132 的可比性.

## 2. 实验设计

**输入**:
- MCKG embedding: `products/task99_mckg_rebuild/entity_embedding.pt`
  - `subspace_item: (3, 11924, 64)` — κ=[+5.05, -0.08, -5.04]
  - `subspace_user: (3, 19412, 64)` — 3 子空间用户表示
  - `fused_item: (11924, 64)` — tangent 拼接投影后
  - `fused_user: (19412, 64)`
- MCKG Toys data: `MCKG_data/toys/{test,train}.txt`,格式 `[user_id] [item_id_1] [item_id_2] ...`

**7 种 history 聚合 → query 向量**:
1. `agg_last_item`: 直接用 user_history[-1] 的 item embedding (如果用户有多次访问)
2. `agg_last_k_mean`: 最近 k=5 个 history item 的 mean
3. `agg_mean_pool`: 全部 train_history items 的 mean
4. `agg_recency_weighted`: weight_i = exp(-i/K), 越近越高权
5. `agg_max_pool`: 逐维 max-pool of history items
6. `agg_random_query`: 随机 query (对照组)
7. `agg_user_emb`: 直接用 `fused_user[u]` 作为 query (MCKG 本身训练好的 user 表

**评估指标**:
- R@5, R@10 (per-method)
- **Smoking gun**: cosine(query, target) - cosine(query, random_item) 在 test 集上的 mean gap, per-subspace + fused

**Test 构造** (leave-one-out):
- 每个 user: query 由 train_history (train.txt) 聚合;target = test.txt 比 train.txt 多出的最后 1 个 item (如果 train_history 是 test_history 的前缀则成立)

**启动命令**:
```bash
python3 scripts/task36_mckg_smoking_gun.py
```

## 3. 决策触发 (vs T5 baseline)

| MCKG gap 量级 | 解读 | PM-RQ motivation 状态 |
|--------------|------|---------------------|
| gap ≥ 0.02 | 强信号, MCKG 真携"下一步买什么"信息 | ✅ 正面验证, PM-RQ 几何保护有价值 |
| 0.001 < gap < 0.02 | 中等信号, 几何结构存在但相关性弱 | ⚠️ 需重新审视 PM-RQ 的端到端增益 |
| gap ≤ 0.001 (≈ T5 0.0002) | 与 T5 等量级, dense retrieval 不work in any space | ❌ PM-RQ motivation 需重新考虑 (换信息源) |

## 4. 预算

| 阶段 | 估算 |
|------|------|
| 数据加载 + test 构造 | ~1 min |
| 7 aggregations × 5 spaces (3 sub + fused + control) | ~5 min |
| smoking gun gap 计算 | ~5 min |
| verdict 撰写 | ~5 min |
| 总计 | **~15-20 min CPU** |

## 5. 风险与缓解

**风险 1**: MCKG Toys 的 test/train split 不是 next-item 标注,leave-one-out 假设可能不严格 → 缓解:同时报告 (a) leave-one-out, (b) 用 KG-based "candidate" 定义 target, 做交叉验证

**风险 2**: 子空间是流形 (κ≠0), cosine similarity 在流形上不直接适用 → 缓解:对 fused (欧氏) 单独算 cosine gap;子空间单独用 task120 的加权 Riemannian 距离

**风险 3**: items 不一定在所有 user history 中都被预训练 → 缓解:过滤出现在 fused_item embedding 范围内的 user history items

## 6. 完成度跟踪

- [ ] test 集构造脚本 (leave-one-out from MCKG test/train)
- [ ] 7 aggregations 实现
- [ ] R@5/R@10 per method per space
- [ ] smoking gun gap per-method per-space
- [ ] verdict 撰写 + 与 T5 0.0002 对比
