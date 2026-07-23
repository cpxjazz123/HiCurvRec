# Task #73 — MCKG 知识图谱构建（基于 RippleNet 数据 + KGAT 代码）

> **任务目的**: 复现 MCKG 论文 LastFM 数据集的 KG 数据构建链路，最终跑通一次 KGAT baseline 训练（HR@10≈0.571, HR@20≈0.614），为后续引入多几何空间或迁移到 Amazon Toys 数据集建立可靠基础。

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

GeneRec 项目当前方向是 GRID（基于语义 ID 的生成式推荐），已有 Task #65（baseline R@10=0.09710）/ Task #68 / Task #71 等流水线索。**本次扩展新方向**：MCKG（Multi-Curve Knowledge Graph），涉及知识图谱增强推荐。

**数据血统**: RippleNet 仓库（https://github.com/hwwang55/RippleNet），MCKG 论文实际使用的 LastFM / MovieLens-1M / Book-Crossing 数据来源。
**代码框架**: KGAT 仓库（https://github.com/xiangwang1223/knowledge_graph_attention_network），借用其图构建 / GCN 训练代码结构。
**目标产出**: 统一格式 + 统计验证的 KG 数据 + 一次跑通的 baseline 训练结果。

**承接关系**: 与现有 GRID 任务**无直接依赖**——是新的研究线索。Task #72（codebook_width ablation）已取消，本任务是 Task #72 取消后的替代活跃任务。

---

## 2. 实验设计

**本任务是多阶段数据工程 + 一次 baseline 训练**，按用户指定的 5 个子任务顺序执行：

### 子任务 1：数据获取与格式勘察

**输入**: RippleNet 仓库 `data/music/`（LastFM）+ KGAT 仓库 `Data/last-fm/`（对比参考）
**动作**:
- 1.1 定位 RippleNet LastFM 数据，确认 `user_artists.dat` / `kg.txt` / `item_index2entity_id.txt` 存在
- 1.2 定位 KGAT Last-FM 数据，查看 `kg_final.txt` / `entity_list.txt` / `relation_list.txt` / `user_list.txt` / `item_list.txt` / `train.txt` / `test.txt` 格式
- 1.3 输出"格式对照表"（每列含义、ID 起始、是否需 remap）

**核验**: 无（仅格式核对）

### 子任务 2：数据统计核验 ★ 关键检查点

**核验指标**（来自 MCKG 论文 Table 2 LastFM 列）：

| 指标 | 目标值 | 容差 |
|------|-------:|------|
| 用户数（Users）| 1,872 | <1% / 1-15% 注明 / >15% 停止 |
| 物品数（Items）| 3,846 | <1% / 1-15% 注明 / >15% 停止 |
| 交互数（Interactions）| 42,346 | <1% / 1-15% 注明 / >15% 停止 |
| KG 实体数（Entities）| 9,366 | <1% / 1-15% 注明 / >15% 停止 |
| KG 关系数（Relations）| 60 | <1% / 1-15% 注明 / >15% 停止 |
| KG 三元组数（Triples）| 15,518 | <1% / 1-15% 注明 / >15% 停止 |

**判定**:
- ✅ 误差 < 1% → 数据源正确，进入子任务 3
- ⚠️ 误差 1-15% → 记录差异，继续（最终报告注明）
- ❌ 误差 > 15% → 停止，回子任务 1 重核（可能拿错文件）

### 子任务 3：格式转换脚本（RippleNet 格式 → KGAT 格式）

**输入**: RippleNet `kg.txt` / `item_index2entity_id.txt` / `user_artists.dat`
**输出**（KGAT 期望格式）:
- `entity_list.txt`（统一编号所有 KG 实体）
- `relation_list.txt`（统一编号所有关系类型）
- `item_list.txt`（org_id / remap_id / freebase_id 三列）
- `user_list.txt`（用户 ID 对照）
- `kg_final.txt`（h, r, t，使用重新编号后 ID）
- `train.txt` / `test.txt`（按 MCKG 4.1.3 节或 KGAT README 切分规则）

**要点**:
- ID 必须重新映射（不能假设两边编号体系一致）
- 明确 train/test 切分口径（与验收指标口径一致）

**核验**: 转换前后统计量**必须完全一致**（0 容差）

**判定**:
- ✅ 完全相等 → 转换无数据丢失/重复
- ❌ 任意一项对不上 → debug（常见：ID remap 漏实体、去重逻辑有误）

### 子任务 4：接入 KGAT 代码验证数据加载

**动作**:
- 4.1 定位 KGAT 代码中读取数据、构建邻接矩阵的模块（`load_data.py` 或 `utility/loader_kgat.py`）
- 4.2 将子任务 3 数据放入 KGAT 期望目录，尝试跑通"数据加载"（先不跑完整训练）

**核验日志变量**：

| 日志变量 | 目标值 | 已知口径差异 |
|----------|-------:|--------------|
| n_users | ≈1,872 | — |
| n_items | ≈3,846 | — |
| n_entities | ≈9,366 | 可能含 items 本身 → 可能 +3,846 |
| n_relations | ≈60 | KGAT 可能自动加逆关系 → 可能 ×2 = 120 |
| n_train + n_test | ≈42,346 | 即交互总数 |

**判定**:
- ✅ 排除已知口径差异后能对应上 → 进入子任务 5
- ❌ 对不上且无法用口径差异解释 → 回头检查子任务 3 / 4

### 子任务 5：跑通 baseline 训练 ★ 最终验收

**输入**: 子任务 4 验证过的数据
**动作**: 跑一次完整 KGAT 训练（标准欧氏版本），超参参考 MCKG 论文 4.1.4 节：LastFM sampling size=4, hop=3

**核验指标**（来自 MCKG 论文 Table 3 LastFM 列）：

| 模型 | HR@10 | HR@20 | NDCG@10 | NDCG@20 |
|------|------:|------:|--------:|--------:|
| HGCF | 0.544 | 0.596 | 0.350 | 0.361 |
| RippleNet | 0.562 | 0.611 | 0.361 | 0.372 |
| **KGAT（对照目标）**| **0.571** | **0.614** | **0.364** | **0.377** |
| KBHP | 0.612 | 0.645 | 0.387 | 0.406 |
| LKGR | 0.628 | 0.678 | 0.396 | 0.418 |
| MCKG | 0.635 | 0.691 | 0.405 | 0.432 |

**判定**:
- ✅ HR@10 与 KGAT 目标 0.571 绝对差 ≤ ±0.05 → 数据构建链路可信
- ⚠️ 差距较大（如 HR@20 < 0.4 或 > 0.7）→ 检查超参 / train-test 切分 / 数据泄漏
- ❌ 训练不收敛 / HR 接近随机水平 → 子任务 3-4 数据格式有严重问题，回退 debug

---

## 3. 决策触发（vs MCKG 论文 KGAT baseline）

| 指标条件 | 结果指标 | 决策 |
|----------|----------|------|
| 子任务 2 误差 < 1% | 全部 6 项 | ✅ 数据源正确，继续 |
| 子任务 2 误差 1-15% | — | ⚠️ 记录差异，继续 |
| 子任务 2 误差 > 15% | — | ❌ 停止，回子任务 1 |
| 子任务 3 转换前后一致 | 0 容差 | ✅ 进入子任务 4 |
| 子任务 3 转换后不一致 | — | ❌ debug 转换脚本 |
| 子任务 5 HR@10 ∈ [0.521, 0.621] | ±0.05 容差 | ✅ baseline 复现可信 |
| 子任务 5 HR@10 < 0.521 或 > 0.621 | — | ⚠️ 检查超参 + 切分 |
| 子任务 5 HR 接近随机 | ≈ K/n_items | ❌ 子任务 3-4 严重问题，回退 |

---

## 4. 预算

| 子任务 | 估算时间 |
|--------|---------|
| 子任务 1：数据勘察 | ~0.5 天 |
| 子任务 2：统计核验 | ~0.5 天 |
| 子任务 3：格式转换脚本 | ~1-2 天 |
| 子任务 4：KGAT 代码接入 | ~0.5-1 天 |
| 子任务 5：baseline 训练 | ~0.5-1 天 |
| **总计** | **~3-5 天** |

> 注：本任务主要是数据工程 + 短训练，无需长 GPU 时间。KGAT 训练本身 ~1-2 小时。

---

## 5. 风险与缓解

**风险 1**: RippleNet / KGAT 仓库结构与预期不符（路径 / 文件名变更）
→ 缓解: 子任务 1 先勘察再写脚本，不假设路径硬编码

**风险 2**: MCKG 论文 Table 2 数字与实际 LastFM 数据有差异（core-filtering 阈值不同）
→ 缓解: 子任务 2 严格执行三级判定（<1% / 1-15% / >15%），差异大时立即停止

**风险 3**: KGAT 代码自动加逆关系 / 含 item 作为 entity，导致 n_entities / n_relations 翻倍
→ 缓解: 子任务 4 表格已注明已知口径差异，对照时手动扣除

**风险 4**: train/test 切分口径与 MCKG 论文不一致 → HR 指标无法对齐
→ 缓解: 子任务 3 明确记录所采用的切分规则（按 MCKG 4.1.3 或 KGAT README），子任务 5 验收时如有差异需追溯

**风险 5**: 与现有 GRID 流水线方向冲突（GPU 资源 / 工程时间）
→ 缓解: 本任务主要是数据工程 + 短训练，不抢占 GRID 流水线 GPU 时间

---

## 6. 完成度跟踪

- [ ] 子任务 1：数据获取与格式勘察
- [ ] 子任务 2：数据统计核验（vs MCKG Table 2）
- [ ] 子任务 3：格式转换脚本（RippleNet → KGAT）
- [ ] 子任务 4：接入 KGAT 代码验证数据加载
- [ ] 子任务 5：baseline 训练（HR@10≈0.571, HR@20≈0.614）
- [ ] 写 verdict → `verdicts/task73_result.md`
- [ ] 更新 §16 历史归档

---

**关键产出物**:
1. 一份格式统一、统计验证的 LastFM KG 数据（可直接 KGAT 训练）
2. 可复用转换脚本（RippleNet 格式 → KGAT 格式，适配 MovieLens-1M / Book-Crossing / Amazon Toys）
3. 一次跑通的 baseline 训练结果（HR@10≈0.571）