# Task 56/57/58/68 — Brand 因果链族 (合并文件)

> **合并来源**:
> - `descriptions/task56_phase3_causal_counterfactual.md` (Task #437)
> - `descriptions/task57_phase4_extend_to_hh.md` (Task #438)
> - `descriptions/task58_mixed_curvature_causal_chain.md` (Task #450)
> - `descriptions/task68_l1_brand_side_feature.md` (Task #68)
>
> **合并日期**: 2026-07-18
>
> **合并原因**: 四个任务共属 "brand side feature / mixed curvature 因果链" 主题, 互相依赖 (Phase 2 → Phase 3 反事实 → Phase 4 多层 H → L1 brand side), 合并为单一"brand 因果链"族, 保留 task56 v1.1 反事实设计 (P0.3 H 子空间实际位于 text 前 32-d, 不是 brand) 为最新结论。
>
> **编号映射**:
> | 合并前 task id | 任务标题 | 本文件中章节 |
> |----------------|----------|--------------|
> | Task #56 (旧 Task #437) | Phase 3 反事实验证 | §4 |
> | Task #57 (旧 Task #438) | Phase 4 H_H_E_E / H_H_H_H | §5 |
> | Task #58 (旧 Task #450) | Phase 2 Mixed-Curvature 因果链 | §3 |
> | Task #68 | L1 品牌 side feature | §6 |

---

## §1 总目标

本族任务围绕一个核心问题: **brand 因子在 RQ-VAE SID 编码中到底起什么作用?**

四个原任务构成一个递进的因果链:

1. **Task #58 (Phase 2)**: 证明 Joint Mixed-Curvature RQ-VAE 提升 Recall 的因果机制 (不是单纯用复杂距离, 而是通过混合曲率让码字真由多因素共同决定)
2. **Task #56 (Phase 3)**: 通过反事实训练, 验证 P0.3 提升的关键机制是 H 子空间 (H 子空间实际在 text 前 32-d, task458 链 3 "brand MFSR +0.170" 是间接效应)
3. **Task #57 (Phase 4)**: 把单层 H 扩展到多层 H (H_H_E_E, H_H_H_H), 验证 brand-cluster 机制在多层 H 下是否更激进
4. **Task #68**: 不动 RQ-VAE, 改 TIGER 端: 把 L1 codeword 解码出的 brand 信息作为 side feature 拼接, 验证 L1 品牌信息能否被利用提升 R@10

**整族任务回答 4 个层层递进的问题**:
- §3: Mixed-Curvature 提升的**因果链是否成立**? (多因素共同决定码字)
- §4: H 子空间是**必要条件**吗? (反事实 A/B/C/D)
- §5: 多层 H 是否**更激进** brand-cluster? (H_H_E_E / H_H_H_H)
- §6: L1 brand side feature 能否**直接利用** brand 信息? (TIGER 端)

---

## §2 因果链时间线

```
Task #58 (Phase 2 Mixed-Curvature 因果链)
  ↓ 证明 MFSR / Dominance / Pareto 保留 / 有效码本容量 / decoder CE / 行为前缀对齐
  ↓ 结论: Mixed-Curvature 优于 Euclidean-Concat, 因果链成立
  ↓
Task #56 (Phase 3 反事实)
  ↓ H_SUBSPACE_OFFSET 扫描: text[0:32] / brand[768:800] / taxonomy[800:896] / behavior[896:928]
  ↓ 结论: P0.3 H 在 text 前 32-d 是关键机制 (brand MFSR +0.170 是间接效应)
  ↓
Task #57 (Phase 4 多层 H)
  ↓ per-layer distance: H_H_E_E / H_H_H_H
  ↓ 结论: 多层 H 是否更激进 brand-cluster? 风险: 过聚类导致 R@10 暴跌
  ↓
Task #68 (L1 brand side feature)
  ↓ 不动 RQ-VAE, 改 TIGER 端 concat L1 brand embedding
  ↓ 结论: L1 品牌信息能否被利用, 提升 R@10 (baseline 0.09710)
```

**依赖关系**:
- Task #56 依赖 Task #58 的 Euclidean-Concat baseline (P0) + P0.3 训练产物
- Task #57 依赖 Task #56 确认 P0.3 H 位置后, 扩展到 per-layer
- Task #68 与 #56/#57 解耦: 复用 Task #65 RQ-VAE ckpt, 只改 Stage 3 TIGER

---

## §3 Task #58 — Phase 2 Mixed-Curvature RQ-VAE 因果链证明

> **目的**: 证明 Joint Mixed-Curvature RQ-VAE 提升 **不是单纯 Recall 变高**, 而是通过完整的因果链
> **数据集**: 仅 Amazon Toys
> **目标仓库**: `/home/wlia0047/ar57/wenyu/GeneRec/GRID`
> **环境**: `conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys`

### §3.1 因果链 (必须证明的完整链路)

```
不同因素具有不同几何结构
   ↓
联合曲率距离减少单一因素主导
   ↓
一个码字真正由多个因素共同决定
   ↓
SID 前缀更符合用户行为
   ↓
解码器更容易预测
   ↓
Recall / NDCG 提升
```

→ 任一环失败 → 方法**不成立**, 不能声称 "use a more complex distance"。

### §3.2 公平对比方法 (4 个 baseline)

| 方法 | 含义 | 关键差异点 |
|------|------|-----------|
| **Euclidean RQ-VAE** | 原始欧氏 RQ-VAE | baseline |
| **Euclidean-Concat RQ-VAE** ⭐ | 输入同样包含类别、属性、行为, 但**全部在欧氏空间计算** | **最关键 baseline**: 证明提升不是来自"加入更多特征" |
| **Single-Curvature RQ-VAE** | 全部使用双曲或其他单一曲率 | 对比 "混合" vs "单一" 曲率 |
| **Joint Mixed-Curvature RQ-VAE** | 本文方法, 一个码字包含多个曲率分量 | 实验目标 |

**Euclidean-Concat 必须用完全相同的信息**, 仅替换距离。

**公平性约束**: L / K / D_total / SID 长度 / decoder / 训练轮数 / seed 全部相同。

### §3.3 8 个诊断实验

| 实验 | 关键指标 | 期望 |
|------|----------|------|
| **1. 多因素贡献分解** | MFSR / Dominance | Mixed 高 MFSR, 分散 Dominance; Euclidean-Concat 低 MFSR, 单因素主导 |
| **2. 反事实商品对** | 总距离变化 / 选中码字 margin / SID 哪层变 | 小扰动稳定 / 大变化 SID 跳变 |
| **3. 尺度放大** | Assignment NMI / ARI / 功率因子 α∈{0.25, 0.5, 1, 2, 4} | Mixed 基本保持; Euclidean-Concat 大量换码字 |
| **4. 因素结构保留** | Spearman / Mantel / kNN hit@k (层级/连续/行为) | Mixed Pareto 优, 不牺牲一个因素换另一个 |
| **5. 码本崩塌 (DRQ-style)** | Active Codes / Perplexity / Gini / 死码字比例 / 有效码本容量 | Recall 提高, 有效码本容量不下降 |
| **6. 每层贡献** | 截断测试 (L1 / L1+L2 / L1+L2+L3) / 逐层打乱 / 条件信息量 | 每层都有新增价值, 而非 L1 已包含绝大多数因素 |
| **7. 解码器容易度** | Token CE / Beam Search 效率 / 错误传播 | 相同 Recall 下 Mixed 用更小 beam, 或相同 beam 下 Mixed Recall 更高 |
| **8. SID 行为前缀对齐** | Behavioral Prefix Hit@K | Mixed SID 前缀更符合用户行为 |

### §3.4 5 个核心消融实验

| 消融 | 配置 | 目的 |
|------|------|------|
| **A1: 去一种曲率** | H+E+B / H+E / H+B / E+B | 哪个分量最重要 |
| **A2: 错误曲率分配** | 层级→欧氏, 连续→双曲 | 是否需要"正确"曲率-因素匹配 |
| **A3: 固定 vs 可学习权重** | $w_H = w_E = w_B$ vs auto-learn | 权重是否需要自动学 |
| **A4: 欧氏拼接** | 同输入, 普通欧氏距离 | **最关键**: 提升是否仅来自距离复杂度 |
| **A5: 因素打乱** | 打乱行为或类别, 重训练 | 模型是否真的用到了该因素 |

### §3.5 最小实验组合 (Phase 2 启动必做 5 组)

1. **多因素贡献分解 (实验 1)**: 证明最终码字至少 2 因素共同支持
2. **因素缩放 + 反事实 (实验 2+3)**: 证明欧氏方法易被单因素主导, Mixed 更稳定
3. **码本利用率 + 有效容量 (实验 5)**: 排除提升来自崩塌/碰撞
4. **逐层截断 + 打乱 + 条件 probe (实验 6)**: 证明每层都有新增价值
5. **Decoder CE + beam 曲线 + Recall/NDCG (实验 7)**: 完成"编码机制 → 解码收益"闭环

### §3.6 完成指标

- MFSR 显著提高, 单因素主导率显著下降
- 层级/连续属性/行为结构同时得到较好保留 (Pareto 优)
- 有效码本容量不下降, 无严重崩塌
- SID 前缀与真实用户行为更一致
- 每一层都带来正的条件信息和 Recall 增益
- 解码 CE 更低, 小 beam 即可获高 Recall
- 去掉某曲率或打乱相应因素后, 优势明显消失

→ 共同证明: **Mixed 不是"用了更复杂距离", 而是通过混合曲率让一个固定长度码字同时吸收多个结构不同、但对推荐有用的因素, 并把这种联合表示转化成了更容易预测的 SID**。

### §3.7 风险

- **总 GPU 预算 ~25 GPU-h**, 单 GPU 串行 ~25 h (3 天)
- Euclidean-Concat baseline (P0) 必须先有, 否则后续诊断无法对比
- 5 个消融实验 (P9) 单项 ~2 GPU-h, 合计 ~10 GPU-h, 风险最高

### §3.8 关键依赖

- 复用 `task443/444` Phase 1 verdict (H_E_E_E R@10=0.0433)
- 复用 Task #2 Stage 1 FLAN-T5 XL embedding
- 复用 Task #17 Euclidean RQ-VAE baseline
- 复用 Task #441 Option C unified 训练脚本

### §3.9 verdict 路径

`task_artifacts/results/mixed_curvature/` (所有 verdict + 数据落地)

---

## §4 Task #56 — Phase 3 反事实验证 (因果验证)

> **目的**: 通过"反事实"训练验证 task458 链 3 结论 (brand-cluster 是 P0.3 提升的关键机制)
> **数据集**: 仅 Toys
> **预计 GPU 时间**: ~3.7 hours (1×A40)
> **优先级**: 🟡 中 (依赖 Task #58 P0.3 baseline)

### §4.1 关键发现 (v1.1, 2026-07-16 修订, 保留为最新结论)

**P0.3 实际 H 子空间位置**: 根据 `task58_p0_mixed_curvature_train.py:50-51`:
- `H_SUBSPACE_DIM = 32` (前 32 维走 Poincaré)
- `E_SUBSPACE_DIM = 896` (后 896 维走 Euclidean)

按 `build_concat_embedding.py:48` 排布:
- `[0:768]` = text (768-d)
- `[768:800]` = brand (32-d)
- `[800:896]` = taxonomy (96-d, 3 levels × 32)
- `[896:928]` = behavior (32-d)

**结论**: P0.3 的 H 子空间实际是 **text 前 32 维**, 不是 brand 子空间。

**task458 链 3 "brand MFSR +0.170" 是间接效应**: H 子空间强制 text 前 32-d 走双曲 → 整个 learning dynamic 改变 → brand 因子也间接更一致。

→ **Task #56 反事实设计必须修订**: 不能做"factor shuffle" (H 子空间在 text 不在 brand)。

### §4.2 修订后反事实设计 (4 个 variant)

| 反事实 | H_SUBSPACE_OFFSET | H_SUBSPACE_DIM | E_SUBSPACE_DIM | 描述 |
|--------|-------------------|----------------|----------------|------|
| **A** | 0 | 0 | 928 | 取消 H (应 ≈ P0.2) |
| **B** | 768 | 32 | 896 | H 在 brand [768:800] |
| **C** | 800 | 96 | 832 | H 在 taxonomy [800:896] |
| **D** | 896 | 32 | 896 | H 在 behavior [896:928] |

**预期**:
- 反事实 A R@10 ≈ P0.2 (~0.0034); P0.3 baseline 显著高于两者 (~0.0047)
- 反事实 B/C/D: H 子空间在不同位置的影响

**因果结论**: H 子空间 (即使在 text 前 32-d) 是 P0.3 提升的必要条件。

### §4.3 对照组 (5 个 variant)

| ID | 配置 | H 位置 | 用途 |
|----|------|--------|------|
| **P0.2** | 全 E | 无 | baseline |
| **P0.3 (现)** | text 前 32-d H | [0:32] | baseline |
| **反事实 A** | 全 E | 无 | 取消 H (应≈P0.2) |
| **反事实 B** | brand H | [768:800] | H 在 brand |
| **反事实 C** | taxonomy H | [800:896] | H 在 taxonomy |
| **反事实 D** | behavior H | [896:928] | H 在 behavior |

### §4.4 实施步骤

1. 修改训练脚本支持 `H_SUBSPACE_OFFSET` 参数 (`JointMixedCurvatureRQVAE.__init__` 加 `h_offset`)
2. 修改 `_compute_distance` 让 H 范围 = `[h_offset:h_offset+h_dim]`
3. 修改 `_lift_to_ball` 用 `x[:, h_offset:h_offset+h_dim]`
4. 修改 `concat_embedding` 输入顺序 (按 H offset 调整)
5. 写 4 个反事实 variant 启动脚本 (A/B/C/D)
6. 训练 4 个反事实 Stage 2.1 ckpt (~15-30 min each, seed=42, max_steps=3000)
7. 推断 SID (Stage 2.2, ~5 min each)
8. Stage 3 + Stage 4 + eval (~25 min each)
9. 收集 R@10, 对比 P0.3 baseline

### §4.5 预期判定规则

| A vs P0.3 | B vs P0.3 | C vs P0.3 | D vs P0.3 | 结论 |
|----|----|----|----|----|
| ≈ P0.2 | >+10% | ±5% | ±5% | ✅ **brand H 是关键**, 反事实 B 优于 P0.3 |
| ≈ P0.2 | ±5% | ±5% | ±5% | ⚠️ H 在哪里影响不大, 链 3 仅"text H"间接效应 |
| ≈ P0.2 | <-10% | <-10% | <-10% | ❌ H 在 brand/taxonomy/behavior 都损害 P0.3 |

### §4.6 完成指标

- 4 个 Stage 2.1 ckpt (A/B/C/D)
- 4 个 SID tensor
- 4 个 Stage 3 ckpt
- 4 个 Stage 4 R@K 评估
- verdict 文档: `verdicts/task56_result.md` (合并前 `task_artifacts/results/exp56/task56_phase3_verdict.md`)

### §4.7 风险

- **总 GPU 时间**: 4 × (Stage 2.1 ~30 min + Stage 4 ~25 min) ≈ 3.7 hours
- 修改 distance function 涉及多处代码 (`_compute_distance` / `_lift_to_ball` / `forward`), 需 unit test
- 反事实 A 应严格等价 P0.2 baseline, 是 sanity check

### §4.8 关键依赖

- `task_artifacts/scripts/mixed_curvature/task58_p0_mixed_curvature_train.py` (修改)
- `task_artifacts/scripts/mixed_curvature/build_concat_embedding.py` (按 H offset 重新生成 input)
- `multi_factor.pt` (11777 商品 × 4 因素)
- Toys dataset

### §4.9 后续

- 若 Task #56 确认 brand H 是最优 → 更新 P5 verdict v3 + paper
- 启动 Task #57 Phase 4 H_H_E_E / H_H_H_H 扩展
- 如果反事实 B/D 优于 P0.3 → 进一步消融 H 在不同位置的边际效应

---

## §5 Task #57 — Phase 4 扩展到 H_H_E_E / H_H_H_H

> **目的**: 把 P0.3 (H_E_E_E, taxonomy+brand 走 H) 扩展到多层 H, 验证 brand-cluster 机制在多层 H 下是否更显著
> **数据集**: 仅 Toys
> **预计 GPU 时间**: ~90 min (1×A40)
> **优先级**: 🟡 中 (依赖 Task #56 链 3 验证)

### §5.1 背景

P0.3 (H_E_E_E) brand MFSR L1 +0.170 显著 (间接效应, 见 §4.1)。

**问题**: 如果 H 子空间强制 brand 聚类, **多层 H (H_H_E_E, H_H_H_H)** 是否会更激进地聚类 brand?

### §5.2 三个变体

| 配置 | L1 距离 | L2 距离 | L3 距离 | 目的 |
|------|---------|---------|---------|------|
| **P0.3 (H_E_E_E)** | H (32-d) | E | E | baseline (Task #58 P0.3) |
| **H_H_E_E** | H | H | E | L1+L2 brand-cluster |
| **H_H_H_H** | H | H | H | 全层 brand-cluster |

**多因素分配**:
- text (768-d): 始终 E
- taxonomy (96-d): 跟随 layer 距离类型
- brand (32-d): 始终 H
- behavior (32-d): 始终 E

注意: 每层有 4 个子空间, 每个子空间的距离 = 该层设置的 (H or E), 不是 fixed H。

### §5.3 实施步骤

1. 修改训练脚本支持 per-layer 距离类型 (~1 hour)
2. 训练 H_H_E_E ckpt (Stage 2.1, ~30 min)
3. 训练 H_H_H_H ckpt (Stage 2.1, ~30 min)
4. Stage 2.2 SID tensor for both (~10 min total)
5. Stage 3 + Stage 4 + eval (~25 min × 2)
6. 对比 P0.3 baseline

### §5.4 预期判定规则

| H_H_E_E R@10 vs P0.3 | H_H_H_H R@10 vs P0.3 | 结论 |
|----|----|----|
| >+10% | >+20% | ✅ 多层 H 更激进 brand-cluster 有效 |
| ±5% | ±5% | ⚠️ 边际效益不足 |
| <-5% | <-10% | ❌ 多层 H 过聚类, 损失 fine-grained |

### §5.5 完成指标

- H_H_E_E / H_H_H_H Stage 2.1 ckpt
- 2 个 SID tensor
- 2 个 Stage 4 R@K 评估
- verdict 文档: `verdicts/task57_result.md` (合并前 `task_artifacts/results/exp57/task57_phase4_verdict.md`)

### §5.6 风险

- Stage 2.1 训练 + Stage 4 eval 总 GPU 时间 ~90 min
- 如果 H_H_H_H 完全过聚类 → 所有商品撞到同一 L1 码字 → TIGER 无法区分
- 需准备 fallback: 如果 H_H_H_H R@10 < P0.3 30% → 立即终止

### §5.7 后续

- 若 H_H_E_E / H_H_H_H 优于 P0.3 → 更新 P5 verdict v4 + paper
- 若无显著差异 → 保留 P0.3 为推荐配置

---

## §6 Task #68 — L1 品牌信息利用 (side feature concat)

> **任务目的**: 验证 Task #67 Exp D3 的发现 (L1 throughput 2.32× 专门捕获品牌) 能否被实际利用, 提升 R@10 baseline (Task #65 = 0.09710)。在 TIGER 输入 embedding 中 concat L1 brand side feature, 看是否突破 baseline。
> **状态**: 🟡 待启动
> **与 §3-§5 解耦**: 复用 Task #65 RQ-VAE ckpt, 只改 Stage 3 TIGER

### §6.1 背景

#### 前置结论

- Task #65 baseline: R@10=**0.09710** ∈ [0.090, 0.105]
- Task #67 Exp D3: L1 raw→L1 throughput = **2.32×** (L1 专门捕获品牌)
- Task #66: 加权几何在 L2 归一化下等价单距离 → 几何路线收尾

#### 假设 R1

如果 L1 真的"专门捕获品牌" (throughput 2.32×), 那么把 L1 codeword 解码出的 brand 信息作为 side feature 加到 TIGER 输入, 应该让 TIGER 学到 brand-aware 的 next-item 预测 → R@10 > 0.09710。

### §6.2 实验设计

**变量**: Stage 3 TIGER 输入是否拼接 L1 brand side feature

**保持不变**:
- Task #65 RQ-VAE ckpt (Stage 2.1)
- Task #65 Stage 2.2 dedup SID tensor (4, 11924)
- sequence_length=120, seed=42, num_hierarchies=4

**两个版本对比**:

| 版本 | TIGER 输入 | R@10 | 备注 |
|------|-----------|------|------|
| Baseline | item token only | 0.09710 (Task #65) | 重跑确认 |
| **+ L1 brand** | item token ⊕ brand embedding (concat) | ? | 本任务 |

#### §6.2.1 Side feature pipeline

```
Stage 2 RQ-VAE → L1 codeword (B,) for each item
                ↓ lookup
                brand name (B,) string
                ↓ hash → embedding
                brand_emb (B, 128)
                ↓ concat
                [item_emb; brand_emb] → TIGER
```

**实现步骤**:
1. 用 Task #65 RQ-VAE 推断 L1 codeword per item (11924 items × 256 cluster ids)
2. 训练 brand lookup table: brand_name → 128-dim embedding
3. 包装 TIGER dataset: 在 item embedding 后 concat brand embedding
4. 训练 Stage 3 (num_hierarchies=4, max_steps=50000)
5. Stage 4 推断 + R@10 eval

### §6.3 决策触发 (vs Task #65 baseline)

| R@10 | 决策 |
|------|------|
| **> 0.09710** | ✅ **L1 品牌信息有效** — 登记 Task #69 进一步探索 (L2/L3 side feature) |
| ∈ [0.090, 0.09710] | ⚠️ 与 baseline 持平 — side feature 无显著影响; 记录到 verdict |
| < 0.090 | ❌ side feature 有害; 回滚到 Task #65 baseline, 登记"非 side feature 改进"路线 |

### §6.4 预算

| 阶段 | 估算时间 | GPU |
|------|---------|-----|
| L1 codeword 推断 | ~5 min | GPU 0 |
| Brand embedding 训练 (lookup) | ~10 min | CPU |
| Stage 3 重训 (max_steps=50000) | ~3 h | GPU 1 |
| Stage 4 推断 | ~5 min | GPU 2 |
| R@10 eval | ~1 min | CPU |
| **总计** | **~3.5 h** | 1 卡主训 + 2 卡辅助 |

### §6.5 风险与缓解

**风险 1**: Brand lookup 维度选错 → side feature 无效
→ 缓解: 试 64/128/256 三档

**风险 2**: Concat 位置不对 (concat 到 token vs concat 到 embedding) 影响结果
→ 缓解: 文档化两种位置都试

**风险 3**: Stage 3 训练时长 ~3h, 单 cron tick 跑不完
→ 缓解: setsid + nohup 后台跑, cron 下次 tick 检查进度

### §6.6 完成度跟踪

- [ ] L1 codeword per item (11924)
- [ ] Brand lookup table (brand_name → 128-dim embedding)
- [ ] TIGER dataset 包装 (concat brand emb)
- [ ] Stage 3 launch (setsid + nohup)
- [ ] Stage 3 finish (val/recall@5 监控)
- [ ] Stage 4 inference
- [ ] R@10 eval (vs 0.09710)
- [ ] 写 verdict → `verdicts/task68_result.md` (合并前 `verdicts/task68_l1_brand_side.md`)
- [ ] 更新 §16 表格

### §6.7 备注

- 这是 Task #67 Exp D3 的直接 follow-up: 理论发现 → 工程验证
- 若成功, P5 paper 增加 "Layer-aware side features boost R@10 from 0.09710 to X.XX" 节
- 若失败, 反而强化"信息分工对 R@10 无显著影响"的结论
- 启动命令: `bash scripts/task68_l1_brand_side.sh` (待写)

---

## §7 合并说明表格

### §7.1 任务对比

| 维度 | Task #58 (Phase 2) | Task #56 (Phase 3) | Task #57 (Phase 4) | Task #68 (L1 side) |
|------|--------------------|--------------------|--------------------|---------------------|
| **阶段** | 因果链建立 | 反事实验证 | 多层 H 扩展 | TIGER 端利用 |
| **核心问题** | Mixed 真由多因素决定吗? | H 子空间是必要条件吗? | 多层 H 更激进吗? | L1 brand 能被利用吗? |
| **改动层** | Stage 2.1 (距离函数) | Stage 2.1 (H offset) | Stage 2.1 (per-layer) | Stage 3 (TIGER 输入) |
| **依赖基线** | Task #17 Euclidean | Task #58 P0.3 | Task #56 P0.3 | Task #65 (TIGER) |
| **GPU 时间** | ~25 GPU-h | ~3.7 h | ~90 min | ~3.5 h |
| **baseline R@10** | Task #17 (~0.040) | P0.2 ~0.0034 | P0.3 ~0.0047 | Task #65 0.09710 |
| **目标 R@10** | 显著高于 Euclidean-Concat | ≈ P0.2 (A), ±5%-10% (B/C/D) | >+10% / >+20% | > 0.09710 |
| **核心风险** | 5 个消融 ~10 GPU-h | 改 distance 需 unit test | H_H_H_H 过聚类 | brand lookup 维度 |
| **verdict 路径** | `task_artifacts/results/mixed_curvature/` | `verdicts/task56_result.md` | `verdicts/task57_result.md` | `verdicts/task68_result.md` |
| **任务优先级** | P0 启动 | 🟡 中 | 🟡 中 | 🟡 待启动 |

### §7.2 因果链依赖关系

```
Task #58 (Phase 2)
   ├─ P0: Euclidean-Concat baseline 训练 (最关键 baseline)
   ├─ P1: 诊断实验 1 (MFSR/Dominance)
   ├─ P2: 诊断实验 5 (码本诊断)
   ├─ P3-P8: 其余诊断实验
   └─ P9: 5 个消融实验
        ↓
   Task #56 (Phase 3) [v1.1: H 在 text 前 32-d]
        ├─ 反事实 A: 取消 H (应 ≈ P0.2)
        ├─ 反事实 B: H 在 brand [768:800]
        ├─ 反事实 C: H 在 taxonomy [800:896]
        └─ 反事实 D: H 在 behavior [896:928]
             ↓
        Task #57 (Phase 4)
             ├─ H_H_E_E: L1+L2 brand-cluster
             └─ H_H_H_H: 全层 brand-cluster
                  ↓
   (与 #58/#56/#57 解耦)
        Task #68 (L1 brand side)
             ├─ 复用 Task #65 RQ-VAE ckpt
             └─ Stage 3 TIGER 加 L1 brand embedding
```

### §7.3 v1.1 反事实设计 (重要: 保留为最新结论)

**P0.3 H 子空间实际位置 = text 前 32-d, 不是 brand** (来自 `task58_p0_mixed_curvature_train.py:50-51`)

**反事实设计修订**:
- ❌ 不能做"factor shuffle" (H 在 text 不在 brand)
- ✅ 改 H_SUBSPACE_OFFSET 扫描 4 个位置: 0/768/800/896
- ✅ 反事实 A (H=0) 是核心 sanity check: 严格等价 P0.2 baseline

**task458 链 3 "brand MFSR +0.170" 是间接效应**:
- H 子空间强制 text 前 32-d 走双曲
- 整个 learning dynamic 改变
- brand 因子也间接更一致

### §7.4 关键交叉引用

| 概念 | 涉及 task | 章节 |
|------|----------|------|
| MFSR / Dominance | #58 | §3.3 实验 1 |
| 反事实对 / 尺度放大 | #58 | §3.3 实验 2-3 |
| 有效码本容量 | #58 | §3.3 实验 5 |
| 每层贡献 / 条件 probe | #58 | §3.3 实验 6 |
| decoder CE / beam 曲线 | #58 | §3.3 实验 7 |
| 行为前缀 Hit@K | #58 | §3.3 实验 8 |
| H_SUBSPACE_OFFSET 扫描 | #56 | §4.2 |
| per-layer distance | #57 | §5.2 |
| brand side feature | #68 | §6.2.1 |
| L1 throughput 2.32× | #68 | §6.1 (来自 Task #67) |

### §7.5 verdict 路径统一

合并后所有 verdict 统一到 `verdicts/task<N>_result.md` 格式:

| 原 task | 合并前 verdict 路径 | 合并后 verdict 路径 |
|--------|--------------------|--------------------|
| #56 | `task_artifacts/results/exp56/task56_phase3_verdict.md` | `verdicts/task56_result.md` |
| #57 | `task_artifacts/results/exp57/task57_phase4_verdict.md` | `verdicts/task57_result.md` |
| #58 | `task_artifacts/results/mixed_curvature/` | `verdicts/task58_result.md` |
| #68 | `verdicts/task68_l1_brand_side.md` | `verdicts/task68_result.md` |

---

**合并完成日期**: 2026-07-18
**合并原因**: 四个任务共属 "brand 因果链" 主题, 互相依赖, 统一管理
**保留结论**: Task #56 v1.1 反事实设计 (H 在 text 前 32-d, brand MFSR +0.170 是间接效应) 为最新结论
