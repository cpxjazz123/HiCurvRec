# Task 22 (主线 #2): HRQ 双曲残差量化 — 把残差空间拉成负曲率, 看层级距离是否与 co-purchase 相关

> **状态**: 🟡 backlog

> **目的**：从"几何"维度改残差量化——把欧几里得码本换成**庞加莱球码本** (Poincaré ball)，看是否能让层级距离与商品 hierarchy / co-purchase 相关。
> **依赖**：task20 量 4 f_radial + 量 5 D_l + task17 Group A SID tensor + flan-t5-xl embedding
> **环境**：`conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys`
> **目标仓库**：`/fs04/ar57/wenyu/GeneRec/GRID`

---

## 总目标

**主 claim**：双曲残差量化 (HRQ, Hyperbolic Residual Quantization) 让**层级距离** (Poincaré 球距离) 与 **co-purchase correlation** (Amazon "also-bought"/"frequently-bought-together" 信号) 显著相关——比欧几里得残差量化 (Group A/B/C) **更显著**。

**3 道硬门（kill line）**：
- 几何 A (双曲 vs 欧氏) — kill = 双曲 / 欧氏 在 co-purchase 相关性上无差异（rank corr 差 < 0.05）
- 几何 B (δ/diam 准则) — kill = HRQ δ/diam 远超理论下界 (>10×) 或 不收敛
- 几何 C (层级与共购) — kill = HRQ 层级距离与 co-purchase 相关 < 欧氏残差（差 < 0）

---

## 理论支撑 (为什么要双曲?)

### 树状层级数据的几何先验
- 类别层级树 (Toys → Outdoor → Bicycles → Mountain Bikes) 是 **指数分支** 结构
- 欧几里得距离无法指数分级（球壳体积增长 r^(d-1) 太快）
- 双曲 (Poincaré ball) 体积增长 e^r，自然匹配层级
- 已有证据：Hyperbolic Embedding (Nickel & Kiela 2017)、Hyperbolic VAE (Mathieu 2019)、Hyperbolic Diffusion (Chen 2022)

### 残差量化的双曲解释
RQ 在欧氏空间把 residual 拆为**嵌套的球壳**。但 Toys 类目层级实际上是**树状**：
```
            Toys
        /   |    \
    Outdoor Craft  Games
    /    \         |
 Bikes  Tents   Cards
  |
MTB
```

如果把"层级"解释为"分支深度"，那么：
- L1 量化 = 主要分支 (Toys / Outdoor / Games)
- L2 量化 = 子分支 (Outdoor → Bikes / Tents; Games → Cards)
- L3 量化 = 叶节点 (Bikes → MTB; Cards → Rummy)

**HRQ 的假设**：L1/L2/L3 在**双曲空间**对应**指数分支**，层级距离 (geodesic) 应与商品**共购网络距离**强相关。

---

## 实现路径

### 三步走

#### Step 1: HRQ 码本训练（双曲 K-means）

**输入**：flan-t5-xl embedding `e_i ∈ R^D`, 残差 `r_0 = e_i`
**算法**：双曲 K-means (hyperboloid model)
```python
# 1. 把 e_i 投到 Lorentz model: h_i = [sqrt(1+||e_i||²), e_i]  (Lorentz hyperboloid)
# 2. 在 Lorentz 距离下做 K-means:
#    d_Lorentz(h, c) = arccosh(-<h, c>_L)   # <- Lorentz inner product
# 3. update centroid: h_c ← exponential_map(h_c, mean_log_map(assigned_points))
# 4. 得到 Cluster 1 centroid h_1 ∈ H^D, 然后 r_1 = exp(-d_Lorentz) × difference
```

#### Step 2: 残差逐层双曲量化
```python
# L1: r_0 = e_i
#    h_1_code = argmin_k d_Lorentz(r_0_lift, C_1[k])
#    r_1 = exp_neg_d × (projection onto tangent at h_1_code)
# L2: 用 r_1, 重复 step 1
# L3: 重复
```

#### Step 3: 推断 (与欧氏 RQ 一样)
- 给出 e_i → 输出 (c_1, c_2, c_3) ∈ [256] × [256] × [256]
- dedup digit 加入 → 4-tuple SID

### 实现复杂度

| 模块 | 改 / 新 | 工作量 |
|------|---------|--------|
| `src/modules/clustering/hyperbolic_kmeans.py` | **新文件** | ~200 行 |
| `src/modules/clustering/mini_batch_kmeans.py` | 改 (加 `use_hyperbolic=True` 开关) | ~50 行 |
| `src/models/modules/semantic_id/rqvae_model.py` 或 semantic_id module | 改 (RQ forward 双曲分支) | ~80 行 |
| `configs/experiment/hrq_train.yaml` | 新 | ~30 行 (复 rkmeans_train + hyperbolic params) |
| `configs/experiment/hrq_inference.yaml` | 新 | ~20 行 |
| task20_group_d_s{21,22,s3,s4,eval}.sh | 5 个新脚本 | ~150 行总 |

### 关键实现选择

**Lorentz model vs Poincaré ball**
- Lorentz: 算术稳定 (numerically stable), centroid update 更简单
- Poincaré: 更直观 (可视化为单位球), 但边界处 numerical issue
- **选择 Lorentz** (内部计算用 Lorentz, 输出日志可投影到 Poincaré)

**残差映射方式**
- "exp_map with magnitude scaling" (推荐): `r_{l+1} = exp_{h_code}(project_to_tangent) × decay_factor`
- "Möbius subtraction": `r_{l+1} = h ⊖ h_code = ((1+||c||²)x - (1+||x||²)c) / (1 + <x, c>²)` (更几何)
- **先实现 exp_map 版本** (更接欧氏 RQ 经验，工程风险低)

### Poincaré 距离 ↔ Co-purchase 关联测

```python
# 1. 取 HRQ SID: (N, 3) digit sequence
# 2. 算 pairwise Poincaré 距离:
#    d_H(i, j) = sum_l w_l × d_Poincaré(C_l[c_l(i)], C_l[c_l(j)])
# 3. 取 Amazon interactions.csv 的 co-purchase 频率: freq(i, j) ∈ [0, N]
# 4. 计算 spearman rank correlation: corr(d_H, freq)
# 5. 对照: 欧氏 RQ (Group A/B/C) 的同样 pairwise distance vs freq
# 6. 判定: HRQ corr > Group A/B/C corr (高 0.05+) → 通过
```

---

## 实验设计

### 阶段 1: HRQ Stage 2.1 (双曲码本训练)
- 输入：Toys flan-t5-xl embedding (11924 × 2048)
- 输出：HRQ codebook H_1, H_2, H_3 ∈ H^2048
- 时间：~1 GPU 1h
- 验证：HRQ 收敛曲线 (QErr vs step) vs RQ (Group A 同配置)

### 阶段 2: HRQ Stage 2.2 (推断 + SID)
- 输出：HRQ merged_predictions_tensor.pt (11924, 4)
- 时间：~5 min
- 验证：碰撞率 / per-layer digit 分布

### 阶段 3: HRQ Stage 3 (TIGER T5 训练)
- 输出：tiger_decoder_only_{best,last}.ckpt
- 时间：~30 min (Group A 同配置)

### 阶段 4: HRQ Stage 4 (推断)
- 输出：(19412, 10, 4) prediction tensor
- 时间：~30 s

### 阶段 5: HRQ eval + Pairwise co-purchase 相关
- 5a: R@5, R@10, N@5, N@10 (与 Group A/B/C 对照)
- 5b: pairwise Poincaré 距离 vs co-purchase 频率 → spearman corr
- 5c: 与 Group A (欧氏) 同样测的 spearman corr 对照

### 数据准备
- 共购网络：从 `data/amazon_data/toys/interactions.csv` 算 **co-occurrence** (i, j) 共现次数
- 同一 CSV 在 Group A 评估时已加载，可复用
- 关键的：`freq(i, j) = |{users who bought both i and j}|`

---

## 期望 / Kill 线

### 期望 A (几何优势)
- HRQ 与 Group A 端到端 R@10 **持平** (差 < 0.01)
- 但 **pairwise co-purchase corr** HRQ > Group A by **+0.10**

### 期望 B (δ/diam 准则)
- HRQ 的层级 δ (1-layer hyperbolic distance) 与 diam (full tree hyperbolic diameter) 之比符合 Gromov (1987) 树 criterion: **δ/diam ∈ [1/n_layer, 1/2]**

### 期望 C (层级与共购)
- HRQ L1 距离 (Coarse-grained) 与 co-purchase 弱相关
- HRQ (L1, L2) 距离 (Fine-grained) 与 co-purchase 强相关
- Coarse 与 Fine 的差值即为 **HRQ 真正的层级信号**

### Kill 线
| 现象 | pass 条件 | fail 动作 |
|------|-----------|----------|
| 几何 A | HRQ pairwise corr > Group A by ≥ 0.05 | 提升 method 评分 (++method-quotient) |
| 几何 A fail | HRQ ≤ Group A by any amount | **主线撤** (但仍可作为基线报告) |
| 几何 B | δ/diam ∈ [1/n_layer, 1/2] | 报告 Gromov 违背，但不动 method |
| 几何 B fail | δ/diam > 1/2 或 < 1/n_layer | **理论不稳**，需重新校准 train param |
| 几何 C | Coarse/Fine 与共购相关性强 | 报告层级信号存在 |
| 几何 C fail | Coarse 与 Fine 一样相关 (差 < 0.03) | **claim 不可证**, 只报告"双曲没破坏即可" |

---

## 产物清单

```
result/task22/
├── task20_hrq_pipeline.json          # Stage 2.1/2.2/3/4 串行 ckpt/时间
├── task20_hrq_eval.json              # R@5/R@10/N@5/N@10 对照 A/B/C
├── task20_pairwise_copurchase.json   # HRQ vs Group A/B/C 的 pairwise corr
├── task20_pairwise_copurchase.png    # 4 算法 pairwise corr 对比图
├── task20_delta_diam.json            # HRQ δ/diam per layer
├── task20_layer_corr.json            # HRQ Coarse/Fine 与共购的相关
├── task20_layer_corr.png             # HRQ 层级 vs 共购 相关曲线
├── task20_hrq_geometry.md            # 几何判据对照 (Gromov 检验)
└── task20_verdict.md                 # 主线判定
```

---

## 完成判定

| 子任务 | pass 条件 | 主线动作 |
|--------|-----------|----------|
| 几何 A | HRQ pairwise corr > Group A by ≥ 0.05 | ✅ 升级 method claim |
| 几何 A fail | HRQ ≤ Group A | ❌ 主线撤 |
| 几何 B | δ/diam ∈ [1/n_layer, 1/2] | ✅ Gromov criterion 满足 |
| 几何 B fail | δ/diam > 1/2 或 < 1/n_layer | ⚠️ 理论不稳 |
| 几何 C | Coarse/Fine 与共购相关差距 ≥ 0.10 | ✅ 层级信号清晰 |
| 几何 C fail | Coarse/Fine 差距 < 0.03 | ❌ claim 不可证 |

## 执行顺序

1. **Step 1 (双曲 K-means 训练)** + **Step 2 (残差逐层)** — 串行
2. **Stage 2.2 推断 + Stage 3 训练** — 与 Group A 评估可并行 (不同 GPU)
3. **Stage 4 推断 + eval** — 串行
4. **pairwise co-purchase 计算 + 相关性** — 与 Stage 4 评估可并行
5. **verdict.md 与论文故事线** — 最终聚合

## 风险与 fallback

- **双曲 K-means 收敛慢** → reduce lr 或 max_iter; fallback 用 sklearn 双曲 K-means (gyorodi 库或 geoopt)
- **Poincaré 数值崩** → 改 Lorentz model (numerical stable)
- **Task 22 训练 + Stage 3 时间超预算** → 共购相关测可只用 Stage 2.2 (无需 Stage 3 训完)
- **Co-purchase 数据稀疏** → 同时测**分类层级** (Toys 类目树从 Amazon metadata 提取) 作为 supplementary

## 工程依赖

| 依赖 | 是否已有 | 处理 |
|------|----------|------|
| geoopt (双曲深度学习) | ❓ 需 check | 若无, `pip install geoopt` |
| flan-t5-xl embedding | ✅ task1 | 复用 |
| Group A SID tensor | ✅ task17 | 复用 |
| TIGER ckpt infrastructure | ✅ task17 | 复用 |
| Amazon interactions.csv | ✅ data | 复用 |

## 与主线 #1 的关系

HRQ 是**主线 #1 之外的另一个独立 axis (几何 vs 信息)**：
- 主线 #1 (task21): 信息集中 L1 现象是否独立于 ReSID
- HRQ (task22): 双曲几何是否能解释类目层级

二者可同时推进，互不阻塞。
