# Task 23 (主线 #4): 加性量化 (AQ) — 消首层独裁的"结构解药"

> **状态**: 🟡 backlog

> **目的**：从"结构"维度改残差量化——把"嵌套" (RQ) 改成"加性" (AQ)，让**每层独立贡献**信息，看是否能打破"first layer dictatorship"。
> **依赖**：task20 量 11 (Δ_1) → 验证"独裁"现象存在 + task21 #1 主线 + task19 量 6 (η_l)
> **环境**：`conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys`
> **目标仓库**：`/fs04/ar57/wenyu/GeneRec/GRID`

---

## 总目标

**主 claim**：RQ 的"first-layer dictatorship" (Δ_1≫Δ_2) 是**结构伪影**——把 RQ 换成 **加性量化 (AQ, Additive Quantization)** 让每层独立编码后，**Δ_1/Δ_2 比值应显著降低**（不依赖 magnitude / 不依赖正交化）。

**3 道硬门（kill line）**：
- 结构 A (逐层 drop) — kill = drop L1 后 R@10 崩塌 > 70% (独裁未被消除)
- 结构 B (Δ_1/Δ_2 ≈ 1) — kill = AQ Δ_1/Δ_2 仍然 > 5 (AQ 也独裁)
- 结构 C (AQ re-encode代理) — kill = AQ SID 与 RQ SID 完全无关 (correlated < 0.3 → 不可作为 proxy)

---

## 第一性原理 (为什么加性?)

### RQ 的"嵌套"问题
```
RQ:  x ≈ C_1[c_1] + r_1,   r_1 ≈ C_2[c_2] + r_2,   r_2 ≈ C_3[c_3]
                    └─ L1 主导 ─┘
```

L1 量化后残差 r_1 在 **正交于 C_1[c_1]** 的子空间上 — L2 只能看 r_1，**L2 没拿到全部 x 的视角**。

→ 这就是为什么 Δ_1≫Δ_2：因为 C_1[c_1] 锁住了**大部分**幅度，L2 的可用空间被压缩。

### AQ 的"加性"立场
```
AQ:  x ≈ C_1[c_1] + C_2[c_2] + C_3[c_3]   (无嵌套, 无 r_l)
```

每层独立编码完整 x 的不同部分：
- L1: C_1[c_1] 主要捕获 x 的低频（粗粒度）
- L2: C_2[c_2] 捕获**残余误差的高频**，与 L1 正交但无顺序依赖
- L3: 继续捕获残差的高频细节

### AQ 关键优势
- 没有"先 L1 后 L2"的次序 → 没有"first-layer dictatorship"
- 整个优化的解更接近**码本联合优化**的解（虽然实际上迭代解）
- 已有论文 (Babenko & Lempitsky, 2014 "Additive Quantization") 在 vector compression 域证明 AQ > RQ

---

## 实现路径

### 三步走

#### Step 1: AQ 码本训练（联合 K-means）

**输入**：flan-t5-xl embedding `e_i ∈ R^D`
**算法**：联合码本 + 分配 优化 (类似 beam search)
```python
# 1. 初始化码本: C_l ∈ R^{256 × D} for l=1,2,3 (随机采样 + K-means warmup)
# 2. 固定码本, 找最优分配: (c_1, c_2, c_3) for each x_i
#    最小化 ||x_i - C_1[c_1] - C_2[c_2] - C_3[c_3]||²
#    → 多起点 beam search (beam_size=10), exhaustive find (c_1, c_2, c_3) for each x
# 3. 固定分配, 更新码本:
#    对每个 (c_1, c_2, c_3) 组合, 用 assigned points 算条件均值
#    或对每个 c_l, 加其他层固定, 更新 C_l[c_l] = mean(assigned points 减去其他层贡献)
# 4. 重复 2-3 直到收敛
```

#### Step 2: AQ 推断 + SID
```python
# 给定 e_i → 输出 (c_1, c_2, c_3) ∈ [256] × [256] × [256]
# dedup digit 加入 → 4-tuple SID
```

### 实现复杂度

| 模块 | 改 / 新 | 工作量 |
|------|---------|--------|
| `src/modules/clustering/additive_quantization.py` | **新文件** | ~250 行 |
| `src/modules/clustering/mini_batch_kmeans.py` | 不动 | 0 |
| `src/models/modules/semantic_id/aq_model.py` | **新文件** | ~80 行 |
| `src/models/modules/clustering/__init__.py` | 加 AQModel | ~5 行 |
| `configs/experiment/aq_train.yaml` | 新 | ~30 行 |
| `configs/experiment/aq_inference.yaml` | 新 | ~20 行 |
| `task21_group_d_s{21,22,s3,s4,eval}.sh` | 5 个新脚本 | ~150 行 |

### 与 RQ 复用

- TIGER T5 训练 (`tiger_train_flat`) 可直接复用, 只换 SID tensor
- 评估 (`task15_eval.py`) 可直接复用
- 推断 (`tiger_inference_flat`) 可直接复用
- 共减少 ~80% 工作量

### beam search 大小选择

| beam_size | 时间 | 准确度 |
|-----------|------|--------|
| 1 (greedy) | ~10 min/iter | 弱 |
| 5 | ~30 min/iter | 中 |
| 10 | ~60 min/iter | 好 |
| 50 | ~5h/iter | 很好 |

**推荐 beam_size=10**, 一共 ~20 iter 收敛 (~20h)。可分批跑, 每 ~3h 存一次。

### 难点

**梯度**: AQ 不易反向传播，但训练完不需要梯度（TIGER 不反向到 AQ codebook），只需**前向**分配。OK。

**初始化**: K-means warmup 容易陷入局部解。可用 RQ (Group A) 的 C_l 作为 AQ C_l 初始（**强先验**）。

---

## 实验设计

### 阶段 1: AQ Stage 2.1 (码本训练)
- 输入：Toys flan-t5-xl embedding (11924 × 2048)
- 输出：AQ codebook C_1, C_2, C_3 ∈ R^{256 × 2048}
- 时间：~2-3 GPU h (beam_size=10, ~20 iter)
- 验证：AQ 收敛曲线 (QErr vs iter) vs RQ (Group A 同配置)

### 阶段 2: AQ Stage 2.2 (推断 + SID)
- 输出：AQ merged_predictions_tensor.pt (11924, 4)
- 时间：~5 min
- 验证：碰撞率 / per-layer digit 分布

### 阶段 3: AQ Stage 3 (TIGER T5 训练)
- 输出：tiger_decoder_only_{best,last}.ckpt
- 时间：~30 min

### 阶段 4: AQ Stage 4 (推断 + eval)
- 5a: R@5, R@10, N@5, N@10 (与 Group A/B/C/HRQ 对照)
- 5b: pairwise AQ digit vs Group A digit 相关性（代理测）
- 5c: drop-L1 实验（首次在 AQ 上验证）

### Drop 实验设计

| 配置 | 实现 | 测什么 |
|------|------|--------|
| Full SID (4 digits) | (c_1, c_2, c_3, c_4) | baseline R@10 |
| Drop L1 | (c_2, c_3, c_4) | L1 贡献 |
| Drop L2 | (c_1, c_3, c_4) | L2 贡献 |
| Drop L3 | (c_1, c_2, c_4) | L3 贡献 |
| Drop dedup | (c_1, c_2, c_3) | dedup 贡献 |
| Random shuffle | 同样 4 digits, 随机打乱顺序 | 对照 |

→ 输入到 TIGER 时把被 drop 的位随机填 (uniform random) 或用 mask

**期望 R@10 drop 比率**:
- RQ: Drop L1 → R@10 跌 70%+ (独裁)
- AQ: Drop L1 → R@10 跌 ~33% (1/3)
- AQ: Drop L3 → R@10 跌 ~33% (1/3)
- **AQ 各层 drop 比率应该接近相等** (no dictatorship)

---

## 期望 / Kill 线

### 期望 A (结构均衡)
- AQ R@10 **>= Group A** (持平)
- AQ 各层 drop 比率 = (33%, 33%, 33%) ± 5%
- RQ 各层 drop 比率 = (73%, 18%, 9%) (L1 主导)

### 期望 B (Δ_1/Δ_2 ≈ 1)
- AQ 的"逐层码本失配 Δ_l" 计算: Δ_l = ||AQ SID[:, :l+1] → AQ SID[:, :l+1 严格best l+1码本|| / ||baseline||
- 期望 Δ_1/Δ_2 ≈ 1 (均匀)
- vs RQ Δ_1/Δ_2 = 15.9/2.29 ≈ 7

### 期望 C (AQ proxy 可用)
- AQ SID digit-1 vs RQ (Group A) SID digit-1 的 rank correlation
- 期望 <= 0.3 (**低相关**: AQ 不只是 RQ 的近似)
- 期望 >= 0.1 (**有相关**: AQ 与 RQ 不完全无关)

### Kill 线

| 现象 | pass 条件 | fail 动作 |
|------|-----------|----------|
| 结构 A | AQ 各层 drop 比率 R@10 (max-min) < 8% | ✅ 结构均衡 |
| 结构 A fail | AQ drop L1 仍 > 50% | ❌ 独裁未消 |
| 结构 B | AQ Δ_1/Δ_2 < 2 | ✅ AQ 不独裁 |
| 结构 B fail | AQ Δ_1/Δ_2 > 5 | ❌ AQ 也独裁 (结构性伪影不成立) |
| 结构 C | AQ vs RQ digit-1 rank corr ∈ [0.1, 0.3] | ✅ AQ 是独立 proxy |
| 结构 C fail | AQ digit-1 corr > 0.5 或 < 0.05 | ❌ AQ 极端 (与 RQ 等价/或纯随机) |

---

## 产物清单

```
result/task23/
├── task21_aq_pipeline.json           # Stage 2.1/2.2/3/4 串行 ckpt/时间
├── task21_aq_eval.json               # R@5/R@10/N@5/N@10 对照 A/B/C
├── task21_drop_experiment.json      # 各层 drop R@10 对照 (AQ vs RQ Group A)
├── task21_drop_experiment.png        # drop R@10 柱状图 (3 算法 × 5 配置)
├── task21_delta_per_layer.json       # AQ Δ_1/Δ_2/Δ_3 序列
├── task21_delta_per_layer.png        # 3 算法 Δ_l 对比
├── task21_aq_rq_proxy.json           # AQ digit 与 RQ digit 相关矩阵
├── task21_aq_rq_proxy.png            # 3×3 digit1/2/3 相关矩阵图
└── task21_verdict.md                 # 主线判定
```

---

## 完成判定

| 子任务 | pass 条件 | 主线动作 |
|--------|-----------|----------|
| 结构 A | AQ drop L1 R@10 < 50% 且 max-min < 8% | ✅ 结构均衡 |
| 结构 A fail | AQ drop L1 > 50% 或 max-min > 10% | ❌ 独裁未消 |
| 结构 B | AQ Δ_1/Δ_2 < 2 | ✅ AQ 不独裁 |
| 结构 B fail | AQ Δ_1/Δ_2 > 5 | ❌ 结构性伪影不成立 |
| 结构 C | AQ vs RQ digit-1 corr ∈ [0.1, 0.3] | ✅ AQ 独立 proxy |
| 结构 C fail | corr > 0.5 或 < 0.05 | ❌ AQ 极端 |

## 执行顺序

1. **Step 1 (AQ 训练)** + **Step 2 (AQ 推断)** — 串行
2. **Stage 3 TIGER 训练 + Stage 4 推断 + eval** — 与 Stage 2 串行
3. **drop 实验 + AQ vs RQ 相关** — 与 Stage 3 平行（只需 SID tensor, 不需 T5 ckpt）
4. **verdict.md 与论文故事线** — 最终聚合

## 风险与 fallback

| 风险 | fallback |
|------|----------|
| AQ 训练收敛慢 | beam_size=5 或 warmup from RQ codebook |
| AQ R@10 显著弱于 RQ | 仍然 report — AQ 提供"结构均衡"对照 |
| Drop 实验 TIGER 多次重训 | **fallback**: 修改 `tiger_inference_flat` 直接 mask SID, **不需要重训** T5 — 已有 TIGER ckpt 也能处理 masked SID (with mask input) |
| beam search 内存超 | subsample N=1000 → 训码本 → 全量 infer |

### Critical Fallback: Inference-Time Drop (不需要重训 T5)

```python
# 已有 Group A TIGER ckpt 也可使用, 用 masked inference:
# 输入: (c_1=random, c_2, c_3, c_4) 模拟 drop L1
# 用 T5 encoder-decoder 仍可解码 (训练时用 mask 见过)
# 时间: ~30 s per drop config (不需要重训)
```

→ 这一招能把 drop 实验时间从 **3 × ~30 min 重训** 降到 **5 × ~30 s**, 大幅降低实验成本。

## 工程依赖

| 依赖 | 是否已有 | 处理 |
|------|----------|------|
| beam search 库 | ❓ 需 check | 自己写一个快速版 (10ms/iter should be enough) |
| group_by_correlation | ✅ 自实现 | 复用 task18_q11 |
| TIGER mask inference | ❓ 需 check | 若无, 改 `predict_dataloader_config` 加 mask 选项 |
| flan-t5-xl embedding | ✅ task1 | 复用 |
| Group A SID tensor | ✅ task17 | 复用 |
| Drop testing | ✅ task15_eval | 加 --drop-layer {L1,L2,L3,dedup,none} |

## 与主线 #1, #2 的关系

AQ 是**主线 #1 之外的另一个独立 axis (结构 vs 信息)**:
- 主线 #1 (task21): 信息集中 L1 是否独立于 ReSID (诊断本身)
- HRQ (task22): 双曲几何是否解释层级 (改几何)
- AQ (task23): 加性结构是否消首层独裁 (改结构)

三者可同时推进。**AQ + HRQ** 同时 pass → 强 evidence "first-layer dictatorship 是结构伪影"

## 论文故事线 (提前)

> "RQ 残差量化受 first-layer dictatorship 影响：Δ_1/Δ_2 = 7-8，drop L1 后 R@10 跌 70%。我们通过加性量化 (AQ) 验证这是结构伪影：在 AQ 上 Δ_1/Δ_2 < 2（pass），drop L1 仅跌 33%（pass）。这表明 RQ 的 L1 主导源于**嵌套结构**而非几何或信息瓶颈。HRQ (双曲几何 axis) 和 AQ (加性结构 axis) 在独立维度上各 pass kill 线，证明 first-layer dictatorship 是结构可消除的工程伪影。"
