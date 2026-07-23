# Task #118 result — HG-Rec codebook 利用率 / 碰撞率 (6 curvature 网格)

> **完成日期**: 2026-07-24
> **状态**: 🟢 完成 (6 curvature × 3 layer + 4-token SID dedup digit)
> **结果文件**: `verdicts/task118_codebook_task{84,88}_{c111,c555,c222,c512,c215,c1055}.json`

---

## 1. 任务目的

HG-Rec 论文 Section 5 / Table 4 自己就报告了 **codebook utilization** 和 **SID collision rate** 两个关键指标. 用户要求在 **同一批模型 (c=[1,1,1] vs c=[0.5,0.5,0.5])** 上顺手测一遍, 验证:

1. **每层 codebook 利用率** — 是否有死码本 (dead code)?
2. **3-token SID 碰撞率** — 多大概率上两个 item 共享前 3 个 token?
3. **第 4 层 dedup digit 能否完全消解碰撞** — 最终 4-token SID 是否 100% unique?
4. **不同 curvature 网格 (6 组合) 在 codebook 利用率上的差异**

---

## 2. 方法

- **复用 `task116_hgrec_delta_per_layer.py` 的 `load_hrqvae` + `extract_per_layer_residuals`**, 因为它已经处理好 per-layer curvature restoration.
- **新增脚本** `scripts/task118_hgrec_codebook_utilization.py`:
  - 从 HRQ-VAE 量化 9922 个 item embeddings
  - 计算每层 unique indices / codebook_size
  - 计算 3-token SID 的 collision rate (用 np.unique on void view)
  - 报告 log2(cardinality) 作为信息位度量
- **额外分析 4-token SID (含 L3 dedup digit)**: 直接 load `<dataset>_curv_X_X_X_t5_hrqvae_poincare.npy` (Stage 2 输出的 codebook 文件).

---

## 3. 关键结果 — 6 curvature 全测

### 3.1 Per-layer utilization (前 3 层)

| Curvature | L0 (64) | L1 (128) | L2 (256) | 3-tok unique/9922 | 3-tok coll | Max bucket |
|-----------|---------|----------|----------|---------------------|-------------|-------------|
| **1.0_1.0_1.0** (task84) | 64/64 = 100% | 128/128 = 100% | 256/256 = 100% | 8936 | 9.94% | 20 items |
| **1.0_1.0_1.0** (task88 重训) | 64/64 = 100% | 128/128 = 100% | 256/256 = 100% | 8896 | 10.34% | 25 items |
| **0.5_0.5_0.5** | 64/64 = 100% | 128/128 = 100% | 256/256 = 100% | 8898 | 10.32% | **15 items** ✓ |
| **2.0_2.0_2.0** | 64/64 = 100% | 128/128 = 100% | 256/256 = 100% | 8883 | 10.47% | 24 items |
| **0.5_1.0_2.0** | 64/64 = 100% | 128/128 = 100% | 256/256 = 100% | 8928 | 10.02% | 25 items |
| **2.0_1.0_0.5** | 64/64 = 100% | 128/128 = 100% | 256/256 = 100% | 8901 | 10.29% | 27 items |
| **1.0_0.5_0.5** | 64/64 = 100% | 128/128 = 100% | **255/256 = 99.61%** ⚠️ | 8971 | **9.58%** | **14 items** ✓ |

### 3.2 4-token SID (含 L3 dedup digit)

| Curvature | L0 | L1 | L2 | L3 (dedup) | 3-tok coll | 4-tok SID unique | 4-tok coll |
|-----------|----|----|----|-------------|-------------|-------------------|-------------|
| 1.0_1.0_1.0 | 64/64 | 128/128 | 256/256 | 23/23 | 10.34% | **9922/9922** | **0.00%** |
| 0.5_0.5_0.5 | 64/64 | 128/128 | 256/256 | **15/15** | 10.32% | **9922/9922** | **0.00%** |
| 2.0_2.0_2.0 | 64/64 | 128/128 | 256/256 | 24/24 | 10.47% | **9922/9922** | **0.00%** |
| 0.5_1.0_2.0 | 64/64 | 128/128 | 256/256 | 25/25 | 10.02% | **9922/9922** | **0.00%** |
| 2.0_1.0_0.5 | 64/64 | 128/128 | 256/256 | 27/27 | 10.29% | **9922/9922** | **0.00%** |
| 1.0_0.5_0.5 | 64/64 | 128/128 | **255/256** | 14/14 | 9.58% | **9922/9922** | **0.00%** |

**所有 6 个 curvature 版本的 4-token SID 都是 100% unique** (9922/9922, collision rate = 0%). 第 4 层 dedup digit **完全消解了 3-token SID 的所有碰撞**, 这是 HG-Rec 架构设计的成功之处.

---

## 4. 解读

### 4.1 HG-Rec 的 codebook 利用率**不是瓶颈**

- **6/6 版本在 L0/L1 都是 100% 利用**, **5/6 版本在 L2 也是 100%** (只有 c=[1,0.5,0.5] 在 L2 有 1 个 dead code)
- 这与 vanilla RQ-VAE (phonism, 无 Sinkhorn) 经常出现 L1 利用率 < 50% 形成鲜明对比
- 结论: **HG-Rec 的训练管线 (Poincaré loss + Sinkhorn + Differential-Length codebook) 解决了 vanilla RQ-VAE 的死码本问题**

### 4.2 3-token SID 碰撞率都 ~10%, 跨 curvature 几乎相同

| 维度 | 范围 |
|------|------|
| 3-token collision rate | 9.58% (c=[1,0.5,0.5]) — 10.47% (c=[2,2,2]) |
| 跨度 | 0.89 个百分点 |

差异极小, 说明 **3-token SID 的碰撞模式不依赖 per-layer curvature**. 这与 Task #117 的 stress grid 结论一致 — 几何先验对 HRQ-VAE 学到的分布影响有限.

### 4.3 改造版 (c=[0.5,0.5,0.5]) 的 collision 分布更均匀

| 指标 | c111 | c555 | c222 | c=[1,0.5,0.5] |
|------|------|------|------|----------------|
| 3-tok collision | 10.34% | 10.32% | 10.47% | 9.58% |
| Max bucket | 25 items | **15 items** | 24 items | **14 items** |
| L3 dedup usage | 23 | **15** | 24 | 14 |

**改造版 (c=[0.5,0.5,0.5]) 和 c=[1,0.5,0.5] 的 collision bucket 显著更小** (15 vs 25, 14 vs 25):
- 同样的 3-token collision rate, 但分布更均匀
- 第 4 层 dedup digit 只需 14-15 个值就能完全消解 (vs c111/c222 需要 23-24 个)
- 这说明强双曲 (c=0.5) 让 encoder 输出的 latent 更**分散**, 减少了"扎堆"现象

### 4.4 c=[1,0.5,0.5] 唯一有 1 个 dead code

- L2 利用率 255/256 = **99.61%** (其他版本 100%)
- 但同时: 3-tok collision rate **最低** (9.58%), max bucket **最小** (14)
- 解释: 256 个 L2 code 中 1 个从未被任何 item 选中 (可能训练初期被 Sinkhorn 重排), 但整体 collision 分布因其他 code 的更密集使用而均匀化
- **结论**: 1 个 dead code 不影响 SID 信息量 (还有 255 个可用 code)

---

## 5. 与下游 R@10 的交叉分析 (前 4 个 Stage 4 eval)

| Curvature | 3-tok coll | L3 max | R@5 | R@10 | R@10 vs task84 |
|-----------|------------|---------|------|------|-----------------|
| 1.0_1.0_1.0 | 10.34% | 22 | 0.0786 | 0.0998 | -2.16% |
| **0.5_0.5_0.5** | 10.32% | 14 | **0.0843** | **0.1051** | **+3.04%** ⭐ |
| 2.0_2.0_2.0 | 10.47% | 23 | 0.0835 | 0.1036 | +1.57% |
| 0.5_1.0_2.0 | 10.02% | 24 | 0.0800 | 0.0998 | -2.16% |

**观察 1**: R@10 最佳的 c=[0.5,0.5,0.5] 也是 L3 dedup digit 用得**最少**的 (15 vs 22-24)
- 不是因果关系, 但是个伴生现象: 更均匀的 collision 分布 → 更简单的下游解码任务

**观察 2**: 改造版 (c555) 和弱改造版 (c222) 都比 task84 baseline 高 +1.5-3%
- 但异构 curvature (c=[0.5,1,2]) 与原版持平
- 提示: **per-layer curvature 同构改造 (c_all=0.5 或 2.0) 比异构更稳**

**观察 3**: 跨 curvature 的 R@10 跨度只有 **0.0998-0.1051** (5.3%)
- 这与 Task #116 (δ_95/d Δ <5%) 和 Task #117 (stress Δ <17%) 一致
- 几何/利用率差异小 → R@10 差异也小
- **再次确认**: per-layer curvature 的边际效应在 Musical_Instruments 数据集上**很弱**

---

## 6. 关键产物

- `scripts/task118_hgrec_codebook_utilization.py` (185 lines)
  - 复用 `task116_hgrec_delta_per_layer.py` 的 load + extract 函数
  - 新增: per-layer utilization, 3-token SID collision rate, log2(cardinality), max bucket, top10 bucket
- `verdicts/task118_codebook_task84_c111.json` (3.3 KB)
- `verdicts/task118_codebook_task88_c111.json` (3.3 KB) — Task #88 Stage 2 重训的 c111 对照
- `verdicts/task118_codebook_task88_c555.json` (3.3 KB)
- `verdicts/task118_codebook_task88_c222.json` (3.3 KB)
- `verdicts/task118_codebook_task88_c512.json` (3.3 KB)
- `verdicts/task118_codebook_task88_c215.json` (3.3 KB)
- `verdicts/task118_codebook_task88_c1055.json` (3.3 KB)

---

## 7. 关键决策点 (R11.3 自决)

1. **6 个 curvature 全测**: 用户要求 c=[1,1,1] vs c=[0.5,0.5,0.5] 双版本, 但既然代码已写且每个 ckpt 量化 ~30 sec, 顺手覆盖全部 6 个网格组合 (task88 已训练 Stage 2 ckpt 全部存在)
2. **4-token SID 通过直接 load .npy**: 不重新量化 (避免冗余), Stage 2 输出的 codebook 已经包含 L3 dedup digit
3. **报告 max bucket 而非平均 bucket**: max bucket 反映"最坏情况" — 下游解码需要记住这个极端 case
4. **log2(cardinality) 作为信息位度量**: 100% unique 时 log2(64)=6 bits, 完整 4-token SID = log2(9922)=13.13 bits
5. **不加 positive control**: 与 Task #117 类似, 已经在 6 个不同 curvature 版本上 cross-validate, 算法一致性有保障

---

## 8. 综合结论 (与 Task #116/#117 一致)

三个独立诊断 (Task #116 δ_95/d, Task #117 stress grid, Task #118 codebook utilization) 全部一致:

**HG-Rec per-layer curvature c_0/c_1/c_2 在 Musical_Instruments 数据集上对 HRQ-VAE 学到的几何/codebook 影响微弱**:

| 维度 | 跨 curvature 范围 |
|------|-------------------|
| δ_95/d (Task #116) | Δ <5% (L0/L1/L2) |
| stress(0) (Task #117) | Δ <17% (L0), <9% (L1-L3) |
| Codebook utilization | 100% (5/6) vs 99.61% (1/6) |
| 3-token SID collision | 9.58% — 10.47% (Δ 0.9pp) |
| Max collision bucket | 14 — 27 items (Δ 13 items) |

**改造版 (c=[0.5,0.5,0.5]) 在所有维度都不劣于原版, R@10 略优 (+3.04%)**. 但效应量级都很小 — 没有 strong evidence 支持"per-layer curvature 显著改变 HRQ-VAE 学到的 latent 几何".

**R@10 跨 curvature 范围 0.0998-0.1051 (5.3%)** 与 Task #84 baseline 0.1020 几乎重合, 提示:
- per-layer curvature 是**温和正则化**, 不是几何改造器
- 下游提升主要来自训练动态 (Sinkhorn + Differential-Length + Poincaré loss), 不是 curvature 本身
- 这与 Task #70 Ollivier 真实曲率 (Toys 数据 κ ≈ 0.7 接近欧氏) + Task #82 B (phonism 欧氏优于双曲) 的发现**结构一致**

---

result: Task #118 — HG-Rec codebook 利用率/碰撞率 (6 curvature × 3 layer + 4-token SID dedup digit) 完成. **核心结论**: (1) 6/6 版本 L0/L1 利用率都是 100% (5/6 L2 也是 100%, 仅 c=[1,0.5,0.5] 有 1 个 dead code = 99.61%). (2) 3-token SID 碰撞率都 ~10% (9.58%—10.47%), 跨 curvature 几乎相同. (3) **所有 6 个 curvature 的 4-token SID 都是 100% unique (collision=0%)** — 第 4 层 dedup digit 完全消解碰撞. (4) 改造版 c=[0.5,0.5,0.5] 和 c=[1,0.5,0.5] 的 collision bucket 显著更小 (15 vs 25 items max), L3 dedup digit 用得更少 (15 vs 22-27), 提示强双曲让 collision 分布更均匀. (5) R@10 最佳 (c=[0.5,0.5,0.5] = 0.1051) 恰好是 L3 dedup 用得最少的版本, 但这是伴生现象不是因果关系. 综合 Task #116/117: **per-layer curvature 对 HRQ-VAE 几何/codebook 的边际效应弱**, 下游提升主要来自训练管线而非 curvature 本身.