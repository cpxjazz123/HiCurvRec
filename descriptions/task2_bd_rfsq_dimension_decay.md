# Task 14 (BD-RFSQ)：残差逐维度衰减速率是否均匀

> **状态**: 🟡 backlog

> **目的**：检验 RQ-VAE / RKMeans / RVQ 三层残差量化中，**不同维度上的衰减速度是否一致**——这是 BD-RFSQ（Block-wise Decaying Residual Finite Scalar Quantization）论文的核心假设之一。
> **数据集**：Amazon Toys（11924 商品，embedding_dim=2048 from flan-t5-xl）
> **目标仓库**：`/fs04/ar57/wenyu/GeneRec/GRID`
> **环境**：`conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys`

---

## 总目标

量化算法（RQ-VAE / RKMeans / RVQ）在逐层量化残差时，**假设所有维度按相同速率衰减**。如果某些维度已经收敛（衰减比例 ≈ 0）而其他维度还有大量信息（衰减比例 ≈ 0.5），则：
- **均匀情形**：用统一阈值或同一衰减 schedule 处理所有维度是合理的，BD-RFSQ 的"分块处理"没有额外收益。
- **不均匀情形**：分维度 / 分块施加不同量化强度有收益，BD-RFSQ 假设成立。

**关键判断**：残差向量在 D 维空间里，每一维独立走自己的衰减路径？还是所有维度"齐步走"？

---

## 具体做法

### 输入
- `embedding` (N=11924, D=2048)：全体商品的 LLM embedding
- 一个已训练好的 RQ-VAE / RKMeans / RVQ 模型（任选其一）：
  - **RQ-VAE**：`task.md` Stage 2 产物 `logs/inference/runs/2026-07-07/20-17-35/pickle/merged_predictions_tensor.pt`（无中间残差，需重训 RQ-VAE 来 dump 各层残差）
  - **RKMeans**：`task11_stage2_*` 产物；训练好的 layer-wise kmeans 模型可直接拿来做残差计算（推荐，因为 RKMeans ckpt 包含 centroids）
  - **RVQ**：`task2_rvq_pipeline` 产物

### 计算步骤
1. 用训练好的量化器对全体商品 embedding (N, D) 做逐层量化：
   - 第 1 层：`r1 = x - nearest_centroid(x)`   → shape (N, D)
   - 第 2 层：`r2 = r1 - nearest_centroid(r1)` → shape (N, D)
   - 第 3 层：`r3 = r2 - nearest_centroid(r2)` → shape (N, D)
2. 对每一层残差矩阵 `r_l ∈ R^(N, D)`，按列求绝对值均值：
   - `m_l[i] = mean_j |r_l[j, i]|`，长度 D 的向量
3. 对每一维 i 计算衰减比例：
   - `decay[i] = m3[i] / m1[i]`
4. 输出：
   - `decay` 长度 D=2048 的 numpy 向量
   - 直方图（横轴 decay 值，纵轴维度数量）
   - 摘要统计：mean, std, min, max, percentiles [10, 25, 50, 75, 90]

### 关键判断

| 分布形态 | 含义 | BD-RFSQ 价值 |
|---------|------|--------------|
| 集中在窄区间（如 [0.05, 0.15]） | 所有维度衰减均匀 | **低**：分块处理没必要，统一量化已足够 |
| 高度分散（如 min=0.02, max=0.5） | 维度间衰减差异大 | **高**：可针对慢收敛维度加强量化（BD-RFSQ 核心动机） |
| 双峰分布 | 部分维度已饱和，其他还在工作 | **中-高**：存在"早停 vs 继续"的两类维度 |

**判断标准**：如果 `std(decay) / mean(decay)`（变异系数 CV）> 0.5 → 衰减不均匀；如果 CV < 0.2 → 均匀。

---

## 实现要点（GRID 代码定位）

`src/models/modules/quantization/` 下应该有以下类：
- `RQVAE.forward()` 内部分层 quantize，返回 `quantized, residual_layers`（含每层残差）
- `RKMeans.quantize(x)` 接收输入，返回 `(quantized, indices, residuals)` 或类似

读模型 ckpt → 加载 codebook → 对全体商品 embedding 做逐层 quantize → dump `r1.pt`, `r2.pt`, `r3.pt` 到 `task_artifacts/results/exp14/`。

### 备用实现（如果 GRID API 不直接给残差）
对 layer-wise kmeans：
```python
# x: (N, D) embedding
# codebooks: list of L tensors, each (W_l, D)
r = x.clone()
residuals = []
for cb in codebooks:
    idx = nearest_neighbor(r, cb)         # (N,)
    quantized = cb[idx]                   # (N, D)
    r = r - quantized
    residuals.append(r.detach().cpu())    # (N, D)
# residuals = [r1, r2, r3]
```

---

## 完成指标

| 指标 | 目标 | 验证 |
|------|------|------|
| 三层残差矩阵 shape | (11924, 2048) | `torch.load` |
| decay 向量长度 | 2048 | 验证 |
| 直方图 PNG | 1 张 | `ls *.png` |
| CV (变异系数) | 任意值，作为后续 BD-RFSQ 决策依据 | JSON 报告 |

---

## 风险与回退

1. **RQ-VAE ckpt 没有直存 codebook** → 改用 RKMeans 或 RVQ（这两者的 ckpt 含 centroids vector，更直接）
2. **embedding_dim 不等于 2048** → 用 `model.codebook_dim`（GRID 内部维度，可能比输入维度小）
3. **直方图差异不显著** → 用 log-scale 直方图，或绘制 sorted decay 曲线（更易看出尾部维度）

---

## 输出示例（伪数据示意）

```python
# decay vector (D=2048)
{
  "mean": 0.10, "std": 0.08, "min": 0.001, "max": 0.45,
  "p10": 0.02, "p50": 0.08, "p90": 0.20,
  "cv": 0.80  # 高 → 不均匀 → BD-RFSQ 假设成立
}
```

---

## 与 BD-RFSQ 论文的关联

BD-RFSQ（Block-wise Decaying RFSQ）主张：
- 把 D 维分成多个 block
- 不同 block 用不同的步长 / 衰减 schedule
- 慢收敛 block 多量化、快收敛 block 早停

**本实验是 BD-RFSQ 假设的"先验可行性检查"**：如果 Toys 数据上 decay 已经均匀，BD-RFSQ 收益就小；如果不均匀，BD-RFSQ 才有意义。

---

## ✅ 完成报告（2026-07-10 loop tick）

### v1 (normalize=True, RKMeans 5 层)
- **产物**：
  - `/home/wlia0047/ar57/wenyu/GeneRec/task12_results/task12_decay.npy` (2048,)
  - `/home/wlia0047/ar57/wenyu/GeneRec/task12_results/task12_per_layer_means.npy` (5, 2048)
  - `/home/wlia0047/ar57/wenyu/GeneRec/task12_results/task12_decay_report.json`
  - `/home/wlia0047/ar57/wenyu/GeneRec/task12_v2_results/task12_decay_histogram.png`（v1+v2 同图）
- **结果**：CV = 0.051, mean = 4.04, std = 0.21（normalize 让 |r| ∈ [0,1] 制造了高 decay 值）
- **判断**：CV=0.051 < 0.2 → **衰减均匀** → BD-RFSQ 分块处理收益有限

### v2 (normalize=False, RKMeans 5 层)
- **产物**：
  - `/home/wlia0047/ar57/wenyu/GeneRec/task12_v2_results/task12_v2_decay.npy` (2048,)
  - `/home/wlia0047/ar57/wenyu/GeneRec/task12_v2_results/task12_v2_per_layer_means.npy` (5, 2048)
  - `/home/wlia0047/ar57/wenyu/GeneRec/task12_v2_results/task12_v2_decay_report.json`
- **训练产物**：`/fs04/ar57/wenyu/GeneRec/GRID/logs/train/runs/task12_v2_norm_false/checkpoints/checkpoint_000_005000.ckpt`
- **结果**：CV = 0.046, mean = 0.75, std = 0.034（真实几何，残差 norm 0.22-0.29）
- **判断**：CV=0.046 < 0.2 → **衰减均匀** → BD-RFSQ 收益有理论依据但 Toys 上几乎为零
- **v1 vs v2 解读**：CV 变化不显著（0.051 → 0.046）→ **衰减模式对 normalize 不敏感**；之前的"×4 衰减"是 normalize 的几何伪影

### 完成判定
- ✅ 三层/五层残差矩阵 shape (11924, 2048) — 验证通过
- ✅ decay 向量长度 2048 — 验证通过
- ✅ 直方图 PNG（v1+v2 比较图）— `task12_decay_histogram.png`
- ✅ CV 已输出（任意值，作为决策依据）— 决策：**BD-RFSQ 在 Toys 上收益有限**
- **结论**：衰减高度均匀（CV ≈ 0.05 无论 normalize 与否），BD-RFSQ 分维度/分块调度在 Toys 数据上没有强动机。