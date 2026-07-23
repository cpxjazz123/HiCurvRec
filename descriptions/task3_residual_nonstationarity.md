# Task 15 (非平稳性)：训练过程中残差分布是否随时间漂移

> **状态**: 🟡 backlog

> **目的**：检验 RQ-VAE / RKMeans / RVQ 训练过程中，**残差分布是否存在随时间的系统性漂移**——这是 Gain-Shape RQ / Online RQ 论文的核心动机之一（"残差分布非平稳，所以静态 codebook 不够好"）。
> **数据集**：Amazon Toys（11924 商品，11924 训练 embedding；测试取固定 1024 商品）
> **目标仓库**：`/fs04/ar57/wenyu/GeneRec/GRID`
> **环境**：`conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys`

---

## 总目标

如果训练过程中**残差分布保持平稳**（不同 step 的统计特征相似），那么：
- 训完一次 codebook 就够了，**静态量化**已经足够
- Gain-Shape RQ / Online RQ 的"动态更新 codebook"收益小

如果残差分布**随训练时间漂移**（后期 step 的残差范数变小、方向偏移），那么：
- 早期训的 codebook 不能捕捉后期分布
- **动态 codebook / 残差预测**有收益
- 这种漂移在深层（layer 3）应该比浅层（layer 1）更明显（残差本身信息更少、更敏感）

---

## 具体做法

### 输入
- 多个训练 checkpoint（layer-wise 训练的中间产物）：
  - **RQ-VAE**：暂未保留中间 ckpt（task.md Stage 2.1 只训了 1 个完整 RQ-VAE），需要重训并加 ckpt callback
  - **RKMeans**：`logs/train/runs/2026-07-10/12-49-50/checkpoints/checkpoint_000_001000.ckpt` (layer 0 训完), `_002000.ckpt` (layer 1 训完), `_003000.ckpt` (layer 2 训完), `_004000.ckpt` (layer 3 训完), `_005000.ckpt` (layer 4 训完)
  - **RVQ**：类似 RKMeans，逐层训练保留中间 ckpt
- 固定测试商品 batch：取 validation 集前 1024 个商品的 LLM embedding（避免训练集数据泄漏）
- 每一层 ckpt 包含该层 codebook 的 centroids

### 计算步骤
1. 加载 step k 的 ckpt（layer 0..L-1 共 L 个 codebook）
2. 对固定测试商品 (1024, D) embedding 做逐层量化，得到各层残差：
   - `r1_k, r2_k, ..., rL_k`，每个 shape (1024, D)
3. 对每一层 l 的残差 `r_l_k`，计算分布特征：
   - `mean_l_k` (D-vec)：残差按列均值
   - `norm_l_k`（标量）：所有残差范数的均值 `mean_i ||r_l_k[i]||`
   - `norm_std_l_k`（标量）：残差范数的标准差
   - 可选：`cov_l_k`（D×D）：残差的协方差矩阵
4. 对每个 k ∈ {1000, 2000, 3000, ...} 计算上述特征 → 时间序列
5. 比较不同 k 之间的变化：
   - **绝对漂移**：`||mean_l_k - mean_l_k'||` (Frobenius)
   - **相对漂移**：`||mean_l_k - mean_l_k'|| / mean(||r_l_k'||)`（除以当前残差大小）
6. 对**第 1 层**和**第 3 层**都做同样分析作为对照

### 关键判断

| 模式 | 含义 | Gain-Shape / Online RQ 价值 |
|------|------|-----------------------------|
| 不同 step 统计特征几乎相同 | 残差分布平稳 | **低**：静态 codebook 已足够 |
| 残差范数单调下降（如 0.5 → 0.3 → 0.1） | 训练后期残差自然衰减（这是正常的） | 需用相对漂移排除数学效应 |
| 残差**方向**持续偏移（如 cov 主成分方向转 30°） | 分布形状在变 | **中**：可能需要重训 |
| 深层漂移 >> 浅层漂移（相对值） | "深层非平稳性更严重" 假设成立 | **高**：值得继续深挖 |

**判断标准**（同时满足以下三点才算"非平稳"）：
1. `相对漂移(第3层) > 相对漂移(第1层)` × 2 倍
2. 漂移趋势是**单调**的（不是 step 间的随机噪声）
3. 残差方向有偏移（不仅是范数变小，因为范数变小是数学效应）

### 对照：相对漂移排除数学效应

绝对漂移：`abs_drift = ||μ_k' - μ_k||`（两个 step 的残差均值之差）
当前残差大小：`cur_scale = mean_i ||r_l_k'[i]||`

**相对漂移** = `abs_drift / cur_scale`

如果 `abs_drift` 看起来很大但 `rel_drift` 不大 → 是残差本身在变小，不是分布真在变。
如果 `rel_drift` 在深层显著大于浅层 → 深层非平稳性更严重。

---

## 实现要点（GRID 代码定位）

`src/models/modules/quantization/rkmeans.py` 应有：
```python
class RKMeans:
    def __init__(self, num_codebooks=3, codebook_size=256, dim=2048):
        self.codebooks = nn.ParameterList([nn.Parameter(...) for _ in range(num_codebooks)])
```

加载不同 step 的 ckpt：
```python
ckpt = torch.load(f"checkpoint_000_00{k}.ckpt")
state = ckpt["state_dict"]
# 不同层在 state 里可能前缀不同（如 "quantization.0.codebooks.0.weight"）
```

提取 codebooks 后做前向：
```python
x_test = ...  # (1024, D)
r = x_test.clone()
for l in range(L):
    cb = codebooks[l]  # (W, D)
    idx = ((r.unsqueeze(1) - cb.unsqueeze(0)).pow(2).sum(-1)).argmin(dim=1)
    q = cb[idx]
    r = r - q
    residuals_l.append(r.detach().cpu())
```

---

## 完成指标

| 指标 | 目标 | 验证 |
|------|------|------|
| 至少 3 个 step 的中间 ckpt | layer-wise 训练 step 1000/2000/3000 | `ls checkpoints/` |
| 每层每 step 的残差 (1024, D) | dumped | `ls *.pt` |
| 漂移时间序列图（绝对 + 相对） | 1 张 PNG | `ls *.png` |
| 深层/浅层漂移比 | 输出 JSON | 验证 |

---

## 风险与回退

1. **RKMeans 没有足够多中间 ckpt** → 改用 RVQ（task2 也保留了 layer-wise ckpt）
2. **ckpt 中 codebook 的 key prefix 不明确** → 用 `ckpt["state_dict"].keys()` dump 出来手动定位
3. **方向漂移不明显** → 主成分分析（PCA）残差分布，看第一主成分方向在不同 step 的变化

---

## 与 Gain-Shape RQ / Online RQ 的关联

- **Gain-Shape RQ** 假设：深层残差的方向比大小更稳定（这跟实验 15 + 实验 16 都相关）
- **Online RQ** 假设：训练过程中残差分布漂移，所以需要周期性更新 codebook

**本实验是 Online RQ 的"先验可行性检查"**：如果 Toys 数据上残差分布平稳，Online RQ 收益小；如果漂移明显且深层更严重，值得做。

---

## ✅ 完成报告（2026-07-10 loop tick）

### v1 (3 个 ckpt: step 3000/4000/5000 跨 layer 完成点)
- **产物**：
  - `/home/wlia0047/ar57/wenyu/GeneRec/task13_results/task13_drift_report.json`
- **做法**：跨层完成点（layer 0/1/2 训完时）取 ckpt，对固定 2000 商品子集算各层残差
- **结果**：每层残差 norm 在 3 个 step 间变化 1-3%（< 5%），无强漂移信号
- **判断**：跨 step 漂移极小，但 v1 只看 step 3000/4000/5000 三个稀疏点，看不到层**内部**漂移

### v2 (17 个密集 ckpt: step 100-1600, every 100)
- **产物**：
  - `/home/wlia0047/ar57/wenyu/GeneRec/task13_v2_results/task13_v2_drift_report.json`
  - `/home/wlia0047/ar57/wenyu/GeneRec/task13_v2_results/task13_drift_curves.png`
- **训练产物**：`/fs04/ar57/wenyu/GeneRec/GRID/logs/train/runs/task13_v2_dense_ckpts_v2/checkpoints/` (17 ckpts + last.ckpt)
- **结果**：
  - layer 0: norm 0.236 → 0.235 (总 shift 0.28%, **0 步内稳定**)
  - layer 1: norm 1.000 → 0.870 (总 shift 13.0%, **100 步内稳定**)
  - layer 2: norm 1.000 → 0.921 (总 shift 7.9%, **100 步内稳定**)
  - layer 3: norm 1.000 → 0.939 (总 shift 6.1%, **100 步内稳定**)
- **判断**：新层内部训练早期漂移真实存在但收敛极快（100 步）——**与 Gain-Shape/Online RQ 假设"残差分布漂移严重"不吻合**

### v1 + v2 综合结论
- v1 看到的"max shift"是 1000 步跨层累计（normalize=True 制造的层间 norm 差异）
- v2 揭示：单层内部实际**前 100 步就稳定**，后续 300 步几乎不变
- → 对早停 schedule 有直接应用价值（每层 100-200 步足够，800-1000 步浪费 80%+ GPU 时间）
- → Gain-Shape / Online RQ 在 Toys 数据上"周期性更新 codebook"收益可能很小（每层早期就稳定）

### 完成判定
- ✅ ≥ 3 个 step 中间 ckpt（v1: 3 个, v2: 17 个）— 验证通过
- ✅ 每层每 step 残差 (1024, D) 已 dump — `task13_v2_results/`
- ✅ 漂移曲线 PNG（双子图）— `task13_drift_curves.png`
- ✅ 深层/浅层漂移比（绝对 + 相对）已输出
- **结论**：Toys 上残差训练非平稳性微弱，新层 100 步内收敛，Online RQ 在 Toys 上无强动机。