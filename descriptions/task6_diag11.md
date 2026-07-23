# Task 16 (Diag11 批 1+2 合并): 量化器 7 个诊断量 —— 判 B / C / D / E / A

> **合并来源**: task16_diag11_batch1.md (量 1-4) + task17_diag11_batch2.md (量 5-7)
> **合并日期**: 2026-07-18
> **合并原因**: 同一 Diag11 批研究问题 (量化器结构诊断), 7 个量互相依赖, 合并消除重复说明

> **目的**：在 GRID 残差上**一次遍历即可**测出 7 个量化器诊断量，每个独立支持一个 idea 的判决。
> 数据集：Amazon Toys（11924 商品，2048-dim flan-t5-xl embedding，已训好的 RKMeans 3 层 + RQ-VAE 3 层各一个 ckpt 可用）
> 目标仓库：`/fs04/ar57/wenyu/GeneRec/GRID`
> 环境：`conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys`

---

## 总目标

B / C / D / E / A 五个 idea 都要求对**残差的内部结构**说清一句话。本批 7 个量就是这句话的物证：

| 量 | 测什么 | 怎么算 | 判哪个 idea |
|----|--------|--------|------------|
| **1. 幅度剖面 μ_l / σ_l** | 残差逐层"变多小" | `μ_l = E‖r_l‖`、`σ_l = Std‖r_l‖` over (N, ) | **地基 / Control** —— 后面所有归一化分母 |
| **2. 谱有效维度 erank_l** | 残差真正铺在几个方向 | `erank_l = exp(H(λ̂))` 或 participation ratio | **C / E（坍缩）** —— 主判据 |
| **3. 水位有效维度 m_l** | 假设高斯/MSE 时码率能容纳几个方向 | reverse water-filling 解 `θ_l`，数 `λ_{l,i} > θ_l` | **B（缓刑）** —— 若全局缩放不改变 m_l |
| **4. Gain-Shape 误差分解 f_radial + cell 内幅度 CV** | 每个量化误差有多少来自幅度错 | `f_radial = Σ e_r² / Σ‖r−q‖²`，cell 内 ‖r‖ CV | **B** —— `f_radial` 大 + cell CV 高 → 单独分离 gain 才有用 |
| **5. 前缀重建误差 D_l** | 只用前 l 个 token 能还原 `z` 多准 | `D_l = E‖z − ẑ_l‖²`，`ẑ_l = Σ_{j≤l} q_j` | **D 原料** |
| **6. 精化效率 η_l** | 这一层每多花 1 bit，误差降多少 | `η_l = (D_{l-1} − D_l) / log₂ K_l` | **D** —— `η_l→0` = 精化失败 |
| **7. 形状非平稳性 SW(r̂_l, r̂_{l+1})** | 扣幅度后,相邻层归一化残差在球面上的距离 | `r̂_l = r_l / ‖r_l‖`，sliced-Wasserstein | **A** —— `SW>0` = 层轴上形状非平稳 |

> 统一记号（贯穿 3 批 11 个量）：
> - `r_l`：进入第 `l` 层、量化前的残差（`r_1 = x`）
> - `q_l`：第 `l` 层量化输出
> - `r_{l+1} = r_l - q_l`
> - `K_l`：第 `l` 层码本大小
> - `N`：商品数（11924）
> - `D`：embedding 维度（2048）

---

## 量 1：幅度剖面 `μ_l` / `σ_l`

**测什么**：逐层测残差的平均长度 `μ_l = E_i ‖r_l[i]‖` 与波动 `σ_l = Std_i ‖r_l[i]‖`。

**怎么算**：
```python
x = torch.load(embedding_path)              # (N, D)
r = x.clone()
mus, sigmas = [], []
for l in range(L):
    c = codebooks[l]                        # (K_l, D)
    idx = nearest(r, c)
    q = c[idx]                              # (N, D)
    r_norm = r.norm(dim=-1)                 # (N,)
    mus.append(r_norm.mean().item())
    sigmas.append(r_norm.std().item())
    r = r - q
```

**判什么 idea**：**本身是 platitude**（逐层变小谁都知道），作用是当 **control** —— 后面几乎所有量都要用它归一化，把"幅度在变"和"结构在变"分开。

**完成指标**：
- `mus_l, sigmas_l` 长度 3（layer 数）数组
- 图：`||r_l||` mean ± std 折线
- 输出：`task16_layer_norm_profile.json`

---

## 量 2：谱有效维度 `erank_l`

**测什么**：每层残差协方差 `Σ_l = Cov(r_l)` 真正铺开在几个方向上 —— 非参数、不依赖分布假设。

**怎么算**：
```python
# 对每层 r_l (N, D):
cov = (r_l - r_l.mean(0)).T @ (r_l - r_l.mean(0)) / (N - 1)   # (D, D)
eigvals = torch.linalg.eigvalsh(cov)                          # (D,) 升序
eigvals = eigvals.clamp(min=1e-12)
p = eigvals / eigvals.sum()
H = -(p * p.log()).sum()
erank = H.exp().item()
# 或 participation ratio: erank = (sum eigvals)^2 / sum(eigvals^2)
```

**判 C / E（维度坍缩）**：`erank_l` 随 `l` 急剧下降就是坍缩证据。

**为什么是主判据**：不依赖任何分布假设（既不需要高斯、也不需要 MSE），鲁棒反映"残差结构坍缩了几维"。

**完成指标**：
- `erank_l` 长度 3 数组
- 图：`erank_l` vs `l` 柱状
- 输出：`task16_erank_per_layer.json`

---

## 量 3：水位有效维度 `m_l`

**测什么**：在给定码率 `R_l = log₂ K_l` 下，反解 water-filling 得到水位 `θ_l`，数 `λ_{l,i} > θ_l` 的方向有几个 —— 即"码率能容纳几个独立方向"。

**怎么算**（reverse water-filling）：
```python
# eigvals 升序 (D,)
# R = log2(K_l)
# capacity M = D 维总方差预算 = sum eigvals
# 解 theta 使得 sum_{i: eigvals[i] > theta} (eigvals[i] - theta) = R * ln(2) / N ... (细节见笔记)
# m_l = #{i : eigvals[i] > theta_l}
```

**判 B（缓刑）**：若 `m_l` 在深层 **不减少** —— 说明 ReSID/Gain-Shape 的"信息仅来自形状" 是成立的，全局缩放确实不应让 `m_l` 改变；这给 B **缓刑**（不是无罪释放，但有理由继续审）。

**与量 2 的关系**：假设高斯 / MSE，与 `erank_l` 交叉验证。两者不一致时以 `erank_l` 为准，`m_l` 当辅助读数。

**完成指标**：
- `m_l, theta_l` 长度 3 数组
- 图：`m_l` 与 `erank_l` 双柱
- 输出：`task16_water_filling_m.json`

---

## 量 4：Gain-Shape 误差分解 `f_radial` + cell 内幅度 CV

**测什么**：每个量化误差 `r − q` 中，"幅度错"占比多少；以及同一码字 cell 内样本的幅度变异系数（CV）。

**怎么算**：
```python
# 对每层 r_l (N, D):
#   q = c[idx] (N, D)
e = r - q                          # (N, D)
e_radial = (e.norm(dim=-1))        # just magnitude of error vector
# 切向/径向正交分解:
e_norm = e_radial
r_norm = r.norm(dim=-1)
q_norm = q.norm(dim=-1)
# radial error = |r_norm - q_norm|
e_r = (r_norm - q_norm).abs()
f_radial_l = (e_r ** 2).sum() / (e ** 2).sum(dim=-1).sum()

# cell 内样本幅度 CV:
import numpy as np
cv_per_cell = []
for k in unique(idx):
    cluster_r_norms = r_norm[idx == k]
    if len(cluster_r_norms) > 1 and cluster_r_norms.std() > 0:
        cv_per_cell.append(cluster_r_norms.std() / cluster_r_norms.mean())
mean_cell_cv_l = np.mean(cv_per_cell)
```

**判 B（Gain-Shape）**：只有当**深层 `f_radial` 大 AND cell 内 CV 高**（码本被迫花码字去编幅度）时，单独分离 gain 才买得到东西；**否则 B 判死**。

**完成指标**：
- 长度 3 数组：`f_radial_l`、`mean_cell_cv_l`
- 图：`f_radial_l` vs `l` + cell CV 分桶直方图
- 输出：`task16_gain_shape_decomposition.json`

---

## 量 5：前缀重建误差 `D_l`

**测什么**：只给前 `l` 个 token，把 `z` 还原得多准。

**怎么算**：
```python
x = torch.load(embedding_path)               # (N, D)
r = x.clone()
prefix_q = torch.zeros_like(x)
Ds = []
for l in range(L):
    c = codebooks[l]
    idx = nearest(r, c)
    q = c[idx]                                # (N, D)
    prefix_q = prefix_q + q                   # cumulative reconstruction
    D_l = ((x - prefix_q) ** 2).sum(dim=-1).mean().item()
    Ds.append(D_l)
    r = r - q
```

**判 D 原料**：理想情况下 `D_l` 应随 `l` 平滑单调下降；而不是"前几层烂到底、最后一层暴降"。前者才是"逐级可精化"，后者是"前几层是凑数、只有最后一层真在做事"。

**完成指标**：
- `D_l` 长度 3 数组（如果用 4 层模型则长度 4）
- `D_l` 曲线 + 跨算法（A vs B vs C 三组）对比图
- 输出：`task16_prefix_reconstruction.json`

---

## 量 6：精化效率 `η_l`

**测什么**：这一层每多花 1 bit，误差降了多少。

**怎么算**：
```python
Ds = ...  # from 量 5
etas = []
for l in range(1, L):
    delta_D = Ds[l-1] - Ds[l]
    R_l = np.log2(Ks[l])                          # bits spent at layer l
    eta_l = delta_D / R_l
    etas.append(eta_l)
```

**判 D**：
- `η_l→0`：这一层输出了 token 却几乎没提供新信息，是"逐级精化失败"的直接量化。
- `η_l` 跨层递减不显著 → D **判死**。

**完成指标**：
- `η_l` 长度 `L-1` 数组
- 图：`η_l` vs `l` 柱状
- 输出：`task16_refinement_efficiency.json`

---

## 量 7：形状非平稳性 `SW(r̂_l, r̂_{l+1})`

**测什么**：把每层残差归一化到单位球 `r̂_l = r_l / ‖r_l‖`，再算相邻层归一化残差在球面上的 sliced-Wasserstein 距离。把幅度信息扣掉后看"形状是否还变"。

**怎么算**（`sliced_wasserstein`）：
```python
import torch
import numpy as np

def sliced_wasserstein(X, Y, n_proj=256):
    """Sliced Wasserstein distance between point clouds X, Y (N, D).
    Projects both onto random directions, sorts, and averages W1 on each."""
    d = X.shape[-1]
    directions = torch.randn(n_proj, d, device=X.device)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    # Project (N, n_proj)
    proj_X = (X.unsqueeze(1) * directions.unsqueeze(0)).sum(-1)  # (N, n_proj)
    proj_Y = (Y.unsqueeze(1) * directions.unsqueeze(0)).sum(-1)
    proj_X, _ = proj_X.sort(dim=0)
    proj_Y, _ = proj_Y.sort(dim=0)
    N = min(proj_X.shape[0], proj_Y.shape[0])
    w1_per_dir = (proj_X[:N] - proj_Y[:N]).abs().mean(dim=0)
    return w1_per_dir.mean().item()

r_norm = []
r = x.clone()
for l in range(L):
    c = codebooks[l]
    idx = nearest(r, c)
    q = c[idx]
    r_hat = r / r.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    r_norm.append(r_hat)
    r = r - q

sws = []
for l in range(L - 1):
    sw = sliced_wasserstein(r_norm[l], r_norm[l+1])
    sws.append(sw)
```

**判 A（形状非平稳）**：
- `SW` 显著 > 0 才有"层轴上的形状非平稳"证据（区别于 NSVQ 的"训练时间轴"非平稳）。
- 注意：`SW` 本身只看"分布变了"，不直接等于"该变对量化有坏处" —— 量 11 提供"matters"证据。

**完成指标**：
- `SW_l` 长度 `L-1` 数组
- 图：`SW_l` vs `l` 折线
- 输出：`task16_sliced_wasserstein_shape.json`

---

## 关键判定（结合量 1-7）

| 量 | 判 idea X | 判据 |
|----|-----------|------|
| 量 1 | **Control** | 地基, 后面所有归一化分母 |
| 量 2 | **C / E（坍缩）** | erank 急剧下降 → 坍缩 |
| 量 3 | **B（缓刑）** | m_l 在深层不减少 → B 有理由继续审 |
| 量 4 | **B（Alive）** | f_radial 大 + cell CV 高 → 单独提增益价值 |
| 量 5 + 6 | **D（逐级精化）** | `η_l` 在深层 ≈ 0 → D 死；η 平稳 → D 活 |
| 量 7 | **A（层轴非平稳）** | `SW_l` 显著 > 0 → 有"形状非平稳"现象 |
| 量 7 + 量 11 | **A 的"matters"** | 量 7 看"变了"，量 11 看"变了有用" |

---

## 输出文件汇总

| 图 | x 轴 | y 轴 | 文件 |
|----|------|------|------|
| 1. 幅度剖面 | layer l | `μ_l` mean + `σ_l` band | `task16_layer_norm_profile.png` |
| 2. erank + m_l 双柱 | layer l | `erank_l`、`m_l` | `task16_effective_rank.png` |
| 3. f_radial + cell CV | layer l | 两条线 | `task16_gain_shape_decomp.png` |
| 4. 前缀重建 | layer l | `D_l` | `task16_prefix_reconstruction.png` |
| 5. 精化效率 | layer l | `η_l` | `task16_refinement_efficiency.png` |
| 6. 形状非平稳 | layer l | `SW_l` | `task16_sliced_wasserstein.png` |

合并 JSON:
```json
{
  "task": "task16_diag11",
  "groups": [
    {"layer": 1, "mu": 0.236, "sigma": 0.012, "erank": 1234, "m_l": 800, "f_radial": 0.05, "mean_cell_cv": 0.04, "D_l": 0.246, "eta_l": null, "SW_l": null},
    ...
  ]
}
```

---

## 完成指标（一遍过）

| 指标 | 目标 |
|------|------|
| 7 个 JSON 输出 | 文件存在 |
| 6 张 PNG 图 | 文件存在 |
| 对 Toy 数据每量输出一句话判据 | `task16_verdict.md` |
| 跑完一轮 RKMeans + 一轮 RQ-VAE | 两种量化器各出一遍数据 |

---

## 风险与回退

| 风险 | 回退方案 |
|------|----------|
| 单层 N=11924 算 `Cov` 内存爆 | 改用 sketch SVD 或 mini-batch estimation |
| `erank_l` 数值不稳定（接近奇异矩阵） | 加 `eigvalsh` `eps=1e-12`，用 `participation_ratio` 做对照 |
| `theta_l` water-filling 在小 `R_l` 时无解 | 退化为 `m_l = R_l / log2(D)` 估计上限 |
| `D_l` 单调不下降 → η_l 为负 | 把 `η_l = max(0, eta_l)` 截断（只关心"有没有贡献"） |
| `SW` 计算慢（每对 11924 样本随机投 256 方向） | 随机降采样到 2000 样本 |
| 量 7 在 Toy 上 `SW ≈ 0`（分布平稳） | 这本身就是对 A idea 的强证据，**不需要回退**——记录结论 |

---

## 与下游的衔接

- 量 1 是所有归一化的分母。
- 量 2 是判 C/E 主判据；量 3 是判 B 的对冲读数。
- 量 4 是判 B 的"具体有几条可走通"。
- 量 5+6 判 D 逐级精化；量 7 判 A 形状非平稳。
- 这 7 个量跑完一遍，B/C/D/E/A 五个 idea 的判决就有**关键证据**。
- 缺：判 A 的"matters"（量 11）。

---

## 输出示例（伪数据示意）

```json
{
  "task": "task16_diag11",
  "data": "Toys RKMeans 3-layer",
  "layers": [
    {"l": 1, "mu": 0.236, "sigma": 0.012, "erank": 1234, "m_l": 800,  "f_radial": 0.05, "cell_cv": 0.04, "D_l": 0.246, "eta_l": null, "SW_l": null},
    {"l": 2, "mu": 0.218, "sigma": 0.011, "erank": 1180, "m_l": 750,  "f_radial": 0.07, "cell_cv": 0.05, "D_l": 0.058, "eta_l": 0.123, "SW_l": 0.014},
    {"l": 3, "mu": 0.207, "sigma": 0.011, "erank": 1100, "m_l": 690,  "f_radial": 0.09, "cell_cv": 0.06, "D_l": 0.011, "eta_l": 0.031, "SW_l": 0.012}
  ],
  "verdict": {
    "C_E_collapse": "NO (erank drops <5% over 3 layers)",
    "B_reprieve_m_l": "m_l decreasing slowly → direction-dominant → B 有理由继续审",
    "B_alive_f_radial": "f_radial < 0.10 at all layers → cell 内幅度 CV 很小 → B 判缓刑，单独提增益价值有限",
    "D_alive": "PARTIAL: η_2 << η_1 → 第 3 层精化能力骤降，D idea 仅部分成立",
    "A_sw_evidence": "SW=0.014, 0.012 → 微弱但非零；层间形状不平稳为弱信号"
  }
}
```

---

## 执行顺序

1. **加载 Toy embedding** `(11924, 2048)`
2. **加载 RKMeans 3 层码本**（从 `task11_stage2_*` 或 task15_group_a_s21 ckpt）
3. **一次前向**得到 `r_l`、`q_l`、`idx_l` for `l=1,2,3`
4. **量 1-7** 各算一遍
5. 出 JSON + PNG
6. 再跑一遍 RQ-VAE 同样七量（如果时间允许）

总预计耗时：**单算法 < 10 min**（无 GPU），单算法 + 出图 < 15 min。

---

## 合并说明

| 原 task | 文件 | 状态 |
|---------|------|------|
| task16_diag11_batch1.md | task16 量 1-4 | 已合并 |
| task17_diag11_batch2.md | task16 量 5-7 | 已合并, 文件删除 |