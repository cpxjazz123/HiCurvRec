# Issue #243 — 数学与数据契约

冻结于 2026-11-23，先于任何生产代码。实现只能服从本文件；确需偏离时先改本文件并记录日期与理由。

曲率符号约定：`k` 为截面曲率，模型为 κ-stereographic 空间
`M_k^d = {x ∈ R^d : −k‖x‖² < 1}`，即 `‖x‖ < 1/√|k|`（k>0 为球面型，k<0 为 Poincaré 球，k=0 为欧氏）。

## 0. 论文与官方实现对照

论文：GraphMoRE，AAAI-2025，arXiv:2412.11085。官方代码：`RingBDStack/GraphMoRE`
(`models.py`, `backbone.py`；依赖 `geoopt==0.5.0` 的 `Stereographic` 流形)。

| 模块 | 论文位置 | 官方实现 | 本项目取舍 |
|---|---|---|---|
| κ-stereographic 映射 | §4.1 Eq.1–2 引 Bachmann et al. | `geoopt.Stereographic.expmap0/logmap0/proju/mobius_matvec/mobius_add` | 用同一组闭式公式自实现（float64 参考 + float32 生产），并用 `geoopt` 数值交叉验证；不引入 geoopt 运行时依赖 |
| κ-GCN 层 | Eq.1–2 | `kappaGCNConv`：`mobius_matvec(W,x)` → `logmap0` → GCN 归一化聚合（`deg^{-1/2}` 双侧）→ `expmap0(project=True)` | 同结构；入边/出边各一套权重（有向适配），聚合在切空间做，投影用 `proju` |
| 专家 | §4.1 | `Encoder`：`proju(origin,x)` → `expmap0` → 2 × `kappaGCNConv`；每专家独立曲率 | 同；曲率取 {-3,-1,0,+1,+3}，`learnable` 记入配置 |
| 专家拼接 | Eq.8 | 各专家输出直接 concat，再 `LayerNorm(n_factors*out_dim)` | 同（concat 后 LayerNorm），**不做**“欧氏加权拼接” |
| 门控 | Eq.5–6 | `Gating`：每尺度用**欧氏** `GCNConv` 编码 ego 子图 → `global_mean_pool` → 尺度间 concat → `Linear(out_dim*len(hops), num_experts)` → softmax，温度 1.0 | 同结构；`T_v` 由双尺度 ego 子图给出（issue 指定 1/2 跳，官方默认 [2,3]，属参数差异） |
| 节点混合 | Eq.8 | `w ⊗_k x = expmap0(w · logmap0(x))` | 同 |
| 对齐权重 | Eq.9 | `softmax(W_u * W_v)` | 同。**注**：这是 *pair* 权重，不是节点权重；#242 把 pair 权重的集中度当成节点门控塌缩来解读，本文件明确区分 |
| 距离 | Eq.10 | `dis = Σ_e W_uv^e · d_e²`，`d_e` 为各专家流形距离 | 同 |
| 失真损失 | Eq.7 | 官方**只在边上**算：`|(d²/g)² − 1|` 直接对 `dis=d²` 与 `g`，`distortion=|dis/g − 1|` | 本项目按 issue 在**距离分桶抽样**的有限非自身对上算，`g` 取可见图无向投影最短路；公式与官方等价（`dis` 即 `d²`） |
| ego 采样 | Eq.3–4 | `nx.ego_graph(G, node, radius=k_hop)`，图按**无向** `nx.Graph()` 构建 | 本项目保留**有向**诱导子图（issue 明确要求方向）；先定节点集再取全部诱导边（官方 `ego_graph` 也是诱导边，但其图无向） |
| 最短路径 | Eq.7 的 `g` | 预计算 `dis_shortest` 矩阵（唯一边用 g=1） | 本项目对抽样根跑 BFS（scipy，无权），仅用可见图 |

## 1. 流形算子（模块 `manifold.py`）

### 1.1 投影 `proju(x, k, eps=1e-3)`
- k>0：`maxnorm = (1-eps)/√k`；若 `‖x‖ > maxnorm`，缩放至 `maxnorm`。
- k<0：`maxnorm = (1-eps)/√|k|`；同上。
- k=0：恒等。
- 边界：`x = 0` 返回 `0`（原点合法）。**禁止**用 `normalize` 把零向量变成非零；零向量的处理是“保持在原点”，不是“投到球面”。

### 1.2 指数映射 `expmap0(x, k)`
`exp_0^k(x) = tan_k(‖x‖) · x/‖x‖`，其中
- k>0：`tan_k(v) = tan(√k v)/√k`
- k<0：`tan_k(v) = tanh(√|k| v)/√|k|`
- k=0：`tan_k(v) = v`
- `‖x‖ = 0` 时返回 `x`（即 0）。

### 1.3 对数映射 `logmap0(y, k)`
`log_0^k(y) = tan^{-1}_k(‖y‖) · y/‖y‖`，其中
- k>0：`tan^{-1}_k(u) = arctan(√k u)/√k`
- k<0：`tan^{-1}_k(u) = artanh(√|k| u)/√|k|`，要求 `√|k| u < 1`（先 `proju`）
- k=0：`tan^{-1}_k(u) = u`
- `‖y‖ = 0` 时返回 `y`。

### 1.4 Möbius 加法 `mobius_add(x, y, k)`
```
den = 1 - 2k⟨x,y⟩ + k²‖x‖²‖y‖²
num = (1 - 2k⟨x,y⟩ - k‖y‖²) x + (1 + k‖x‖²) y
x ⊕_k y = num / den
```
`|den| < 1e-9` 时抛错（契约不允许静默产出 Inf）。

### 1.5 Möbius 矩阵向量乘 `mobius_matvec(M, x, k)`
`M ⊗_k x = expmap0(M · logmap0(x), k)`，输出经 `proju`。

### 1.6 距离 `dist(x, y, k)`（统一式，官方口径）
```
d_k(x,y) = 2/√|k| · tan^{-1}_k( √|k| · ‖(−x) ⊕_k y‖ )        (k ≠ 0)
d_0(x,y) = ‖x − y‖
```
其中 `tan^{-1}_k` 同 1.3。**等价的闭式（仅 k<0）**：
```
d = (1/√|k|) · arcosh(1 + 2|k|‖x−y‖² / ((1 − |k|‖x‖²)(1 − |k|‖y‖²)))
```
两者必须在 float64 下一致到 1e-9（这是参考实现的自校验）。

**#242 的错误**：使用 `(2/√c)·arcosh(...)`，比正确值大 2 倍；且球面用 `normalize` + `arccos(k⟨u,v⟩)`，
该 `arccos` 式只对半径恰为 `1/√k` 的**球面上的点**成立，而 `expmap0` 产生的是球内点。

### 1.7 已知答案（写进 `tests/test_known_answers.py`）
| 输入 | 期望 |
|---|---|
| 任意 k，`dist(x,x,k)` | `0`（±1e-9） |
| k=−1，x=0，y=0.5·e1 | `2·artanh(0.5) = 1.0986122886681098` |
| k=−1，x=0，y=r·e1，r→1⁻ | `2·artanh(r)`，随 r 单调增，r=0.9 → 2.9444389791664403 |
| k=+1，x=0，y=r·e1，r=0.5 | `2·arctan(0.5) = 0.9272952180016122` |
| k=+1，球面上两点夹角 θ | `θ`（半径 1）：`θ = arccos(⟨u,v⟩/R²)` |
| k=0，x, y | `‖x−y‖` |
| 对称性 | `dist(x,y)=dist(y,x)` 到 1e-9 |
| 方向反转 | 交换入/出边后 `T` 仍为 128 维，且方向池化项发生变化 |

## 2. κ-GCN 层（模块 `kappa_gcn.py`）

对节点 u，第 l 层：
```
m_u = Σ_{v∈N_in(u)} c_uv · W_in · logmap0(h_v)  +  Σ_{v∈N_out(u)} c_uv · W_out · logmap0(h_v)
a_u = m_u / (|N_in|+|N_out|)                     # 归一化，空邻域得 0
h_u' = expmap0( W_self · logmap0(h_u) + a_u + b, k )
```
- `c_uv = (deg_u · deg_v)^{-1/2}` 可选；本项目固定为**均值归约**（`1/(|N_in|+|N_out|)`），与官方
  `deg^{-1/2}` 双侧归一化不同，故在契约里固定为均值，并在对照表注明差异。
- 有向适配：入/出各一组权重；k=0 时退化为普通有向 GNN（此时 `logmap0`/`expmap0` 为恒等）。
- 空邻域：`a_u = 0`，不产生 NaN。

## 3. 数据契约

| 项 | 值 | 来源 |
|---|---|---|
| 训练图 | `results/research/issue242/graph/ego_csr.npz`（与 #242 同一份，sha256 `b17c2754…`） | 复用并重核 |
| 商品映射 | stage-0 item id，无重映射 | `graph/metadata.json` |
| 语义特征 | `stage1_GeneEmbedding/output/sentence_t5.npy`，768 维，sha256 `6490c717…` | 只读 |
| 合法划分 | 51,695 可见用户 / 5,744 评估用户；目标边 29,536；评估边 32,729 | `graph/metadata.json` |
| 隔离 | 评估标签不进训练图、标定、采样；训练图距离只用可见图 | 断言测试 |
| 禁止输入 | 任何未来交互、商品标签、可学习 ID 表 | 代码审查 + 测试 |

## 4. 训练配置（先冻结，不扫参）

| 项 | 值 |
|---|---|
| 专家初始曲率 | {−3, −1, 0, +1, +3}，`learnable=False`（记录可学习性为后续项） |
| 专家输出维 | 16 / 专家，合计 80 |
| `T_v` 维度 | 128 = 2 尺度 ×（root + mean_in + mean_out + mean_all）× 16 |
| 门控 | `Linear(128→64) → ReLU → Linear(64→5) → softmax`，温度 1.0 |
| 距离监督 | `L_D = mean |(D_geo/g)² − 1|`，`D_geo² = Σ_e W_uv^e d_e²`，`W_uv = softmax(W_u ⊙ W_v)` |
| 采样 | 根按 out-degree 五分位均匀；伙伴按 hop∈{1,2,3,4} 均匀后再均匀；seed 242001 |
| 训练步数 / batch | 400 步 / 192 对（按墙钟预算预先固定，不按观测指标调整） |
| 优化器 / lr | AdamW / 1e-3，clip 1.0 |

## 5. 验收口径（开训前冻结）

- **主口径**：独立评估对（eval 用户转移、不在可见图中）上 `Spearman(D_geo, g)` 必须**同时**高于
  两个非训练对照：原始 768 维语义欧氏距离、Jaccard 邻域重叠（真实算交集/并集）。
- **次口径**：`|(D_geo/g)²−1|` 的均值与中位数，按距离桶与度数五分位分组。
- **重叠子图口径**（步骤 4）：`E_t = mean_{i≠j∈t} |(D(i,j)/g(i,j))² − 1|`；
  多子图商品报平均 `E`、最差子图 `E`、p90、同时保真比例；同时保真阈值预先定为 `E_t ≤ 0.5`。
- **失败判定**：常数距离、门控塌缩、或独立评估不优于两个对照任一 → 按 §6.1 报告，不改判据。

## 6. 参考实现与生产代码的边界

- `reference.py` 只用 numpy，float64，**不 import 生产模块**；提供 1.1–1.6 的独立实现与暴力版本。
- 测试的期望值来自本文件的已知答案表，**不得**用生产函数的输出生成期望。
- 变异测试（`MUTATIONS.md`）至少覆盖：距离系数 ×2、删掉一条边、交换入/出方向、常数输出、
  切断梯度、错误的指标（如用 Pearson 代替 Spearman）。

---

## 7. 修订记录（2026-11-23，#244 触发）

### 7.1 正曲率的距离公式：原诊断部分错误，已更正

**原判断（错）**：`2·tan_k^{-1}(‖(−x)⊕_k y‖)` 是 Poincaré 专用公式，不能用于正曲率。

**实测**：在坐标域不受限时，该式与「逆球极投影到 S^{d+1} 再求夹角」的定义路线在
k = 0.5 / 1.0 / 3.0 的随机点上一致到 **1e-15**。它是 κ-stereographic 的统一公式，对 k≠0 全部成立。

**#243 的真实缺陷（三处，均已有对应的变异测试）**：

1. `proju` 对 **k>0 也套用了双曲半径截断**。单位球面 `x=0.9, y=-0.9` 的 Möbius 和范数是 9.474，
   被截到 0.9945，于是 `2·arctan(0.9945) = 1.5653 ≈ π/2`；正确值是 `4·atan(0.9) = 2.93126`。
   这是「π/2 常数」的**唯一**来源。
2. `expmap0` 在 k>0 用了 `tan(√k‖v‖)/√k`，正确是 `R·tan(‖v‖/(2R))`；
   `logmap0` 又漏了半径因子，正确是 `2R·arctan(r/R)`。两者必须互为逆，且 `d(0, expmap0(v)) = ‖v‖`。
3. 参考实现的 `stereo_to_sphere` 把「单位球的球极逆映射」整体乘以 R，而 R 球的坐标要嵌进构造里
   （`X = 2R²u/(R²+|u|²)`）。这个错误由「独立路线」检查出来，不是靠同公式互证。

**修正后的口径**：

| 量 | 定义 |
|---|---|
| 坐标域 k>0 | 全平面（极点位于无穷远），只做 1e6/√k 的数值防护，**不得**用双曲半径截断 |
| 坐标域 k<0 | 开球半径 (1−1e-3)/√|k| |
| expmap0 k>0 | `R·tan(‖v‖/(2R)) · v/‖v‖`，R = 1/√k |
| logmap0 k>0 | `2R·arctan(‖u‖/R) · u/‖u‖` |
| dist k>0 | 生产：闭式 `R·arccos((4R²⟨u,v⟩+(|u|²−R²)(|v|²−R²))/((R²+|u|²)(R²+|v|²)))`；参考：逆球极投影 + 3D 夹角 |
| dist k<0 | `2·tan_k^{-1}(‖(−x)⊕_k y‖)`（标准 Poincaré，与 k>0 同一族公式） |
| dist k=0 | `‖u−v‖` |

**k=0 不是 k→0 的连续极限**：标准 Poincaré 度量 `ds = 2/(1−|k|‖x‖²)|dx|` 的平坦极限是 **2×** 欧氏距离。
本项目把 k=0 当作独立空间（与 geoopt 一致），该 2 倍差写入 §7.2，并在
`test_euclidean_is_the_documented_flat_limit` 里固定下来，避免它悄悄漂移。

### 7.2 跨曲率比较的单位

混合距离 `D_geo² = Σ_e W_e d_e²` 会把不同空间的距离放进同一个求和。由于上述归一化，
双曲距离在「单位速度」口径下是欧氏/球面的 **2 倍**。所以 §3 的专家尺度标定必须显式吸收这个因子，
并在 `CONTRACT.md` 里写明标定后的单位；不得把「门控均衡」当作「距离贡献均衡」。

### 7.3 变异清单的更正

「把球面闭式换成 Möbius 式」**不是有效变异**（两者等价），已替换为「距离减半」。
有效的球面变异是 `sphere-proju-hyperbolic-clamp`（复现 #243 的真实缺陷）与
`sphere-logmap-drops-radius`。
