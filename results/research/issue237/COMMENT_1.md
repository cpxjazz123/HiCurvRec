## 验收清单冻结(29 条)、训练判定与 infra

| 项 | 值 |
|---|---|
| 正文 sha256 | `8baddc18bf23a92bbf18736b84c435812ab5d4ce9c61253aa8e47da5db0265d4` |
| 正文长度 | 2,112 字符 |
| 清单条数 | **29 条(C1–C29)** |
| 清单文件 | `results/research/issue237/RUN_LOG.md` |

### 是否需要训练:**需要**

引 issue 原文:

> 本轮允许训练 GraphMoRE 表示学习模型,仅交付 RQ-VAE 前的增强 Embedding。

验收指标(held-out 行为边 AUC/AP、图距离失真、语义保真)**只能由新权重产生**。
训练的是 Stage1 表示学习模型,**不是** Stage2/Stage3,也**不重训 TIGER**。

### infra 核查

| 项 | 值 | 判定 |
|---|---|---|
| TIGER 原始 Embedding | `stage1_GeneEmbedding/output/sentence_t5.npy`,**(24,587, 768) float32** | PASS |
| 行为图 | `results/research/behavior_graph/graph.npz`,sha `6dbac454…` | PASS |
| 图节点/边 | 24,587 / 315,391;主分量 24,556;31 孤立点 | PASS |
| 构图源 | 仅 `train.parquet` | PASS |
| 解释器 | `genrec_env/bin/python3.10`,python 3.10.20 | PASS |
| torch | 2.13.0+cu130,CUDA 可用,4×A100-40GB | PASS |
| **geoopt** | **缺失** | 见下 |
| torch_geometric | 2.8.0 可用 | PASS |
| scipy | 1.15.3 | PASS |

### 一个必须提前声明的实现决定(C8)

**`geoopt` 未安装**,而官方 `models.py` 依赖 `geoopt.Stereographic`(Möbius 运算、指数/对数映射、
可学习曲率)。因此本实现**自行编写 κ-stereographic 流形运算**,并在 REPORT 里逐项对照官方代码:

| 官方(`geoopt.Stereographic`) | 本实现 |
|---|---|
| `manifold.expmap0` / `logmap0` / `proju` | 按 Bachmann et al. (2020) κ-stereographic 公式实现 |
| `manifold.mobius_matvec`(几何闭式) | `expmap0(W @ logmap0(x))`(切空间线性 + 回流形) |
| `manifold.mobius_add` | 按 κ-stereographic Möbius 加法公式实现 |
| `RiemannianAdam` | 普通 Adam(参数本身在切空间参数化) |
| `Stereographic(k, learnable=True)` | 曲率用 `±softplus(raw)` 约束,初值 {-3,-1,+1,+3},0 冻结 |

**关键:κ-GCN 的聚合是流形上的。** 官方 `kappaGCNConv` 的做法是
`logmap0 → GCN 归一化和 → expmap0`,本实现完全照此,**不是**欧氏加权拼接。
差异只在 `mobius_matvec` 的闭式与切空间形式之别,以及优化器,均会写进 REPORT。

### 下一步

§2 先写流形单元测试(红):曲率符号、`logmap0∘expmap0` 往返、距离对称性与三角不等式、
κ→0 的欧氏极限、Möbius 加法与官方公式一致。
