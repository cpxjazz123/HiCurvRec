# INFRA — Issue #237

| 需求 | 命令 | 输出摘要 | sha256 | 判定 |
|---|---|---|---|---|
| TIGER 原始 Embedding | `np.load(...sentence_t5.npy)` | (24,587, 768) float32 | `6490c71753952b2d8788bb7e4e52cff6a5b0b59819a2cfbd737bee7bc07a77fb` | PASS |
| 冻结行为图 | `np.load(behavior_graph/graph.npz)` | 4,319,971 字节 | `6dbac4545f23779c26f95e3f128daeb916079f53fad7ec159dd8bfb08e7121e8` | PASS |
| 图节点/边 | CSR `len(indptr)-1` / `nnz//2` | 24,587 / 315,391 | — | PASS |
| 自环/对称 | `diagonal().sum()` / `(A != A.T).nnz` | 0 / 0 | — | PASS |
| 构图源仅训练划分 | `metadata.policy.directed_source` | `train.parquet` | `80597e38…` | PASS |
| 边划分可用 | `data.split_edges` | train 264,929 / val 25,231 / test 25,231 | — | PASS |
| ego 指示矩阵 | `data.ego_indicator` | 2-hop 1,556,959 nnz;3-hop 3,140,946 nnz | — | PASS |
| 失真对与图距离 | `data.graph_distances` | 20,000 对,全部有限 | — | PASS |
| 解释器 | `genrec_env/bin/python3.10` | python 3.10.20 | — | PASS |
| torch / CUDA | `torch.__version__` | 2.13.0+cu130,CUDA 可用 | — | PASS |
| **geoopt** | `import geoopt` | **缺失** → 自建流形运算,已声明 | — | 见下 |
| torch_geometric | `import torch_geometric` | 2.8.0(未使用,手写稀疏传播) | — | PASS |
| 硬件 | `nvidia-smi` | 4×A100-40GB(训练用其中 1 张) | — | PASS |

## geoopt 缺失的处理

官方 GraphMoRE 从 `geoopt.Stereographic` 取全部流形算术。本 issue 在
`graphmore/manifold.py` 中按 κ-stereographic 定义实现同一套运算,并用 43 项单元测试锁定:

- `expmap0`/`logmap0` 互逆;`dist` 对称、对角为零、满足三角不等式;
- 负曲率距离的**闭合公式**(`acosh`)与 `mobius_add` 形式互相印证;
- κ→0 退化为 `2‖x−y‖`;
- `proju` 把点严格压入球内(球面专家不炸);
- 可学习曲率**不跨越零**。

差异仅在 `mobius_matvec` 用切空间线性形式(而非 geoopt 的闭式)与优化器用 Adam
(而非 `RiemannianAdam`),两者都在 REPORT §3 与 VERDICT 中声明。
