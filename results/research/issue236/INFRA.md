# INFRA — Issue #236

| 需求 | 命令 | 输出摘要 | sha256 | 判定 |
|---|---|---|---|---|
| 冻结图存在 | `ls -la results/research/behavior_graph/graph.npz` | 4,319,971 bytes | `6dbac4545f23779c26f95e3f128daeb916079f53fad7ec159dd8bfb08e7121e8` | PASS |
| 图哈希匹配 issue 声明 | `sha256sum` | 与 issue C1 完全一致 | 同上 | PASS |
| 节点数 | CSR `len(indptr)-1` | 24,587 | — | PASS |
| 主分量节点数 | BFS 分量分解 | 24,556(99.874%) | — | PASS |
| 孤立节点 | 度==0 计数 | 31(31 个单点分量) | — | PASS |
| 边数 | `nnz//2` | 315,391 | — | PASS |
| 无自环 | `A.diagonal().sum()` | 0 | — | PASS |
| 对称 | `(A != A.T).nnz` | 0 | — | PASS |
| 构图源仅训练划分 | `metadata.policy.directed_source` | `results/stage0_build_parquet/train.parquet` | `80597e38…` | PASS |
| #235 结果可读 | `ls results/research/issue235/summary.json` | 存在 | — | PASS |
| #235 geodesic 三元组 | `summary.json.geodesic.triples` | 40,566,929 | — | PASS |
| #235 neighbour 三元组 | `summary.json.neighbour.triples` | 42,206,243 | — | PASS |
| #235 无三元组商品 | `per_node_summary.json.items_without_triples` | 87 | — | PASS |
| 解释器 | `genrec_env_v2/bin/python3.9` | numpy / scipy 可用 | — | PASS |
| 内存容量 | `free -g` | 1511 GB 总,1463 GB 可用 | — | PASS |
| 硬件 | `nvidia-smi` | 4×A100-40GB(本 issue 不用 GPU) | — | PASS |

**结论:全部 PASS,无 FAIL。** 唯一风险是计算规模(见下)。

## 计算规模评估

| 项 | 规模 | 内存(流式,不存稠密) |
|---|---|---|
| 参考节点数 | 24,556 | — |
| 每参考点一次 BFS | 24,556 × O(V+E) | 单次距离向量 24,556×int16 = 49 KB |
| 累计平方距离和 | 3 个数组 × 24,587 × int64 | 590 KB |
| 三元组数(geodesic) | 40,566,929 | 索引数组 ×3 × int64 = 974 MB |
| 三元组数(neighbour) | 42,206,243 | 同上 |

**关键:** 流式累计只需 3 个长度为 V 的 int64 数组(每个存 Σ d(a,v)²),不需要稠密距离矩阵
(那会是 24,556² × 2 B = 1.2 GB,虽可放下但违反 C12 的实现约束)。

BFS 24,556 次的墙钟时间待实测;若单次约 3 ms 则总计约 75 秒,可接受。
