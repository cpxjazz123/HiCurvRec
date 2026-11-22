## C7–C11：冻结 bank 的隔离审计（六个切分全为 0 对重叠）

对象：`results/research/issue230/banks/{chain,fork,merge}_{supervision,validation,test}.npy`，哈希在 `banks/banks.json`。

| 结构 | 监督∩验证 | 监督∩测试 | 验证∩测试 | 隔离 |
|---|---:|---:|---:|---|
| chain | 0 | 0 | 0 | ✅ |
| fork | 0 | 0 | 0 | ✅ |
| merge | 0 | 0 | 0 | ✅ |

**隔离方式**：无序对由**唯一一个切分拥有**（`graphlet_banks.split`）；一个实例只有在它的三对都归同一符号时才会被放入该切分，跨切分的实例被丢弃并记录原因（`banks.json` 的 `dropped` 与 `dropped_sample`）：

| 结构 | 采样 | 监督 | 验证 | 测试 | 丢弃 |
|---|---:|---:|---:|---:|---:|
| chain | 20,000 | 7,651 | 1,285 | 1,222 | 9,842 |
| fork | 20,000 | 6,854 | 820 | 788 | 11,538 |
| merge | 20,000 | 6,322 | 957 | 978 | 11,743 |

**数据来源**：仅 `results/research/behavior_graph_directed/graph.npz`，其 `metadata.json` 记录 `policy.source = results/stage0_build_parquet/train.parquet, training histories only`、`validation_test: not read`；train.parquet sha256 `80597e38…8815`。

**度数分布**（防止只抽 hub）：bahn 的节点 out-degree 中位数 11–14，图整体 13.163；out-degree 均值 22.8–32.5（高于整体，因为 fork/merge 的采样中心天然是高入/出度节点 —— 这一点在 §5 报告里必须与"hub 偏置"一起说明）。

**C6 运行时断言**：`graphlet_banks.assert_loss_isolated(kind, triples, evaluation)` 会在训练期对**实际进入 loss 的 pair（含负样本）**检查与评估 bank 的交集，非零即抛错。该函数已有单测（`tests/test_graphlet_banks.py`，7 passed）。

证据：`results/research/issue230/isolation.json`、`banks/banks.json`。
