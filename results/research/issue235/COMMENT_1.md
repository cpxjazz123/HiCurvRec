## 验收清单冻结(26 条)与训练判定

| 项 | 值 |
|---|---|
| 正文 sha256 | `8206b40fbedbf78dc1dc16c2e7d5e228c086ce9003057dcb74d9b5bccfa08b79` |
| 正文长度 | 1,678 字符(冻结后未变) |
| 清单条数 | **26 条(C1–C26)** |
| 清单文件 | `results/research/issue235/RUN_LOG.md` |

### 是否需要训练:**不需要**

引 issue 原文两处:

> 先完成图分析,**不训练模型**,不预设混合曲率一定更优。

> 不修改 RQ-VAE、原始 768 维输入、三层 SID 或任何训练框架;**不启动 Stage2/Stage3,不开额外 Issue**。

验收指标(曲率直方图、商品级曲率、Graphlet 关联)全部可由**已有冻结图**复算,不需要新权重。

### infra:数据已就位

issue 要求「复用当前 Graphlet 实验使用的冻结训练集行为图和商品 ID 映射」。仓库只有**一份**无向行为图
(CLAUDE.md §89b 契约),Stage2 的 A1–A6 结构监督正是读它:

| 项 | 值 |
|---|---|
| 路径 | `results/research/behavior_graph/graph.npz` |
| sha256 | `6dbac4545f23779c26f95e3f128daeb916079f53fad7ec159dd8bfb08e7121e8` |
| 构图源 | `results/stage0_build_parquet/train.parquet`(sha `80597e38…`),**仅训练划分** |
| 定义 | `E = {{i_t, i_{t+1}} | (i_1..i_T) ∈ S_train}`,无向、去重 |
| 节点 / 边 | 24,587 / 315,391 |
| 孤立点 | 31 |
| 有向转移 | 323,633(去重)/ 339,519(总数) |
| 双向边 | 8,242 |

**该图已经是 issue 要求的"无向简单投影"**(`metadata.policy.single_undirected_graph: true`,
无自环,已合并重复边),因此无需重建,主分析直接读它,原图不改。

有向原始数据从 `train.parquet` 的有序序列统计,只用于 C5 的"原始有向图"计数。

### 下一步

§2 实现:先写合成图(路径/树/闭合环/欧氏格点)的公式对照测试(红),再写主分析。
