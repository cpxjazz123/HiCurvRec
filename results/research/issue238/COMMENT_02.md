## 节点 2：infra 完成，无 FAIL

证据：`results/research/issue238/INFRA.md`（原始输出 `infra/files.txt`、`infra/env.txt`）。

### 数据与图（全部 PASS，哈希已记）

| 项 | sha256 前缀 |
|---|---|
| `stage0_build_parquet/train.parquet` | `80597e38d081029e…` |
| `stage1_GeneEmbedding/output/sentence_t5.npy` | `6490c71753952b2d…` |
| `behavior_graph/edges.parquet` | `57a4477331019274…` |
| `behavior_graph/graph.npz` | `6dbac4545f23779c…` |
| `behavior_graph/metadata.json` | `786a12813f89aad0…` |

`metadata.json` 的 `definition` 为 `E = {{i_t, i_{t+1}} | (i_1..i_T) in S_train}`，
`sources` 里的 train.parquet 哈希与上表一致 → **C5（行为图只来自 train）成立**。

### 冻结基线与对照（PASS）

- 基线 `train_rqvae.py` `a06d9f97…`、`model/layers.py` `0d5eacf6…`、`config/train_config.json` `27461832…`
- TIGER SID `025ad8bd…`、checkpoint `af8efc29…`、Stage3 入口 `7edf83f8…`
- 基线超参逐字记录在 INFRA.md 第 2 节，作为 C2 的比对基准

### 解释器与硬件（PASS）

- Stage0/1/3：`genrec_env/bin/python3.10`（torch 2.13.0+cu130）
- Stage2/机制测试：`genrec_env_v2/bin/python3.9`（torch 2.8.0+cu128）
- 4 × **L40S 46 GB**（不是 A100 40 GB，不影响协议）；端口 50201/50202 空闲

### 一处必须记录的差异

本机是 L40S 46GB，`CLAUDE.md` 里提到的是 A100 40GB。四卡/全局 batch 4096 的协议不变，
但显存上限不同，`INFRA.md` 已写明。

### 下一步

实现（TDD 先红后绿）：图曲率公式与三元组构造 → 32 维分块混合曲率距离 D_q
→ 曲率保真 L_curv（含冻结的尺度/权重标定规则）→ Sinkhorn 代价接线与 STE 梯度路由。

*（`results/research/issue238/COMMENT_02.md`）*
