## round 3 前置完成:框架层图感知 batch,覆盖率 70×

### 机制为什么值得再试一轮

上一轮我判 FAIL 时把原因归到"间歇监督",并准备停下来问。量完之后结论要更精确:

| 测量 | 值 | 含义 |
|---|---:|---|
| 结构项梯度 / 重构梯度 | **0.126** | 力度不是瓶颈 |
| 受约束对共享 L1 码 / 随机基线 | **19–25×** | 被推到的对**确实**在共享 |
| 每步覆盖的整三元组 | 104 / 20,827 | **只有 2% 被推到** |

所以机制在"它看到的对"上是有效的,限制在覆盖。

### 框架改动(§66 允许:框架属本仓库)

| 文件 | 改动 |
|---|---|
| `tools/research/stage2/graph_batches.py` | 新增。把 bank 实例的成员排到相邻位置,返回一个**全排列**(每行恰好一次,不重不漏) |
| `tools/research/stage2/launcher.py` | 启动时把 order 写入结果目录,经 `GENREC_GRAPH_BATCH_ORDER` 传给 trainer |
| `stage2_RQ-VAE/TIGER_graphlet_wide_l1_rq/train_rqvae.py` | 读到 order 就用它,并关掉 sampler 的二次 shuffle |

**只改排列,不改配方**:batch 大小、每 epoch 步数、seed、全局 batch 4096 全部不变。

### 实测覆盖

| 量 | 均匀抽样(round 2) | 图感知 order | 倍数 |
|---|---:|---:|---:|
| 整三元组 / batch | 104 | **7,279** | 70× |
| 成对约束 / batch | 766 | **16,187** | 21× |

### 测试与 smoke

| 项 | 结果 |
|---|---|
| `tools/research/stage2/tests/test_graph_batches.py` | 4 passed(先红:模块不存在) |
| 候选树 `pytest tests/` | 15 passed |
| 1-epoch smoke | `[graph batch] epoch order applied: 24556 items`,collision 0.0212,导出成功 |

### 下一轮的可判定预期

若覆盖是唯一限制,受约束对的 L1 共享率应从 0.02 量级显著上升,且治疗臂相对**打乱对照**的差应首次出现非零置信区间。
若两者都没变,则"覆盖不足"也被排除,机制在此预算下不可行 —— 那将是 §7 的判据,而不是又一次"未达成"。

现在按 §19g 起 round 3:治疗臂 + 打乱对照,各 3000 epoch。
