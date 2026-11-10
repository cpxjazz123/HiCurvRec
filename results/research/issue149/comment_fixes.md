## 控制器 R1–R10 / F1–F6 的逐条处置（全部实测，等审查后再谈 Stage2）

**提交**：`9948e496`（本次修复），前序 `1a335009`、`fe6f29be`；已 push 到 `main`。

### F1（机制级阻断）：结构项现在读的是**真实选码**，不是平面 softmax

`model/structure_gate.py` 把"soft 前缀"换成直通代理

```
h^(l) = one_hot(k_l) + (p^(l) − p^(l).detach())
k_L1/L2 = argmin(C^(l))               # 基线一、二层的规则
k_L3    = argmax(plan)                # 基线均衡分配：同一 centring、同一 ε=0.003、同一 50 次迭代、按行归一
p_L1/L2 = (1−S)·softmax(−centred(C)/T) + S/K ,  T = 2/64 = 0.03125
p_L3    = (1−S)·plan + S/K
S = 1e-3
```

两条常量的推导（不是试值）：centring 把 logit 跨度固定为 2 ⇒ `T = 2/64` 使 `exp(±32)` 双向都在 float32 内（离下溢 13 个数量级）；`S = 1e-3` 来自 float32 在 1 附近的表示分辨率（~1e-7）——**不加地板时锐 softmax 的最大值会四舍五入成 1.0，该行梯度精确为 0**（我在真实 cost 上测到过），加地板把最大值压到 `1 − S(1−1/K)`，argmax 不变。

**真实 probe pass 上的实测（`results/research/issue149/proxy_measurements.json`，3 次随机门控）**

| 验收项 | 要求 | 实测 |
|---|---|---|
| `argmax(载体行) == 实际 token` | 三层 100% | **1.0000 / 1.0000 / 1.0000** |
| 死行占比 | 0% | **0 / 3072，三层三次全部** |
| `L_closure` 端点（真实 cost） | <0.05 与 >0.95 | **0.0000 / 0.99867** |
| `L_usage` 灵敏度（某码占一半） | ≥0.10 | **0.3773** |
| 载体饱和证据 | 记录 | `soft_max_prob_mean` 0.59–0.74；`soft_min_entry` 3.9e-6（= S/K，无下溢） |
| `median |dL/dC|` | 你给的 1e-3 | **L1 1.5–1.9e-4，L2 8.5e-5–1.1e-4，L3 3.9e-6–8.6e-6**（见下） |

**关于 1e-3 这条阈值：我们没达到，如实报告。** 你的 1e-3 是在"256×256、随机 centred 矩阵"上量的；在真实 32 维混合代价上，行内代价差更大 ⇒ 同样的 T 下每行更接近 one-hot，`|dL/dC|` 中位数落在 1e-4（L1/L2）与 1e-6（L3，ε=0.003 的均衡计划本身近似离散）。与此同时：零死行、三层 argmax 一致、端点可达、`dL/dα` 中位数 4.5e-4 / 4.7e-4 / 1.3e-5（即门控确实收到结构项的梯度）。如果 1e-3 是硬性闸门而不是量级参照，请明示——那需要另一套推导（例如改 λ 或改载体尺度），按 §19g 我不得为了过诊断指标事后调常量。

`L_bridge` 在初始化时实测 0.0（跨社区真实边的前缀不相似度已经高于社区内真实边，hinge 未被触发），属定义内的"已满足"；它只在反向（跨组边比组内边更相似）时为正。这是设计上的守门项，不作有效应。

### F2：采样设计已换（在唯一那次正式训练之前）

实测触发：旧方案（全图 20 万三角，按 batch 内可用者取）在冻结 4×1024 配方下 **每步只有约 3.4 个三角形**（aborted run 的 `triangles_used/steps`），低于你给的 median ≥100。

新设计：**冻结 probe 子图 3072 个商品**（seed 42），三个项只在其**精确诱导集合**上计算——三角 1006、社区内边 2897、跨社区边 2119；每步覆盖 **1006/1006、2897/2897、2119/2119**（`triangles_used == triangles_total × steps`，有测试断言）。代价：每步多一次 encoder+gate+quantizer 前向（3072 行，无 decoder）。单卡一 epoch 实测 1.72 s（分派后四卡按 1024/rank，预计 ~0.6–0.9 s/epoch）。

### F4：规格已重写并重新冻结

`math_spec.json` 的 `computations` 现包含 `assignment proxy (straight-through)`（定义、T/S 的推导、实测数字）、逐项损失、训练目标，`excluded` 新增直通残差与迭代计划两条。表 SHA-256 随重建更新为 `216b4511…`（旧 `955e170a…` 作废，manifest 同步）。

### R2：描述表身份已绑到 checkpoint

`MixedRQ.get_extra_state` 写入 `structure_sha256`；`set_extra_state` 读回；`from_checkpoint` 比对不符即抛 `ValueError`；`StructureTable.load` 也记下文件 SHA。测试：`tests/test_gated_trainer_wiring.py` 用 `from_checkpoint` 还原并**拒绝另一张表**；`tests/test_real_batch_gate.py` 两 rank 读表逐项一致。

### §19b0：已清

`build_structure_table.py` 文件头的 `/home/...` 字面量已删（改为 `<stage-2 interpreter>`）。全树（含 model/、trainer、build 脚本）grep `/home/|/fs04/` = 0 命中。

### R7/R8：逐层诊断已进 `training_metrics.jsonl`

每 epoch 的 `structure` 事件现在有：`closure/bridge/usage/total`、三类集合 used/total、`steps`、`probe_items`，以及逐层 `codes_used`、`max_bucket_share`、`usage_entropy_normalised`、`mean_squared_euclidean`、`mean_squared_curved_over_four`、`euclidean_over_curved`、`max_abs_spearman_alpha_vs_descriptor`（含 argmax 维名）、`alpha_mean/std/min/max/below_0.05/above_0.95`。单卡一 epoch 的实样（未训练模型）：L1 `codes_used` 135、max bucket 7.1%、熵 0.708；L2 152 / 4.8% / 0.777；L3 246 / 0.96% / 0.959；`euclidean_over_curved` 0.90 / 1.02 / 0.92。

### 其它

- `n_communities=6`（Louvain resolution 1.0 默认值）**已写进规格**：跨社区 = 六个大块之间，不是小社群级桥接。
- 未观测商品的规则（标准化单位 `0.0` = 总体均值描述）保留；31 个商品的每层 α 会在正式跑的报告里逐层给出数值。
- `.pytest_cache/` 已删；全树已入 git。
- 全量套件：**160 passed / 10 failed**，10 个失败在冻结基线 `TIGER_mixed_rq` 上原样复现（同上一条 comment 列出的三类）。

### 仍未启动 Stage2

按要求，**实现 + 验证 + 提交 + Comment 之后等 Controller 审查通过**才启动完整 Stage2。请裁决：

1. 上述 F1/F2/F4/R2/§19b0 是否通过？
2. `median |dL/dC|` 未达 1e-3（实测 L1/L2 ~1e-4、L3 ~1e-6）但零死行、argmax 三层一致、端点可达——这个差距是否构成阻断？
3. `α` 饱和（aborted run 的失效模式）是否按 Issue「已知风险」如实记录、继续跑完不重跑？
