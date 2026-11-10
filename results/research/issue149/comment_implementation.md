## 实现完成，等待 Controller 审查后启动正式 Stage2

**候选树**：`stage2_RQ-VAE/TIGER_af3_gate_mixed_rq`（起点 `TIGER_mixed_rq`，只改机制面）
**提交**：`1a335009`（实现）、`fe6f29be`（证据 + 中止记录），均已 push 到 `main`

### 机制（对应 Issue §2）
- **冻结结构描述 `s_i`**（8 维，仅 TRAIN 图）：度、三角数、AF3 邻居均值、AF3 邻居分位差（q90−q10）、参与三角形邻居占比、不参与三角形邻居占比、邻居度数分布熵、跨社区邻居占比；按训练观测维度逐维标准化，测不到的维度写 `NaN` 并在 mask 里标记（不补 0 冒充有值），网关读到标记位时按**总体均值**路由，路由规则与缺失计数写入 manifest 和 run 记录。
- **逐层门控**：`α_{i,l} = σ(W2_l·GELU(W1_l s_i + b2_l))`，输出层零初始化 ⇒ 起点 `α ≡ 0.5`（与基线 `ALPHA_INITIAL` 逐点一致）。
- **代价**：`C^(l)_{ik} = 2[α_{i,l}‖rE−eE‖² + (1−α_{i,l}) d_c²(rH,eH)/4]`；α 只进入代价矩阵，**Sinkhorn / argmin / 残差递推 / 读出 / 导出全部沿用基线**（§19h）。
- **三项结构损失**（λ=1，均按定义落在 [0,1]）：`L_closure`（三角三边 coarse soft 前缀重叠）、`L_bridge`（跨社区真实边的 soft 前缀不相似度须高于社区内真实边，对称 hinge，不把低 AF3 边当负样本）、`L_usage`（逐层 soft 码使用熵亏损）。
- **三个开关**在 `model/structure_gate.py`：`BEHAVIOR_OFF`（读训练均值描述 ⇒ α 与商品无关）、`GEOMETRY_OFF`（c=0 ⇒ `d_0²/4` 即平方欧氏距离，1e-6 内）、`BOTH_ON`（出厂配置）。**不训练任何关闭开关的臂**（§19g）。

### 冻结输入（训练前算一次、写盘、记 SHA）
`results/research/issue149/structure_table.npz` + `structure_table.json`

- 重建 TRAIN 图与 issue #136 的冻结 AF3 缓存**逐边、逐值完全一致**（24556 节点 / 315391 边；AF3 精确相等，`af3_cache_status=MATCHED`）
- 表 SHA-256：`955e170abb0d9ed0346819497f573d665f34fe148e2f06333accd6a6d6906646`
- 覆盖 24587 个商品；31 个商品无图结构（全维 `NaN`）、82 个商品度=1（熵维不可测）
- 冻结集合：三角 200000、跨社区真实边 128450、社区内真实边 186941；Louvain resolution 1.0 / seed 42 ⇒ 6 个社区

### 代码验证（TDD，先红后绿）
- 先写测试 → 红：`ImportError: cannot import name 'structure_gate'`（4 个文件收集失败），记录在 `results/research/issue149/red_run.txt`
- 绿：`test_structure_table.py` 12、`test_structure_gate.py` 11、`test_structure_losses.py` 11、`test_real_batch_gate.py` 5、`test_reference_contract.py` 3、`test_gated_trainer_wiring.py` 3 全通过
- 覆盖内容：门的定义域与零初始化等于基线、`BEHAVIOR_OFF` 逐项同一 α、`GEOMETRY_OFF` 的欧氏极限、三项损失的闭式与有界性、真实 batch 上**编码器与门参数都收到梯度**、同一商品 id 在不同 batch 下 α 一致、训练与导出走同一门控、两 rank 读表逐项一致、`main()` 端到端导出与 checkpoint 还原一致、参考契约的 SymPy 语句与实现取值一致（rtol 1e-9）
- 候选树全量套件 `151 passed / 10 failed`；**10 个失败在冻结基线 `TIGER_mixed_rq` 上原样复现**（`test_prefix_eligibility` 8 项、`test_integration_regressions` 1 项、`test_support_regressions` 1 项），属基线既有红，不是本次改动引入
- 四卡一 epoch 冒烟（MCP worker 自己跑的）：`SMOKE_PASS`，24 步全部正梯度范数、`dead_terms=[]`

### 关于 Stage2 的现状（如实记录）
上一版指令下发的正式训练已经**在 epoch 644 中止**：新指令要求先提交 Commit + Comment 并经 Controller 审查通过后才允许启动完整 Stage2。中止时尚未到 epoch 3000，因此**没有写出 checkpoint、`item_sids.json` 或任何结果产物**；部分指标已移出结果目录、作为证据保存在 `results/research/issue149/aborted_run/`（`summary.json` + `training_metrics.jsonl`）。

中止前的诊断（只记录，不据此改任何常量、初始化或 λ）：
- 三码碰撞：epoch 50 `0.8297` → 100 `0.8640` → 250 `0.9756` → 400 `0.9960` → 600 `0.9985`
- α：三层均值在 epoch ~500 后**饱和到 0.0**（`alpha_std ≈ 0`），即 Scheme 里点名的“门控退化 / 某一几何分量被关掉”失效模式

### 请 Controller 裁决
1. 是否批准启动**一次**正式 Stage2（seed 42、3000 epoch、冻结全局 batch 4096、四卡 50202）？
2. 若批准：跑完后我会再提交 Commit + Comment 报告 `collision`、§63 分层指标（`record_single_sid`，在读 Stage3 之前产生）与诊断，**等待 Controller 确认碰撞率 < 0.40 后再启动 Stage3**。
3. 上述 α 饱和是否按 Issue §“已知风险”作为**失效模式如实记录、继续跑完**（不调 λ/初始化、不重跑）？
