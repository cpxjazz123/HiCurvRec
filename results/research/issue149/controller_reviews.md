===== 2026-11-10T05:03:00Z =====
## 控制器复审（第 2 次）：0 comments / 0 commits / 无候选树，关闭条件 0/3

- Stage2 完整训练：**未启动**（无 `stage2_RQ-VAE/TIGER_af3_gate_mixed_rq/`，无 `results/stage2_RQ-VAE/TIGER_af3_gate_mixed_rq/`，无 trainer 进程）
- SID 三码碰撞 < 0.40：**未测**
- Stage3 完整训练与结果验证：**未启动**

最后一条提交是 `f492552b`（15:20:58+1100），早于本 issue 创建（16:01+1100）；issue 更新时间即创建时间，无后续活动。

以下是**实现前必须先落进规格与候选树**的要求。四条都来自对冻结基线的实测读码，不是风格建议。

### R1 `ARM_VARIANT` 必须换成新标签，不能保留 `"F0"`

基线 `train_rqvae.py:68-81`：`ARM_VARIANT = "F0"`、`OUTER_LOOP = ARM_VARIANT in ("G","L","E")`、`AUX_AF3 = ARM_VARIANT == "A"`、`CURVATURE = 0.0 if ARM_VARIANT == "E" else 1.0`。而 `:646-666` 对 `ARM_VARIANT == "F0"` 在 **epoch 1500** 触发 `resolve_shared_readout(READOUT_FILE)`：

- `model/af3_runtime.py:110-132` 的 `validate_readout` 比对 `identity`，其中 `common_code` 含 `mixed_rq.py` 的 SHA（`train_rqvae.py:285-290`）——本机制**必然修改 `mixed_rq.py`**，身份校验失败；
- 即使身份通过，`train_rqvae.py:656-657` 还会比对 `snapshot_sha256`，与已冻结的 `results/research/issue139/readout.json`（`source_arm=F0`，`snapshot_sha256=8955e525…`）必然不符并抛 `ValueError`。

后果是 1500 个 epoch 之后才崩。新变体标签必须同时导出：`OUTER_LOOP=False`、`AUX_AF3=False`、`CURVATURE=1.0`。`OUTER_LOOP=False` 是硬要求：否则 `{0.25, 0.5, 0.75}` 的外循环会在 1500/2000/2500/3000 覆盖 α（`:70-71,646-661`），这既是 §19g 禁止的调参扫描，也让门控与表值混在一起、机制不可识别。基线 `AlphaTable` 的 `ALPHA_INITIAL = 0.5` 与门控零初始化起点一致，正是"起点逐点等于冻结基线"的依据。

### R2 α 的权威来源与 checkpoint 契约必须写死

现状：α 是**非参数** `AlphaTable`，靠 `MixedRQ.get_extra_state` / `set_extra_state` 序列化，且 curvature 不符时失败关闭（`model/mixed_rq.py:311-327`）。门控是**参数**，因此规格必须回答两件事：

1. 门控权重进 `state_dict`；`extra_state` 记结构描述表的 SHA-256 与行序身份，缺失/不符即失败关闭（不得静默退化成 `initial=0.5`）；
2. 导出（`train_rqvae.py:740-746`，`get_indices(infer_use_sk=False)`）用的 α 必须与最后一个训练 epoch 相同，并给出"训练 assignment == 导出 assignment"的测试。最后一个 epoch 用 Sinkhorn、导出用最近码字是基线既有差异，不得因此顺带改动 α。

### R3 结构描述表的行序必须与 `all_embeddings` 的图位置行序一致

导出按 `all_embeddings` 的固定顺序遍历（`train_rqvae.py:741-747`）。`s_i` 若按另一套行序索引，会静默把 α 发给错误的商品，而且指标上看起来完全正常。必须有一个测试：**打乱描述表行序后导出的 SID 必须改变**（证明表真的参与选码），外加加载时的行数与身份断言。

现成的训练图 AF3 缓存可直接用：`results/research/issue121/community_curvature/train_edge_af3.npz`（键 `rows` / `cols` / `af3`，315391 条边，训练划分）。它是距离的原始来源，`s_i` 需自行聚合出度、三角形、邻居 AF3 分位等。

### R4 碰撞必须按前三码算，且不得用"导出无重复"充数

本基线的 `item_sids.json` 是**4 个 token**：3 个语义码 + `_extend_collisions` 的 RecBole 扩展位（`train_rqvae.py:218-233, 747-749`），导出器只保证**4 元组**唯一（`:748` 的 `raise`）。因此"导出器没报错/没有重复 4 元组"**不能**当作碰撞达标的证据。参照实测：`struct_mixed_rq_ste_fix` 三码碰撞 `0.1855452` 而四元组 100% 唯一。报告必须写明：用的是前几码、读的是哪个路径、SHA-256 是多少，并把 §63 的 `l3_collision` 并列、带名字。

### R5 工作区脏，影响可审性

当前 `git status`：76 个 `idea/` 下的删除未提交，另有 `results/stage2_RQ-VAE/TIGER_af3_cost_side_rq/`、`results/stage2_RQ-VAE/TIGER_struct_mixed_rq/`、`review`、`write` 等未跟踪项。第一次提交请把机械规格、候选树与测试单独成一次提交，使之后每次 diff 都能分辨"本 issue 的改动"与之前遗留；`/tmp` 产物一律不落（§86）。

---

关闭条件（三条全中才可关闭，当前 0/3）：完整 Stage2（真实入口冒烟 + 3000 epoch + 导出 `item_sids.json`）、三码碰撞 `< 0.40`、完整 Stage3（冻结协议，`test_final.json` / `HG_Rec.log` / `training_metrics.jsonl` 可核验，变体目录同名）。
===== 2026-11-10T05:03:19Z =====
## 更正：R1 的第一条失败路径写错了

我上一条把 `validate_readout` 说成会因 `identity` 不符而抛错。读码更正：`model/af3_runtime.py:87-99` 明确**只记录**差异（`identity_differences`），不据此拒绝（owner 2026-11-08 的指示：摘要只作证据，不作许可）。它真正失败关闭的是 schema / `source_arm == "F0"` / `source_epoch == 1500` / `tau` 与 density 校准，已冻结的 `results/research/issue139/readout.json` 都满足。

R1 的结论不变，但**唯一**的失败路径是 `train_rqvae.py:656-657`：`ARM_VARIANT == "F0"` 时把当前模型的 `snapshot_sha256` 与文件里记录的 `8955e525…` 比对，不符即 `ValueError("existing F0 readout is bound to a different model snapshot")`。所以保留 `"F0"` 仍然会在 **epoch 1500** 崩，只是原因不是身份校验而是快照绑定。

R1 的其余部分（`OUTER_LOOP=False`、`AUX_AF3=False`、`CURVATURE=1.0`）不受影响。
===== 2026-11-10T05:06:24Z =====
## 控制器复审（第 3 次）：仍 0 提交 / 0 候选树 / 无 trainer 进程，关闭条件 0/3

距上次复审 90 秒，执行侧没有任何动作（`git log --since='2026-11-10 16:01'` 为空；`stage2_RQ-VAE/*af3_gate*`、`results/stage2_RQ-VAE/*af3_gate*`、`results/stage3_T5Train/*af3_gate*` 均不存在）。本轮不重复提要求，只把两处**会误导实现**的表述从 issue 正文改掉，留痕如下。

### 更正 1：`check_assignment_rule` 不是准入闸门

正文原话写「`check_assignment_rule` 仍在源码层面判不合格」，容易被当成派发闸门。实际：`tools/research/stage2/checks.py:573-585` 的 `run_all` 只跑 `check_entry`（§88），`check_assignment_rule` 是独立诊断。它的判据是：全树必须出现 `sinkhorn` 字面量，且不能有「同时含 `argmin` 与 `score`/`off`/`mechanism`/`D_k`/`distance -`/`distances -`」的行（`:424-464`）。所以"靠闸门替我兜住 argmin"是错的——这条只作诊断，选码规则必须自己守住。

### 更正 2：身份命名只有一条规则，不需要新增身份常量

正文只写了"结果目录 = `results/stage2_RQ-VAE/TIGER_af3_gate_mixed_rq`"，没有说清它与 MCP 解析器、Stage3 变体名的绑定关系。已按解析器与已配对臂核对并补进正文：

- `tree_arm` 读 `^EXPERIMENT_ARM`、`tree_result_dir` 优先 `^RESULT_DIR_NAME`、否则退回 `results/stage2_RQ-VAE/<树名>`、`tree_log_paths` 找 `树/logs/<arm>/` 再退 `树/logs`（`tools/genrec-mcp/server.py:680-720`）。
- 因此**同一个字符串**必须同时是：候选树目录名、`ARM_NAME`、`results/stage2_RQ-VAE/<名字>`、`results/stage3_T5Train/<名字>`。满足后三处解析自动一致，**不要**新增 `EXPERIMENT_ARM` / `RESULT_DIR_NAME`——声明它们的旧树用的是 `Path("/home/...")` 绝对路径，§19b0 明令禁止。
- 参照已配对臂 `TIGER_struct_mixed_rq_ste_fix`：`ARM_NAME` 等于树名，Stage2/Stage3 目录同名，`item_sids.json` 与 `test_final.json` 都在。
- `ARM_NAME` 是唯一被改写的身份常量；`ARM_VARIANT` 是机制开关（R1），不属于身份改名。

关闭条件：完整 Stage2 **0/1**、三码碰撞 < 0.40 **0/1**、完整 Stage3 与结果验证 **0/1**。
===== 2026-11-10T05:08:26Z =====
## 控制器复审（第 4 次）：仍 0 提交 / 0 候选树 / 无进程 — 补三条实现前约束 R6–R8

状态不变（`git log --since='2026-11-10 16:01'` 为空）。本轮补的三条都来自对基线选码规则的完整读码，目的只有一个：让 `math_spec.json` 与 `training_metrics.jsonl` 能**机械判定**机制是否真的在起作用。

### 基线的选码规则（先钉死，后面三条都依赖它）

`model/mixed_rq.py:299-310` 里 `sk_epsilon = sk_epsilon if last else -1.0`，`:145-152` 的 `_indices`：

- **只有第三层**走均衡分配，且只在 `self.training` 为真时（导出调用 `infer_use_sk=False`，所以导出时三层都是 `argmin`，见 R2）；
- 均衡分配是 `center_distance(distances)`（`:122-130`，按本 batch 的 max/min 居中并除以振幅）→ 双精度 → `sinkhorn(ε=SK_EPSILON=0.003, iters=SK_ITERS=50)`（`:132-141`）→ `argmin` 换成 `argmax(q)`；
- **一、二层是 `torch.argmin(distances)`**（`:152`）。这是基线规则，机制不得改。

所以 α 进入 `distances` 之后确实穿过均衡分配（§19h 满足），但它在 **L1/L2 是直接 argmin**——而坍缩恰好发生在那里（`af3_cost_side_rq`：L1 只用了 40/256，最大码占 62.9%）。

### R6 可微前缀概率必须与真实选码同源，且温度可推导

`L_closure` / `L_bridge` 需要"soft 前缀概率"。它必须是**真实选码所用的同一张代价矩阵**的函数，不能是 embedding 空间的替代量：

- L3：用 Sinkhorn 的传输计划 `q`（`mixed_rq.py:132-141` 的同一 ε、同一迭代数、同一 `center_distance`）；`argmax(q)` 必须等于该 batch 实际选出的 token——请写一条测试在**真实 batch**上验证这个对应关系（这是"训练 assignment == soft 近似"的唯一证据）；
- L1/L2：用 `center_distance(distances)` 上的 softmax，温度取基线自己的 `SK_EPSILON`（`train_rqvae.py:105`）；**不得**另挑温度，且必须写明其 `argmax` 与 `argmin(distances)` 一致；
- 必须写明 `center_distance` 是**逐 batch** 的 max/min 归一化，因此 soft 概率带 batch 依赖；任何"跨 batch 可比"的说法都要先给证据，否则不得据此下结论。

### R7 usage 项要落在 L1/L2，不是 L3

L3 的训练期选码已经被 Sinkhorn 强制均衡（`:299-310`），在 L3 上加熵亏损几乎恒等于常数项；真正可能坍缩的是一、二层的无约束 `argmin`。请把 `L_usage` 明确作用在**每层**的码使用分布上，并在报告里给出**逐层**使用数/最大桶占比（当前基线只记最终 SID 层统计，不足判据）。若坚持把 L3 也计入，必须解释它为什么不是冗余项。

### R8 必须记录"门控是否真的在起作用"的逐层证据

混合度量允许一个码在 E 半与 H 半各取折中，所以门控可能把所有商品推向同一类"折中码"——正是 `af3_cost_side_rq` 的坍缩形态。`training_metrics.jsonl` 每 `EVAL_INTERVAL` 必须记录：

1. 每层 α 的 mean / std / min / max 与落在 `[0,0.05] ∪ [0.95,1]` 的占比（塌到 0 或 1 意味着退化回单一几何）；
2. 每层**被选码**的 `d_E²` 与 `d_H²` 的均值、以及两者的比（若比值在全体商品上趋同，权重实际上没按商品分化）；
3. 每层码使用数、最大桶占比、使用熵；
4. 结构描述 `s_i` 与 α 的秩相关（至少中心化后的 Spearman），用于判断门控有没有学到结构依赖，而不是学成常数。

这些是**诊断记录**，不作闸门（§19g / §19 的"行为类诊断只记录"）；但缺它们就无法区分"机制无效"与"机制没跑起来"，报告不得省略。

关闭条件：完整 Stage2 **0/1**、三码碰撞 < 0.40 **0/1**、完整 Stage3 与结果验证 **0/1**。
===== 2026-11-10T05:10:56Z =====
## 控制器复审（第 5 次）：候选树已出现，但仍是未改动的基线副本 — 1 个硬阻断 B1

### 先更正我自己：第 4 次复审说"0 候选树"是错的

`stage2_RQ-VAE/TIGER_af3_gate_mixed_rq/` 在 **05:05–05:07Z 之间**出现。证据：第 3 次复审（05:04:34Z）对 `stage2_RQ-VAE/*af3_gate*` 报 not found，第 4 次复审（05:07:35Z）同一 glob 已命中，我当时漏看了那一条、只读了两条 results 错误行。目录 mtime 是 Nov 7 23:54，那是 `cp -a` 保留的源文件时间，不是创建时间。

### 实测：树与基线逐字节相同，机制尚未开始

| 文件 | 与 `TIGER_mixed_rq` 的 sha256 |
|---|---|
| `train_rqvae.py` | **SAME** |
| `model/mixed_rq.py` | **SAME** |
| `math_spec.json` | **SAME** |
| `reference_contract.json` | **SAME** |

`tests/` 只有基线继承来的 14 个文件，没有任何新机制测试。`git status` 显示该树仍未被跟踪（`??`），本轮 0 提交。

好的部分：副本里没有绝对路径、没有 `/tmp` 字面量；缺 `dataset/` 无影响——训练器只用 `torch.utils.data.TensorDataset`（`train_rqvae.py:554-557`），基线的 `dataset/` 是数据目录不是代码。

### B1（硬阻断，派发前必须改）：身份常量没改，结果目录撞已冻结的 F0 臂

`train_rqvae.py:62,68` 仍是 `ARM_NAME = "TIGER_mixed_rq_F0"`、`ARM_VARIANT = "F0"`。后果两条，都是实测：

1. `RESULT_DIR = REPO_ROOT/"results"/"stage2_RQ-VAE"/ARM_NAME`（`:82`）指向 `results/stage2_RQ-VAE/TIGER_mixed_rq_F0`——**该目录已有一个跑完的 #139 F0 臂**（`item_sids.json`、`rqvae_best.pth`、`sids_for_hgrec.npy`、`stage2_metrics/`，12 MB），且 `results/stage3_T5Train/TIGER_mixed_rq_F0` 也已存在。原样运行会写入/覆盖一个冻结臂的产物，§19b 明文要求这种输出路径冲突在训练前被拒绝。
2. 即使不看路径冲突，`ARM_VARIANT == "F0"` 会在 **epoch 1500** 撞上 `:646-666` 的 `resolve_shared_readout`，随后 `:656-657` 的 `snapshot_sha256` 与已冻结的 `8955e525…` 不符即抛 `ValueError`（见第 2 次复审 R1）。

修法就是已写进正文的一条身份规则：**同一个字符串**同时是树目录名、`ARM_NAME`、Stage2 结果目录名、Stage3 变体名（`TIGER_af3_gate_mixed_rq`）；`ARM_VARIANT` 换成 `{F0,G,L,E,A}` 之外的新标签，使 `OUTER_LOOP=False`、`AUX_AF3=False`、`CURVATURE=1.0`。不要新增 `EXPERIMENT_ARM` / `RESULT_DIR_NAME`（旧树那两者配的是 `/home/...` 绝对路径，§19b0 禁止）。

### 下一步的验收顺序（不跳步）

1. 先落 `math_spec.json`（覆盖 R6 的 soft 概率同源约束、R7 的逐层 usage、R8 的逐层诊断键），冻 SHA 后再动实现；
2. 按 §32b 先写会失败的测试（门控定义域、零初始化起点等于基线 `α=0.5`、三个开关、真实 batch 上 L3 `argmax(q)` 与实际 token 一致、描述表行序敏感性、训练/导出选码一致）；
3. 再改实现；4. 提交并推送（§43），提交里必须能分辨新增机制改动与基线副本；
5. 才谈 `stage2_start`。

拷贝本身不是交付物；当前无正式运行、无 SID、无碰撞数值。

关闭条件：完整 Stage2 **0/1**、三码碰撞 < 0.40 **0/1**、完整 Stage3 与结果验证 **0/1**。
===== 2026-11-10T05:12:56Z =====
## 控制器复审（第 6 次）：无新提交；补 R9 — 门控的索引管线是这次实现最容易静默写错的地方

状态：`train_rqvae.py` / `model/mixed_rq.py` / `math_spec.json` 与基线仍逐字节相同，0 提交，无结果目录，无进程（距上次复审 2 分钟）。本轮补的 R9 是"能跑但算错"的那一类错误，正好卡在结构描述与选码之间。

### 已核实的可行链路（照抄即可，别自己造图）

`train_rqvae.py:236-264` 的 `_frozen_structure()` 已经做了整条验证：`af3_structure.load_graph(AF3_CACHE)` → 与 `tools.research.science.community_curvature.data.build_graph("train")` 逐边比对（不等即 `ValueError`）→ `graph["item_ids"] = canonical.nodes`（图位置 → 商品 id）→ `target_kappa` 给出每条边的 `kappa` 与 **`degree`** → 再用 `EMBEDDING_FILE` 的行数校验 `item_ids` 越界。所以 `s_i` 的度、邻居 AF3、以及"同一张 TRAIN 图"的一致性证明都是现成的，**不要**另从 parquet 重建一张图（会与冻结的 AF3 目标不是同一张）。

缺口只有一个：`target_kappa` 只返回 `kappa` 和 `degree`，**三角形数 `triangles` 在函数内部算完就丢了**（`af3_structure.py:96-110` 的邻居求交）。`s_i` 要的三角占比必须在 `model/af3_structure.py` 里扩出返回值（候选树自有文件），或按同一 `_neighbours` 逻辑重算——两者都行，但必须在 `math_spec.json` 里写明是哪一个。

### R9：α 必须按**商品 id** 索引，不能按 batch 行号

实测索引语义（`train_rqvae.py:518-525, 554-557`）：

- `embeddings = maybe_apply_pca(load_embeddings(EMBEDDING_FILE), PCA_DIM, SEED)`，`all_embeddings = torch.from_numpy(embeddings)` ⇒ **`all_embeddings` 的行号就是商品 id**（PCA 不改行序）；导出端 `get_indices(all_embeddings, ...)`（`:741-746`）按此顺序遍历。
- `train_ids = np.unique(train_frame["target"])`（**升序**），`train_embeddings = all_embeddings[train_ids]`，训练集是 `TensorDataset(train_embeddings)`（`:554`）——**batch 的第 0 行是 `train_ids[0]`，不是商品 0**。
- 但 `model.encoder → model.rq(z)` 的 `forward` 只拿到 `z`，**拿不到商品 id**（`:527` 之后的训练步与导出步都是如此）。

因此直接按 batch 行号查 `s_i` 会得到"看起来完全正常"的错误结果：α 存在、训练收敛、指标不缺项，但每个商品拿的是别的商品的权重。正确做法只有一条：让 id 随数据进模型——`TensorDataset` 换成返回 `(row_index, item_id)` 的数据集（或等价的 sampler 派生），`forward` 增加 item-id 参数，门控用 `index_select(alpha_by_item, item_id)`；导出端传 `all_embeddings` 的恒等 id（0..N-1）。DDP 下 `DistributedSampler` 只切分样本、不改 id，别在 rank 之间对 id 再做 `all_gather`（会改变对齐关系）。

配套测试（都属于 §32b 的"先红"）：

1. 构造一个 `train_ids` 非连续的小 batch，断言门控实际取到的 α 等于 `alpha_by_item[train_ids[row]]`，且**不等于** `alpha_by_item[row]`；
2. 打乱 `alpha_by_item` 行序后导出的 SID 必须改变（证明表真被使用，R3）；
3. 训练末轮与导出的逐商品 α 全等（R2）；
4. `graph["item_ids"]` 与 `train_ids` 的交集/补集行为写清：训练只看得到 `train_ids`，但导出对全部商品都要 α，因此 `s_i` 表必须覆盖**全量**商品（图里没有训练边的商品怎么办，必须在规格里给规则，不能默认 0 或静默丢弃）。

关闭条件：完整 Stage2 **0/1**、三码碰撞 < 0.40 **0/1**、完整 Stage3 与结果验证 **0/1**。
===== 2026-11-10T05:13:41Z =====
## 控制器复审（第 7 次）：无新提交；补 R10 — 一 epoch 冒烟探针的调用契约（决定第一次派发能不能过）

状态：树与基线仍逐字节相同，0 提交，无结果目录，无进程。R9 说完"id 从哪来"，R10 说"框架已经先一步把 id 递进来了"——这两条合起来决定了实现的最省路径，也决定了第一次 `stage2_start` 是真通过还是假通过。

### 框架已经这么做（`tools/genrec-mcp/server.py:1236-1302`）

一 epoch 冒烟在跑完 `T.main()` 之后，用**真实 checkpoint + 真实 batch** 再探一次：

```python
item_ids = np.unique(frame["target"].to_numpy(dtype=np.int64))
rows = torch.from_numpy(embedding)[torch.from_numpy(item_ids)]
loader = DataLoader(TensorDataset(rows, torch.from_numpy(item_ids)), ...)
probe_model = _RQVAE.from_checkpoint(...) 或 _RQVAE(config, in_dim=...).load_state_dict(state, strict=True)
batch, batch_ids = next(iter(loader))
probed = probe_model(batch, batch_ids)                       # 位置参数，两个
total, recon_only = probe_model.compute_loss(reference, reconstructed, quant_loss)
```

`_RQVAE` 就是候选的 `model/__init__.py` 导出的 `RQVAE = MixedRQVAE`。**`batch_ids` 与训练端的 `train_ids = np.unique(train_frame["target"])` 是同一个升序数组**，所以探针递进来的 id 语义与 R9 要求的一致。

### 四条必须满足的后果

1. **`forward(embeddings, item_ids=None)` 这个签名不要改掉**——基线 `model/model.py:82-87` 已经接受它并注明"冒烟会带 id 调用"。门控应当**用它**（`item_ids=None` 时才退化成常量 α，并且那条退化路径必须在测试里显式覆盖），这样探针测的是真实机制而不是"忽略 id 的捷径"。
2. **`compute_loss(reference, reconstructed, quant_loss)` 拿不到 id**。所以 `L_closure` / `L_bridge` / `L_usage` 必须在 `forward` 里算完、并入返回的 `quant_loss`（或走第五个返回值的既有约定），**不能**指望在 `compute_loss` 里补 ids。返回值保持 4 元组或 5 元组，别改形状。
3. **`load_state_dict(state, strict=True)`**（除非该树定义 `from_checkpoint`）：门控参数是 `nn.Parameter`，自然进 `state_dict`；但**描述表 / 逐商品 α 表**若存成普通属性，`strict=True` 不会保护它——加载后 α 可能悄悄变成未初始化值。必须有一条测试：构造 → `state_dict()` → 新建模型 `load_state_dict(strict=True)` → 在同一批 id 上逐元素比较 α 完全相等。`model/mixed_rq.py:311-327` 的 `get_extra_state` / `set_extra_state` 是现成的挂载点，curvature 不符已失败关闭，描述表 SHA 应挂在同一处。
4. **探针报告 `recon_param_grad_norm_max` 与 `total_param_grad_norm_max`**（`:1293-1299`）。如果门控只经由 `L_closure`/`L_bridge` 拿到梯度，任何一处多余的 `detach` 都会让这两个数变成 0 而入口检查仍然"通过"。必须有一条测试：**只有辅助项**时，对门控参数求 `autograd.grad` 的范数 > 0；另一条：只有重构项时，编码器参数收到梯度（这是 §84 的 `encoder_gradient` 形态，虽然它已不是准入闸门，但门控失效是同一个失效）。

注：这条 checkpoint 探针**只在结果目录已存在时才跑**（`server.py:1213-1226` 的 `absent` 分支），所以第一次派发看不到它——别用"第一次冒烟过了"推断第 3、4 条成立。

关闭条件：完整 Stage2 **0/1**、三码碰撞 < 0.40 **0/1**、完整 Stage3 与结果验证 **0/1**。

===== 2026-11-10T05:17:07Z =====
## 控制器复审（第 8 次）：无新提交；实测结构描述的覆盖率并把 R1–R10 收进正文

状态：`train_rqvae.py` / `model/mixed_rq.py` / `math_spec.json` 与基线仍逐字节相同，0 提交，无结果目录，无 trainer 进程。

### 实测：描述表覆盖率（这决定规格里要不要一条缺失值规则）

| 量 | 实测值 |
|---|---|
| 目录商品数（`stage1_GeneEmbedding/output/sentence_t5.npy` 行数） | 24587 |
| 训练集唯一 target（`results/stage0_build_parquet/train.parquet`） | 24556 |
| 训练图节点（`train_edge_af3.npz` 的 `nodes`） / 边 | 24556 / 315391 |
| 零度节点 | **0** |
| 度分位 min/p5/p50/p95/max | 1 / 5 / 13 / 82 / 2220 |
| 从不出现在训练 target 的商品 | **31**（如 7352、14065、14773、17033…；可由"目录 id 集 − 唯一 target 集"确定性重算） |

结论两条，已写进正文：

1. **训练侧没有缺失值问题**：24556 个可训练商品全部至少有一条训练边，`s_i` 对每一个都能定义，不需要"缺维"分支。
2. **导出侧必须有一条显式规则**：那 31 个商品没有图行，但导出要为全部 24587 行给 SID。规则定为逐层 **中性 `α = 0.5`**（等于冻结基线的值，可推导、不制造结构信息），规格里写明，报告里列出这 31 个 id 与描述表 SHA-256；禁止邻居插补（无训练边）、禁止静默 0 或 `NaN`。

### 正文已重写为单一 brief

R1–R10 现在集中在新章节「实现约束清单（R1–R10，派发前必须全部满足）」里，每条一句 + `file:line` 证据；上面的实测覆盖范围与 31-id 规则放进了描述表那一节；完成标准新增"必须报告 31 个商品的中性 α 处理与碰撞用的是前三码"。这样冷启动不需要读五个评论串。

关上去留：**不关**。关闭条件仍是 完整 Stage2 **0/1**、三码碰撞 < 0.40 **0/1**、完整 Stage3 与结果验证 **0/1**。
===== 2026-11-10T05:22:28Z =====
## 控制器复审（第 9 次）：机制代码出现了 — 测试 4 过 1 败，另有 2 条机制级缺陷

首次有实质实现：`model/structure_gate.py`(269 行) + `model/structure_table.py`(421 行) + `mixed_rq.py` 的门控分支 + 4 个新测试文件。`train_rqvae.py`、`model/model.py`、`math_spec.json` 仍与基线逐字节相同，0 提交，无结果目录，无 trainer 进程。

### 测试实跑（`genrec_env_v2`，4 个新文件，`-x`）

`4 passed, 1 failed in 128.92s`。失败是**测试自身**的 bug，不是实现的：

```
tests/test_structure_gate.py:88
    head[-1][0].weight.normal_(0.0, 1.0)
E   TypeError: 'Linear' object is not subscriptable
```

`head[-1]` 已经是 `nn.Linear(hidden,1)`，再 `[0]` 取不到子模块；应为 `head[-1].weight`。修掉后请把 BEHAVIOR_OFF 那条断言跑通——**该开关的"与商品无关"目前尚未被证明**，`-x` 也让后面三个文件本轮没有执行。

### F1（机制级，必须先修）：第三层的 soft 前缀不是实际分配

`structure_gate.soft_prefix(cost) = softmax(-centred(cost)/TEMPERATURE)`，`TEMPERATURE = 1.0`，三层共用。但第三层的真实规则是 `argmax(Sinkhorn(centred(cost)/ε))`，`ε = SK_EPSILON = 0.003`（`mixed_rq.py:132-152`；`train_rqvae.py:105`）。两处不一致：

1. **argmax 不一致**：`argmax(softmax(-C)) = argmin(C)`，而 L3 的 token 是 `argmax(q)`，一般不等于 `argmin(C)`——所以 `L_closure`/`L_bridge` 在 L3 上优化的分布不是产生 token 的那个分布；
2. **尺度不一致**：T=1 对 ε=0.003，代理分布比真正决定 token 的分布平坦约 333 倍。后者让闭合项可以在**完全不改变任何 hard token** 的情况下被压到 0（把 soft 前缀抹平即可），这正是"代理 ≠ 机制"的失效形态。

要求：L3 的 soft 前缀改成同一 `center_distance`、同一 ε、同一迭代数得到的传输计划 `q`（按行归一：`q / q.sum(-1, keepdim=True)`）；并在**真实 batch**上证明 `argmax(q)` 逐元素等于该层实际 token。一、二层（`argmin(cost)`）保持现在的 softmax 即可——它的 argmax 与 `argmin` 天然一致。现有 `test_real_batch_gate.py:136` 只验了一、二层的这个对应，所以这条缺口测试看不见。

### F2（量化）：闭合项每步只看到约 14 个三角形

`structural_objective` 只使用**全部顶点都在本 batch 内**的集合（这是对的，别改成静默丢弃），于是按实测规模：

| 量 | 实测/推算 |
|---|---|
| 每卡 batch（`train_rqvae.py:106`） | 1024（全局 4096） |
| 可训练商品 | 24556 |
| `TRIANGLE_CAP` / `PAIR_CAP` | 200000 / 200000 |
| 每步可用三角形 ≈ 200000×(1024/24556)³ | **14.5**（占集合 7.3e-5） |
| 每步可用同社区/跨社区对 ≈ 200000×(1024/24556)² | **348 / 348** |

也就是说闭合项每步只有约 15 个三角形（约 45 个对项）在贡献梯度，且会有整步为 0 → `_connected_zero`。要求：用真实 sampler、冻结 batch size 实测 `triangles_used` 的 median / p5 / 为 0 的步占比，以及三类集合的 used/total，写进 metrics 与报告；若"为 0 的步"超过 5% 或 median 低于 100，**必须在唯一那次正式训练之前改采样设计**（例如在 batch 内按冻结邻接构造三角形/对，而不是从 20 万条的固定集合里碰运气）。§19g 只给一次机会，项"存在但基本不触发"不算耦合成立。

### F3：门控还没接进模型，且过不了框架的 checkpoint 探针

- `model/model.py` 仍与基线相同 ⇒ `MixedRQ(gate=None)`，没有任何地方构造 `StructureGate`；`train_rqvae.py` 也没把 id 递进 `forward`。当前 4 个新测试测的是模块，不是这条臂（这是进行中状态，不是缺陷，但别据此认为 R9/R10 已满足）。
- `MixedRQ.structure` 是 `register_buffer(..., persistent=False)`，构造时不读盘。框架探针的路径是 `_RQVAE(config, in_dim=...)` + `load_state_dict(strict=True)`（`tools/genrec-mcp/server.py:1252-1257`）：非持久 buffer 不会被加载，恢复出来的模型 `structure` 为空，`alphas()` 直接抛 `no structure table is attached`。要么实现类方法 `from_checkpoint(config, state, in_dim=...)` 在内部重新读表并 `verify()`（框架只用它替代裸构造），要么把表身份放进 `get_extra_state` 并失败关闭。否则第二次派发时的探针会崩，或者更糟——量到的是没有门控的基线路径。
- 现状 `get_extra_state` 仍只序列化 `alpha_table` + curvature（`mixed_rq.py:311-327` 未绑定描述表 SHA-256），R2 的"描述表身份随 checkpoint 失败关闭"还没落。

### F4：规格仍未冻结，顺序反了

`math_spec.json` 与基线**逐字节相同**，实现先落地。brief 要求"先冻 `math_spec.json` 再动实现"，§81/§82 也要求实现能被冻结规格逐条比对；现在没有规格，`structure_gate.py` 里的 `TEMPERATURE=1.0`、`GATE_HIDDEN=2×DIMS`、`LAMBDA=1` 这些取值只能靠读代码推断依据，无法做"规格 ↔ 实现"的一致性裁决。

### F5：31 个无结构商品的处理已按实现更新正文

实现把未观测项落成标准化单位下的 `0.0`（= 总体均值描述，`standardise` 里有明确注释），即那 31 个商品取 `σ(f_l(0))`——对商品恒定、与 `BEHAVIOR_OFF` 自洽，比我原先写的固定 `0.5` 更一致。已把正文改为"总体均值描述（标准化单位 `0.0`）"，并要求报告写出它们每层的具体 α 值与数量。

### F6：卫生

树里出现 `.pytest_cache/`，不要入库；整棵树仍未跟踪、0 提交（§43 要求每次迭代提交并推送）。

关闭条件：完整 Stage2 **0/1**、三码碰撞 < 0.40 **0/1**、完整 Stage3 与结果验证 **0/1**。
===== 2026-11-10T05:25:31Z =====
## 控制器复审（第 10 次）：接线与冻结表落地；F1/F2/F4 仍未修

自上次复审（05:23Z）新增：`model/model.py` 的门控接线 + `from_checkpoint`、树根新增 `build_structure_table.py`、以及**冻结结构表本身已生成并可用**。仍 0 提交、无结果目录、无 trainer 进程。

### 已核实通过的部分

| 项 | 实测 |
|---|---|
| 冻结表 `results/research/issue149/structure_table.npz` | 24587 行 × 8 维，可被候选自己的 `StructureTable.load` 读取 |
| manifest 绑定 | `structure_table.json` 的 `table_sha256 = 53bdfbde78f4…` **等于** npz 的 sha256（我独立算过），`verify()` 通过 |
| 无结构商品 | **31**（与我在第 8 次复审独立算出的数字一致） |
| 未观测条目（逐维） | 299 |
| 冻结集合 | triangles 200000、same_pairs 200000、cross_pairs 128450（`n_communities=6`） |
| 图核对 | `n_nodes=24556`、`n_edges=315391`、`af3_cache_status=MATCHED`，与我的独立测量一致 |
| `model.py` 接线 | `forward` 递 id、`get_indices` 缺省用恒等 id、`from_checkpoint` 重新读表、`_repository_root()` 用标记目录而非数父目录（§19b0） |

### 仍必须修

**F1（机制级，主要阻断）**：`structure_gate.py:156-162` 的 `soft_prefix` 仍是 `softmax(-centred(cost)/TEMPERATURE)`、`TEMPERATURE=1.0`，三层共用；`grep sinkhorn|transport` 在 `structure_gate.py` 里仍是 0 命中，`test_real_batch_gate.py` / `test_structure_losses.py` 里也没有第三层的 `argmax(q)` 对应测试。要求不变：L3 的 soft 前缀必须是同一 `center_distance`、`ε=SK_EPSILON=0.003`、同一迭代数的传输计划按行归一，并有真实 batch 测试证明 `argmax(q)` 逐元素等于实际 L3 token。

**F2**：集合规模仍是 200000，每卡 batch 1024 ⇒ 每步可用三角形 `200000×(1024/24556)³ ≈ 14.5`。要用真实 sampler 实测 `triangles_used` 的 median/p5/零步占比并写进报告；零步 >5% 或 median <100 就在唯一那次训练前改采样设计。

**F4**：`math_spec.json` 仍与基线**逐字节相同**。`TEMPERATURE=1.0`、`GATE_HIDDEN=2×DIMS`、`LAMBDA_*=1`、`ALPHA_INIT=0.5` 这些依据只写在代码注释里，规格还没有可裁决的语句。

**R2 的新缺口**：`get_extra_state`（`mixed_rq.py:361-365`）仍不包含描述表身份，而 `from_checkpoint` 只用 `verify()` 校验"表与它自己的 manifest 一致"——**这不绑定到 checkpoint**。换成另一张表（新 manifest、SHA 合法）也能静默恢复，α 就变了。要求：`extra_state` 里写入 `table_sha256`，`from_checkpoint` / `attach_structure` 比对不符即抛错，并加一条"换表后恢复必须失败"的测试。

**§19b0 字面违规**：`build_structure_table.py` 文件头 docstring 里有 `/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9` 绝对路径字面量，而该文件位于 `stage2_RQ-VAE/` 下。同时它写着"the trainer refuses to start without the manifest-verified table"，但 `train_rqvae.py` 仍与基线逐字节相同，这句话现在不成立——要么实现该拒绝，要么删掉这句承诺。

**可报告属性（不必改，但必须写进报告）**：`n_communities=6`（Louvain resolution 1.0 的默认值）。6 个社区意味着 `cross_community_share` 这个维度与 `L_bridge` 的对比粒度是"6 大块之间"，不是细粒度社区；报告里要说明这点，别让读者把它理解成小社群级桥接。

**卫生**：`.pytest_cache/` 仍在树内；整棵树仍未跟踪、0 提交（§43）。

关闭条件：完整 Stage2 **0/1**、三码碰撞 < 0.40 **0/1**、完整 Stage3 与结果验证 **0/1**。
===== 2026-11-10T05:28:43Z =====
## 控制器复审（第 11 次）：训练器接线完成且基本正确；测试 20 过 1 败；F1/F4/R2/§19b0 未修

自上次复审（05:26Z）`train_rqvae.py` 被改写。仍 0 提交、无结果目录、无 trainer 进程。

### 已核实通过（逐项对照 R1/R7/R8/R9/R10）

| 项 | 证据 |
|---|---|
| R1 身份 | `ARM_NAME = "TIGER_af3_gate_mixed_rq"`（= 树名 = 结果目录 = Stage3 变体名）；`ARM_VARIANT = "AF3GATE"` ⇒ `OUTER_LOOP=False`、`AUX_AF3=False`、`CURVATURE = 0.0 if GEOMETRY_OFF else 1.0` |
| R9 id 管线 | `TensorDataset(train_embeddings, train_ids)`、`model(batch, batch_ids)`、`raw_module.init_codebook(..., ids)`；导出/评估走 `get_indices(all_embeddings)`，`model.get_indices` 在门控存在时缺省用恒等 id，且加载时断言 `structure_table.item_count == len(all_embeddings)` |
| 表加载 | 每个 rank 都 `StructureTable.load`（manifest SHA 校验）+ 行数校验；缺表即 `FileNotFoundError`（`build_structure_table.py` docstring 里那句承诺现在有代码支撑了） |
| R7/R8 记录 | 每步 `structural_objective(last_prefixes, batch_ids, ...)` 并入 loss；每 epoch `_record("structure", ...)` 带 `closure/bridge/usage/total`、`triangles_used/total`、`same_pairs_used/total`、`cross_pairs_used/total`、`steps`、逐层 `alpha_mean/std/min/max/below_0.05/above_0.95` |

### 测试实跑（`genrec_env_v2`，`test_structure_gate.py` + `test_structure_losses.py`）

`20 passed, 1 failed in 30.64s`：

```
tests/test_structure_gate.py:81 test_behaviour_off_makes_every_item_share_one_alpha_per_level
    assert bool((alpha == alpha[0]).all())
E   tensor([0.5819, 0.5819, ...]) == tensor(0.5819) -> False
```

这是**测试自己**用了浮点精确相等：六个值在打印精度内一致，低位因 batched GEMM 的累加顺序不同而不相等。开关实现本身没问题（`structure_gate.py:140` 用 `zeros_like`，标准化单位下就是总体均值描述）。改成 `torch.allclose(alpha, alpha.expand_as(alpha), rtol=1e-6, atol=1e-7)` 或断言 `alpha.max() - alpha.min() < 1e-7`。

### 仍必须修（按阻断度排序）

1. **F1（机制级，唯一真正的机制缺陷）**：`structure_gate.py:156-162` 的 `soft_prefix` 仍是 `softmax(-centred(cost)/1.0)`，三层共用；`sinkhorn`/`transport` 在门控模块里 0 命中，两个损失测试文件里也没有第三层的 `argmax(q)` 对应。第三层的 token 由 `argmax(Sinkhorn(centred(C)/0.003))` 决定，而闭合/桥接项优化的是 `softmax(-C)`——**两者在 argmax 上一般不同、尺度上差约 333 倍**，所以这两个项可以在不改动任何一个 hard token 的情况下被压到 0。修法与测试见第 9 次复审 F1。
2. **F2 的判定方式**：现在每 epoch 都记 `triangles_used/total` 与 `steps`（好），但那是跑完之后才知道。判定必须用 `stage2_start` 的**一 epoch 冒烟**给出的数字：`triangles_used / steps` 的均值（以及能否给出零步占比）。若每步可用三角形中位数 < 100 或零步 > 5%，就在 3000 epoch 启动前改采样设计，别把它留成"训完才发现项没触发"。
3. **F4 规格仍未冻结**：`math_spec.json` 与基线逐字节相同。`TEMPERATURE=1.0` 与真实分配的 `ε=0.003` 的关系、`GATE_HIDDEN=2×DIMS`、`LAMBDA_*=1`、`ALPHA_INIT=0.5` 目前只有代码注释作依据，没有可裁决的语句。
4. **R2 描述表身份**：`mixed_rq.py` 里 `table_sha256` 出现 **0 次**，`get_extra_state` 仍不绑定描述表；`from_checkpoint` 只校验"表与自己的 manifest 一致"，换一张合法表也会静默恢复。
5. **§19b0**：`build_structure_table.py:3` 的 docstring 仍有 `/home/wlia0047/...` 绝对路径字面量（该文件位于 `stage2_RQ-VAE/` 下）。
6. **卫生**：`.pytest_cache/` 仍在树内；整棵树未跟踪、0 提交（§43）。

小瑕疵（不阻断）：epoch 结束时 `gate_alphas` / `structure_logs` 各被清空两次（rank-0 记录块内一次、紧随其后一次），第一处是冗余的。

关闭条件：完整 Stage2 **0/1**、三码碰撞 < 0.40 **0/1**、完整 Stage3 与结果验证 **0/1**。
===== 2026-11-10T05:32:40Z =====
## 控制器复审（第 12 次）：规格已落地，但规格把"结构性监督失效"写成了定义 — 附实测

自上次复审（05:29Z）：`math_spec.json` 不再是基线副本（新增 `computations / switches / coupling / excluded`，逐条给出定义、输出与梯度），并新增 `tests/test_gated_trainer_wiring.py`。仍 0 提交、无结果目录、无 trainer 进程。

### 本轮的核心发现：T=1 的 soft 前缀在数值上不可能携带信号

规格 `computations["soft prefix approximation"]` 写的是 `q^(l) = softmax(-centred(C^(l)) / 1)`，理由是"centring 已经把矩阵映射进约 [-1,1]，所以温度为 1"。这个推导只对了一半：centring 固定的是**取值区间**，而真实分配的**锐度**由 `ε = SK_EPSILON = 0.003` 决定。

我在本项目解释器里实测（`genrec_env_v2`，numpy，1024×256，用基线自己的 `centre_distance`）：

| 量 | T = 1（当前实现/规格） | T = 0.003（真实分配尺度） |
|---|---|---|
| centred 取值范围 | −0.99998 … +0.99998 | 同左 |
| 每行最大概率（均值） | **0.00488** | 0.690 |
| 行熵（log 256 = 5.545） | 5.532 | 0.834 |
| 两两 overlap 的 mean ± std | 0.003906 ± **0.000007** | 0.005452 ± **0.048711** |
| `L_closure = 1 − overlap` | 0.996094 ± 0.000007 | 0.994548 ± 0.048711 |
| `L_usage`（熵亏损） | **0.0023** | 0.850 |

这是**结构性的**，不是分布假设：`_centred` 把矩阵严格映射进 [-1,1]，所以 T=1 时每个 logit 都在 [-1,1]，任意一行的单个码概率上界 `e²/(e²+255) = 2.82%`。也就是说"同码"与"异码"在 soft 前缀里最多差 2.8%，`L_closure` / `L_bridge` 在 6 位小数内是常数，`L_usage` 恒为 0.0023。三个结构性项在当前设定下**几乎不随训练变化**，门控实际只能从重构损失里学到东西——而报告很可能把召回变化归因于"结构耦合"。

要求（实现与规格同时改，改完重新冻 `math_spec.json` 并记录新 SHA 与理由）：

1. 代理分布的锐度必须与真实分配同尺度：第三层用同一 centring、`ε = SK_EPSILON = 0.003`、同一迭代数的传输计划按行归一；一、二层用 `softmax(-centred(C)/SK_EPSILON)`（可推导的理由：这是基线判别码字的尺度），或任何被**推导**且在测试里展示其能达到项端点（0 与 1）的温度；
2. 真实 batch 测试：三层都断言 `argmax(代理行) == 该层实际 token`（第三层即 `argmax(q)`）；
3. 测试里给出动态范围证据：同一 batch 上 `L_closure` 在"全部同码"与"无同码"两种构造下分别接近 0 与 1（现有 `test_structure_losses.py` 用手造 one-hot 前缀测过端点，但**没有**在真实 cost 上展示可达到性）。

### 其余状态

- **F4 已推进**：`math_spec.json` 现在有完整语句、`switches`（三个开关的定义与我 brief 一致）、`coupling`（行为侧/几何侧/不在通路上的东西）、`excluded`（离散取码、centring 的 max/min、detach、解码器 STE 逐条附理由）。唯一问题是它把 T=1 固化成定义，需按上条修订后重新冻结。
- **R2 仍开**：`mixed_rq.py` 里 `table_sha256` 仍 0 次出现，`from_checkpoint` 只做"表 ↔ 自身 manifest"校验，换一张合法表仍会静默恢复。
- **§19b0 仍开**：`build_structure_table.py:3` 的 `/home/...` 字面量还在。
- 新测试 `test_gated_trainer_wiring.py` 覆盖导出等价、表覆盖不足即拒绝、`BEHAVIOR_OFF` 下导出与商品无关；我已发起一次真实执行，结果在下一轮报出（不预判）。
- 卫生：`.pytest_cache/` 仍在树内；整棵树未跟踪、0 提交（§43）。

关闭条件：完整 Stage2 **0/1**、三码碰撞 < 0.40 **0/1**、完整 Stage3 与结果验证 **0/1**。
===== 2026-11-10T05:32:47Z =====
## 补记：上一轮承诺的测试实跑结果

命令：`genrec_env_v2/bin/python3.9 -m pytest tests/test_structure_gate.py tests/test_structure_losses.py tests/test_gated_trainer_wiring.py -q`

```
24 passed in 66.52s
```

- 第 11 次复审里那条浮点精确相等（`alpha == alpha[0]`）已修，`BEHAVIOR_OFF` 的"与商品无关"现在由通过的测试支撑；
- 新增的 `test_gated_trainer_wiring.py` 三项（导出等价、表覆盖不足即拒绝、`BEHAVIOR_OFF` 下导出与商品无关）全部通过。

这不改变上一轮的结论：这些测试证明的是**接线正确**，而 T=1 代理分布的动态范围问题是数值事实（overlap `0.003906 ± 0.000007`，`L_usage ≈ 0.0023`），现有测试用手造 one-hot 前缀验证端点，**没有**证明在真实 cost 上这些端点可达。
===== 2026-11-10T05:34:32Z =====
## 控制器复审（第 13 次）：§19b0 已修；代理分布仍是 T=1（代码与规格都未改）；补可机械判定的验收阈值

自上次复审（05:34Z）：`structure_gate.py` 与新增的 `tests/test_reference_contract.py` 有改动；`build_structure_table.py` 的绝对路径字面量已删除。仍 0 提交、无结果目录、无 trainer 进程。

### 已关闭

- **§19b0**：`build_structure_table.py` 中 `/home/...`、`/fs04/...` 均为 0 命中（我重新 grep 过）。
- **F/接线**：`test_gated_trainer_wiring.py` 三项通过（上一轮实跑 24 passed）。

### 仍未改（唯一机制级阻断）

`structure_gate.py:161-167` 仍是 `q = softmax(-centred(C)/TEMPERATURE)`、`TEMPERATURE = 1.0`；`math_spec.json` 的 `computations["soft prefix approximation"]` 仍写着"temperature is one because the centring fixed the scale"。第 12 次复审给出的实测没有变化：T=1 时 `L_closure = 0.996094 ± 7e-6`、`L_usage ≈ 0.0023`，三个结构性项在数值上是常数。

新增的 `test_reference_contract.py` 校验的是门控调用与契约语句（GELU 形式）一致，**不触及锐度**——契约值和实现值都等于 T=1，所以它通过与"项是否有效"无关。这正是 §87 之后这类自证测试的通用风险：契约与实现同源。

### 把"项是否活着"变成可机械判定的阈值

修完之后，请用**真实 cost 矩阵**（不是手造 one-hot）满足以下三条，并把数值写进报告与测试：

1. **对应性**：三层都满足 `argmax(代理行) == 该层实际 token` 的行占比 ≥ 99.9%（第三层即 `argmax(q_sinkhorn)`）；
2. **动态范围**：在同一 batch 上构造"全部同码"与"无同码"两种前缀，`L_closure` 分别 < 0.05 与 > 0.95；`L_bridge` 在"跨社区对全部异码、同社区对全部同码"时取 0，反向时取 1；
3. **usage 灵敏度**：把某个码的 soft 使用率人为抬高到 50% 后，`L_usage` 相对均匀分布时必须变化 ≥ 0.10。

现有 `test_structure_losses.py` 用**手造** one-hot 前缀验过端点；那证明的是函数定义，不证明在真实 cost 上这些端点可达——第 2 条就是补这个缺口。

### 其余

- **R2 仍开**：`mixed_rq.py` 里 `table_sha256` 仍 0 次出现；`from_checkpoint` 只做"表 ↔ 自身 manifest"校验。
- **提交**：整棵树仍未跟踪、0 提交（§43）。下一轮如果能读到提交，我就按提交的 diff 复审，不再逐文件扫工作区——这样也才能区分"本 issue 的改动"与残留。
- 卫生物理：`.pytest_cache/` 仍在树内。

关闭条件：完整 Stage2 **0/1**、三码碰撞 < 0.40 **0/1**、完整 Stage3 与结果验证 **0/1**。
===== 2026-11-10T05:34:45Z =====
## 更正：§19b0 **没有**修好，是我上一条读错了

上一条我把"§19b0 已修"写成了结论，错了。实际复查：

```
$ grep -c '/home/\|/fs04/' stage2_RQ-VAE/TIGER_af3_gate_mixed_rq/build_structure_table.py
1
$ sed -n '3p' ...
    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 build_structure_table.py
```

绝对路径字面量仍在第 3 行，§19b0 仍开。

原因：我把多个检查串在 `&&` 链里，前一条 `grep -c 'table_sha256'`（R2，命中 0）以退出码 1 结束，**后面的 §19b0 grep 根本没执行**，我把"没有输出"误读成"0 命中"。这是检查方法的问题，不是树的进展。

同轮其它结论不变，且都有独立证据：F/接线（`24 passed`）、代理分布仍是 `TEMPERATURE = 1.0`（代码 `structure_gate.py:161-167` 与规格 `computations["soft prefix approximation"]` 两处都还在）、R2 的 `table_sha256` 在 `mixed_rq.py` 中 0 次出现（这一条是实际执行过的）、0 提交、无 trainer 进程。

后续我的检查会逐条独立执行，不再用 `&&` 串联互不相关的断言。
===== 2026-11-10T05:37:24Z =====
## 控制器复审（第 14 次）：状态无变化 — 并更正我上一轮给的验收阈值（纯 softmax 代理**到不了**端点）

05:36Z 复核（逐条独立执行）：0 提交；`structure_gate.py:65` 仍 `TEMPERATURE = 1.0`（`def soft_prefix` 仍在 161）；`mixed_rq.py` 中 `table_sha256` 0 命中；`build_structure_table.py` 仍有 1 处 `/home/...`；无结果目录、无 trainer 进程。

### 更正：我第 13 次给的"动态范围"阈值对纯 softmax 代理不可达

我在本项目解释器里把这件事量了一遍（256 行 × 256 码，用基线 `centre_distance` 归一，测的是闭包项这类非平凡泛函的梯度）：

| T | 每行最大概率 | 下溢为 0 的条目占比 | `|dL/dC|` 最大 | 零梯度行占比 | 两两 overlap 的 std |
|---|---|---|---|---|---|
| 1.0（当前） | 0.0090 | 0.0% | 4.1e-7 | 0% | 8.0e-5 |
| 0.1 | 0.0752 | 0.0% | 4.1e-4 | 0% | 2.0e-3 |
| 0.03 | 0.2251 | 0.0% | 1.35e-2 | 0% | 7.8e-3 |
| 0.01 | 0.4911 | 65.7% | 1.38e-1 | 0% | 2.5e-2 |
| **0.003**（= SK_EPSILON） | 0.7809 | **89.4%** | 6.4e-1 | **5.5%** | 3.9e-2 |

结论三条：

1. T=1 的项确实死（梯度 4e-7，overlap 变动 8e-5），这一点不变；
2. 但**把 T 调到 ε=0.003 也不行**：float32 下 89% 的条目下溢、**5.5% 的行梯度恒为 0**（这些行在数值上是精确 one-hot）。在每步只有约 14 个三角形的条件下，等于随机丢掉一部分商品的结构监督；
3. 任何纯 softmax 代理在 K=256 下都**到不了**我上轮写的端点（`L_closure < 0.05` 需要每行概率接近 1，而那就必然下溢）。所以那条阈值是我给错了。

### 因此把修法改成可同时满足"语义正确 + 梯度存活"的形式

**用直通（straight-through）硬分配来算三个项**：`h = one_hot(argmax(q_hard)) + (q_soft - q_soft.detach())`，其中

- 第三层 `q_hard` 就是本层的 Sinkhorn 传输计划 `q`（同一 centring、同一 ε=0.003、同一迭代数，按行归一）——这样 `overlap` 在语义上就是"真实 token 是否相同"，端点 0/1 可达；
- 一、二层 `q_hard = centred(C)` 的 argmin（`argmax(-centred(C))`，与基线取码一致）；
- `q_soft` 的温度按**可推导**规则取"不触发下溢的最锐值"：centred 跨度为 2，取 `T = 2/64 = 0.03125`（最大指数 32 ≈ ln(8e13)，离 float32 的 1e-38 下溢线还有十几个数量级），于是"无条目下溢、行行有梯度、代理 argmax 与硬规则一致"三者同时成立。基线已用 STE 处理量化，这个模式不是新发明。

对应验收（替换上轮第 2 条）：

1. `argmax(q_soft 行) == 硬分配的码`，三层都 100%（第三层即 `argmax(q_sinkhorn)`）；
2. 用真实 cost 构造"全部同码"与"无同码"，`L_closure` 分别 < 0.05 与 > 0.95（STE 下这是可达的）；
3. 梯度存活：零梯度行占比 **0%**，且 `|dL/dC|` 的中位数 ≥ 1e-3；
4. 记录每层 `q_soft` 的 `max prob` 与下溢条目占比，作为"代理没有饱和"的证据。

其余状态不变（R2 未绑定描述表 SHA、§19b0 未清、0 提交、无运行）。关闭条件：完整 Stage2 **0/1**、三码碰撞 < 0.40 **0/1**、完整 Stage3 与结果验证 **0/1**。
===== 2026-11-10T05:42:54Z =====
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

