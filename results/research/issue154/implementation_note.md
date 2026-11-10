# #154 实现前的规格缺口：硬前缀嵌入没有回到编码器的梯度

## 事实

基线 `stage2_RQ-VAE/TIGER_mixed_rq`（本臂起点）的量化路径不含 straight-through：

- `model/mixed_rq.py:187-194`：commitment 用 `picked_e.detach()`、`picked_h.detach()`，codebook 项用 `residual.detach()`，残差更新只 detach 被选码；
- `MixedRQ.forward` 返回 `readout`（`quantized_readout(codes_e, codes_h, ...)`），readout 由**被选码向量**构成，`tokens` 是 `argmin` 的硬索引。

因此，若按 Issue 第 3–4 条把「第 ℓ 级前缀嵌入」取成该商品前 ℓ 级被选码向量（训练与导出同源的那条定义），`d_ℓ(i,j)`、`s_ℓ(i,j)`、`L_gap`、`L_tri` 全部是 `embed.weight` 的硬索引函数：对编码器参数的梯度**恒为零**，两项损失只会推动码本向量，不会约束编码器。`tests/test_prefix_geometry.py` 里已证明梯度能到 `prefix_e/prefix_h` **输入**，但 trainer 交给它的会是常量张量。

## 影响

Issue 第 6 条要求测试覆盖「两项的定义域与梯度」，第 1 条要求 200 步拟合后 `L_gap`/`L_tri` 下降且末级改码 ≥10%——在硬前缀下这个梯度不存在，无法满足，也不能用「只改码本」的解释替代（那会变成与 #150–#153 同类的容量层改动）。

## 两条可实现的读法（都需要 Issue 侧确认，均不改变机制语义）

1. **软前缀（推荐）**：前缀嵌入取该级 `mixed_cost` 的 softmax 期望码向量（温度取冻结的 `SK_EPSILON` 同一量级，或不引入温度而直接对 `mixed_cost` 归一化），使 `s_ℓ` 对编码器可微，且训练/导出仍用硬 `argmin` 取码。代价：前缀嵌入是软量，与导出所用硬前缀不完全同一对象，需要在报告里并列。
2. **给 readout 加 STE**：`readout = residual + (picked - residual).detach()`，使前缀嵌入带编码器梯度。代价：改动基线读出（Issue 第 5 条声明"读出不动"），超出本 Issue 允许的改动面。

## 本轮结论

按 Issue 允许的改动面（第 5 条：代价、Sinkhorn、残差递推、读出、导出都不动），第 1 条读法是唯一不越界的选择；但它需要一个 Issue 未写明的温度/归一化选择，属于"改变机制定义"，因此不擅自实现，先留证。

## 更正（同一 Issue 内，不改上文证据）

上节把「软前缀」判为需要 Issue 未写明的选择，**过严**。重读 Issue 后：

- 第 3 条标题就是「可微前缀归属度」，并规定 `d̄_ℓ` 取同 batch 距离中位数、`τ_ℓ` 取其 MAD，「实测量的比值，不留全局可调常数」——归一化**已经指定**，不存在待定温度；
- 第 5 条的「其余不动」清单是代价函数、末级 Sinkhorn、残差递推、读出、导出、三 token 结构；**损失自己的前缀定义不在其中**。

因此正确读法是：`s_ℓ` 的前缀嵌入取该级 `mixed_cost` 的 softmax 期望码向量（权重即同一 `mixed_cost` 的归一化，不引入新常数），训练与导出仍用硬 `argmin` 取码；`d̄_ℓ/τ_ℓ` 用 batch 实测中位数/MAD。这样梯度经软权重回到编码器，且未触碰任何冻结表面。上一轮把它当作「未授权」而停手是错的，本轮记录更正，后续按此实现。

## 第二个接线缺口（实测，不是推测）

`train_rqvae.py` 的训练循环是 `for (batch,) in loader:`——**batch 里没有商品 id**，只有 embedding 行。本臂两项损失要按「batch 内同时出现的边/三角形」构造（`high_edges`/`low_edges` 与 3118 个全高三角形都是节点对），没有 id 就无法把行对应到图的节点，也就无法选出边。

处理：不猜、不按行号硬配。两条可行路径都写在这里，实现时二选一即可——
1. 让候选树的 `dataset.py` 额外产出 `item_ids`，循环改成 `for batch, ids in loader:`（改动只在候选树，不碰冻结超参）；
2. 训练前一次性建「embedding 行 → 商品 id」的精确映射（行内容哈希），循环里按行查 id，不改 dataset。

在 id 通路打通之前，损失**没有**接进训练；模型侧已经具备可微前缀（`soft_prefix` + `compose_prefix`，`last_prefixes` 记录），并由 `probe_gradient.py` 单独测量两项对编码器的梯度，以证明这个读法确实把梯度带回了编码器，而不是又一次只动码本。

## 第三个缺口（实测中，未解决）

`probe_gradient.py` 在真实 batch（2048 行、冻结边表）上把 `L_gap` 反传时抛
`element 0 of tensors does not require grad and does not have a grad_fn`——即**软前缀目前仍未带梯度**。

已排除：`MixedRQ.forward`、`MixedRQVAE.forward`、`split` 都没有 `@torch.no_grad()`；`mobius_add` 的实现是普通张量运算；`layer.distances` 是 no-grad 表面，所以已改成损失自己用 `mixed_distance` 重算同一代价（`permixed` 广播版，代码里注释说明"冻结代价函数本身不动"）。

下一步的确定诊断（一次就能定位）：在同一 batch 上按顺序打印
`model.encoder(batch).requires_grad` → `expmap0(...).requires_grad` → `pre_cost.requires_grad` → `soft_prefix(...)[1].requires_grad`，第一个为 `False` 的那一环就是断点；同时检查 `initialize_tiger_weights` 之后 encoder 参数是否 `requires_grad=True`（若初始化被包在 `no_grad` 里且顺手把参数设成了常量，这是最可能的原因）。
