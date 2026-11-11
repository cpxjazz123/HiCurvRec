# #180 现状：一级选码代价模块与开关已落地（5/5 通过）；**训练循环尚未把逐行 κ 传进去**

## 已完成

- 新树 `stage2_RQ-VAE/TIGER_r48k2u_rq`（自 #179 树复制，`ARM_NAME="TIGER_r48k2u_rq"`、`ARM_VARIANT="R48K2U"`）。
- `model/r48k2_assign.py`：`level1_cost(latent, codes, kappa)` = `d_{κ_i}(proj(r_i), proj(c_k))`（逐行 κ、内部 float64、返回输入 dtype）；`CONTRAST_ONLY` 开关（关闭时**逐位**返回欧氏表达式，即 #179 行为）；`GEOMETRY_OFF` 同样退回欧氏；`set_batch_kappa/clear` 钩子。
- `model/layers.py::_distances` 挂钩：钩子里没有 κ（或行数不匹配）时**原样返回冻结表达式**；**`sinkhorn` 本体一字未改**（§19h：只换它居中的矩阵）。
- 测试 `tests/test_r48k2u_assign.py` **5/5**：`CONTRAST_ONLY` 下代价与欧氏 `torch.equal`；`BOTH_ON` 与手算逐元素一致（rtol 1e-12）；代价非恒定且可居中；`GEOMETRY_OFF` 退回欧氏且逐位相等；层钩子在设置 κ 时改变代价、清除后恢复原值。

## 未完成（阻塞点，下一轮）

训练循环 `for (batch,) in loader` 只拿到**嵌入行**，没有 item id；`TensorDataset(train_embeddings)` 且 DataLoader 有 shuffle，因此无法把行映射回 `train_ids`。要真正启用选码侧曲率，需要让数据集同时产出 id（`TensorDataset(train_embeddings, train_ids_tensor)` 并同步改三处解包：训练循环、pair loader、导出路径），或改用按批索引切分的加载器。这是结构性改动，需改动 loader 契约，故**本轮未派发**（按 Issue 第 8 条不改 τ/权重/Sinkhorn/码本，且未通过验收 1–6 不得派发）。

## loader 契约已改完，**21/21 通过**

`train_rqvae.py` 的三处改动（都在框架允许的范围内：只让行能被映射回 item id，机制常数未动）：

1. `TensorDataset(train_embeddings, train_ids_tensor)` —— 每行带上 item id。
2. 训练循环 `for batch, batch_ids in loader:`，在 `model(batch)` 前后 `set_batch_kappa(_assign_kappa(batch_ids))` / 清除（`try/finally`）。
3. 导出 `_encode_all(module, all_embeddings)`：全量行按 `arange` 取 κ 后编码，**与训练同一条规则**（验收 4）。

`_assign_kappa` 在 `CONTRAST_ONLY` 或 `GEOMETRY_OFF` 时返回 `None`，于是 `layers._distances` 原样走冻结欧氏表达式——两个开关的语义因此是"代价侧完全不介入"。

测试 **21 passed**（含 #179 的 16 项与本轮的 5 项）。
