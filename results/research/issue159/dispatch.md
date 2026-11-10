# #159 派发记录

- 预派发门：1（占用 256/256 码、最大码 0.39%）、2（保真 0.965/0.828/0.915 ≤ 1.01）、3（取整 9.5 ms ≤ 50 ms）、5（尺度不变性，损失级测试）、6（`c=1` 1.8381 对 `c=0` 1.8116，差 0.0264 > 1e-6）、7（`STRUCTURE_OFF` 逐位一致；**训练/导出一级 1.0、二级 1.0、三级 1.0**）、8（200 步护栏全过）均通过；第 4 条的字面比值是分母为负的无定义情形（良定义形式 `R ≥ 2·max(N,0)` 成立）；第 9 条的红是与模块同一步写出，已如实标注。
- 派发：`stage2_start(source_dir="TIGER_plan_rounded_rq")` → worker 1422200；真实入口一 epoch 冒烟 **SMOKE_PASS**：24 步、8 个梯度范数全正（2.81/3.43/0.0026/5.52/0.0033/0.0035/4.14/0.0041）、`dead_terms=[]`、`nonfinite_terms=[]`、导出 SID 唯一。
- 配方：seed 42、3000 epoch、全局 batch 4096、四卡 50202；产物 `results/stage2_RQ-VAE/TIGER_plan_rounded_rq/`。

## Stage3 就绪检查（Stage2 仍在跑时预置）

- 入口：`stage3_start(item_sids="results/stage2_RQ-VAE/TIGER_plan_rounded_rq/item_sids.json")`；路径必须在 `results/stage2_RQ-VAE/` 下，变体名由其所在目录推导 → 目录名 `TIGER_plan_rounded_rq` 已与 Stage2 臂目录同名，满足 §64b(a)。
- 可见卡 ≥4 时走冻结四卡 DDP（全局 train batch 16384），<4 时走单卡分支（全局 train batch 4096）——当前 4 张卡全空，符合四卡分支。
- 前置：§63 先记（`post_stage2.py` 已脚本化），入口碰撞 < 0.40 才允许启动。
- Stage3 跑完后按 §6b 把四件（Stage2 源码树、`item_sids.json`、`rqvae_best.pth`、完整 recall 块）归档进 `genrec/<NN>/`，SHA 记 `r2_ledger.json`。

## §6b 归档就绪

- 归档脚本 `results/research/issue139/upload_stage3.py` 存在，凭证从仓库根 `r2_apikey` 读取（已 gitignore，日志不得包含其内容），目标桶 `genrec/<NN>/`，四件齐全才上传。
- 现存账本 `results/research/issue139/r2_ledger.json` 已有 68 条记录；本轮的结果届时按新编号追加，SHA 一并写入。

## 导出面核对（消费路径）

训练终点导出 SID 走的是：

```python
raw_tokens = raw_module.get_indices(all_embeddings.to(device), infer_use_sk=False)
```

`get_indices` → `encode_tokens`，而本臂在 `encode_tokens` 里对末级**无条件**走 `layer._indices(cost, infer_use_sk=True)`——即计划 + 计划质量取整。因此：

- **导出用的就是取整规则**（不是 argmin），且在整个 24587 商品集上**一次调用、确定性执行**，与 Issue 第 4 条的要求一致；
- 训练中途用于碰撞监控的同一条路径也走取整，故监控口径与最终导出同源；
- 代价：导出时要在全量商品上跑一次末级 Sinkhorn + 取整（可接受，且已完成于冒烟）。

这一条在 gate 7 的合成 batch 之外，验证了**消费面（Stage3 读的那个文件）**确实带取整——即 #157 那类"接线对但导出仍走旧规则"的坑在本臂不成立。
