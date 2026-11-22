## 两臂源码差异审计:除对照开关外,机制代码逐行相同

§19b 要求候选树之间不产生机制漂移。治疗臂与打乱对照臂是从同一棵树复制的,本轮把差异逐文件核对完:

### `train_rqvae.py`

`diff` 去掉 `ARM_NAME` 那一行后 **完全为空**。两臂的唯一差别是结果目录名。

### `losses/hard_prefix.py`

差异只有三处,都是对照臂需要的接线,不触碰机制本身:

| 差异 | 内容 | 是否影响机制 |
|---|---|---|
| `ARM_CONTROL = True` | 对照臂置位;治疗臂无此常量 | 否,只选 bank 来源 |
| `RoleIndex.shuffled()` | 整实例置换(不是逐角色独立抽样) | 否,只在对照臂被调用 |
| `present(..., source=)` | 按 `bank_source` 取冻结或置换 bank | 否,默认值 `"frozen"` 与治疗臂行为一致 |

`prefix_margin` / `_temperature` / `soft_prefix_share` / `hard_gate` 等**机制函数在两臂之间逐字节相同**。
因此两臂的差异只可能是"给结构哪条实例算 Δ",这正是 C33 要隔离的变量。

### 一处必须说明的实现细节

`shuffled()` 最初写成对三个角色槽各自独立抽 `rng.permutation`,运行时被 C9 的隔离断言拦下:
`chain: 3 loss pairs also sit in an evaluation bank`。原因是独立抽样会把 A、B、C 重新组合成
训练集中不存在的新三元组,其中三条的商品对落在评估 bank 里。改成**整实例置换**后,每个三元组的
三个商品对原样保留,隔离断言通过。这条是测试真的在起作用,不是噪音。

证据:两臂 `diff`(命令见评论正文),`stage2_RQ-VAE/TIGER_graphlet_prefix_shuffled_rq/tests/test_shuffled_control.py`(5 个测试)。
