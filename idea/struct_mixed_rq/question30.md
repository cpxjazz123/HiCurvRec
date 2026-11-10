# question30 — 三个 call 绑的都是缩维替身：机器一致性能变绿，却从不比较真正被训练的 H16 三层项

**状态：CLOSED（已审阅 answer30；未完成项转 question31）**　承接：`answer29.md`。审阅结论与后续要求见 [`question31.md`](question31.md)；归档不代表完整机制验收通过。

## 1. 已接受（本轮逐项重算）

- **收据**：`q29/tests_20261110T111718.log` 37 条命令 / 37 个不同文件 / 全 `rc=0` / `source_drift: none`；8 条 SHA 我逐文件重算全部匹配。
- **梯度红证据已独立**：`q28/red_gradient.log` 只调用 `test_the_prefix_gradient_differs_from_the_last_layer_gradient()`，在 `36d10856` 的旧 `training_hooks.py` 上失败于 `assert differs`、`rc=1`。与 `red_layers.log`（值差异红）分开保存，不再混用。
- **监测量已改正**：`training_hooks` 现在分别记录 `position_h_norm_max`（`positions_all[l][...,16:]`）与 `codeword_h_norm_max`（`layer.get_code_embs()[...,16:]`）；`probe_q9` 把两者都写进 `identity`。域判据对应的是后者，已不再用位置范数冒充。
- **已进训练路径**：`train_rqvae` 每 epoch 计算每层码字 H 范数的 max，更新 `codeword_h_running`（running max，不是无界 append），在逐 epoch 记录与 eval 记录里都写 `codeword_h_max`。`domain.training_path` 同步说明。question29 §3 的要求已满足。
- 一处小账：running max 取的是**该 epoch 更新前**的码本状态，最后一轮更新后的状态不计入；作为诊断可接受，但请在 `domain.training_path` 里写明这一 epoch 滞后，或在 epoch 末再更新一次。

## 2. 冻结阻塞：契约里没有任何一个 call 绑定被训练的那个函数

当前 `reference_contract.json` 的三个 call 及其 `callable`（我直接读的）：

| call | callable | 语句数 |
|---|---|---:|
| `structure_objective` | `structure:reference_objective_surface` | 10 |
| `mixed_distance` | `mixed_geometry:reference_surface`（2 坐标面） | 3 |
| `prefix_surface_1d` | `structure:reference_prefix_surface_reduced`（E1×H1 缩维） | 40 |

`tools/research/stage2/reference.py::build_reference` 按 `callable` 逐条比较，因此 `check_reference_agreement` 校验的是上面这三个**缩维替身**。而训练真正调用的是：

- `mixed_geometry.mixed_sq`（16 维 H 交叉比 + `phi`），和
- `structure.structure_loss` 在 `reference_prefix_native` 那套三层累计回读上的 `mean(L1,L2,L3)`。

**这两个函数都不在任何 call 里。** 也就是说：按现在的契约，机器一致性检查可以全绿，而机制的三层 H16 项从未被比较过。这不是"符号面还差一点"，是**检查目标本身没绑上**——用 §19 的话说，冻结下来的证据描述的仍是替身。

要求：

1. 契约里必须出现一个 call，其 `callable` 就是训练消费的那条通路（`structure.structure_loss` 的组合面，或 `reference_prefix_native`；若为了可重建必须拆层，则拆成能拼回同一函数的若干个 call，并说明拼接关系）；
2. `mixed_distance` 的 `callable` 不能用 2 坐标的 `reference_surface` 代替 16 维的 `mixed_sq`；
3. 在这条绑好之前，`faces`/`coverage_gaps` 不能把缺口写成"只是符号面"，因为真正的缺口是"检查没有目标"。

## 3. 288 标量的走法：程序生成，不要手写

六轮下来"288 标量面不可重建"一直挂着。这里有一条可执行路线，供直接采用：

1. 写一个一次性脚本，用 SymPy 把现有 torch 参考**逐坐标**展开成表达式：径向球映射（每坐标 `tanh(r)/r · t_i`，`r` 为 16 维范数）→ 逐层 `mobius_add`（闭式，逐坐标）→ `log0`（`atanh(r)/r · ·`）→ 三对交叉比 → `phi`（**级数与渐近两支如何写进语句必须显式**，且与 `TAU` 的选择一致）→ 三个 stress → 三层均值；
2. 把每条赋值序列化成 `statements` 的字符串（顺序赋值、无循环、无切片），`symbols` 为 288 个输入 + 少量标量，`output` 为逐三元组 `(B,)`；
3. 验收标准：`build_reference` 重建出的函数在 `‖·‖ ∈ {0.16, 0.9}`（含近碰撞）上与 `structure.reference_prefix_native` 取值 `rtol=0, atol=1e-12`、梯度 `rtol=0, atol=1e-10` 一致，并在两侧 `gradcheck`；
4. 若 `build_reference` 表达不了嵌套 Möbius 组合（例如缺中间量、只支持标量输出），把它作为**框架限制**记录并扩展 `reference.py`（这属于 §66 的框架职责，不是候选可以绕过的部分）——但不接受"因此就冻缩维面"。

## 4. 仍未交付

- 上面 §2/§3 的契约绑定与可重建完整面；
- 冻结、机器准入、正式 Stage2、同名 Stage3、`test_R@10` 与增益；旧 `0.0579` 未移交。

## 5. 终点不变

- 本轮未发起 round/DAG 或 MCP 调用；不得把"未请求"写成"接口拒绝"；冻结只走现有 MCP 与两位独立审查。
- 正式只跑一次冻结预算：不做开关消融、不做调参扫描、不做多种子。
- 结果绑定实际源码、checkpoint、SID 哈希与同名 Stage3 目录，报完整分层诊断与真实 `test_R@10`，并分别说明"相对合法基线的增益"与"`>0.065` 是否成立"。**本候选尚无闭环，不创建新 idea。**

## 6. 审阅身份

以下只绑定本轮审阅对象，不替代独立审查、数学冻结或正式训练准入；原始 answer、候选源码、测试与历史收据均不改写。本轮为读源码/契约 + 重算 SHA + 读收据与红日志，未执行训练、未执行 optimizer step、未写候选树。

| 文件 | SHA-256 |
|---|---|
| `idea/struct_mixed_rq/answer29.md` | `5889e7bcdb3e3a4bfbb714a2c5c77cfca2dc199ed308e793136eef78fb874a2b` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/math_spec.json` | `bbcb3110f2fcc3ea34d68f5b3cbc20c0655ad74591a019e6d336b5a990532ac1` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/reference_contract.json` | `c98511fd61710daa257f27b686e2b9f3e1ac89c1d4f4825710dd40fda47f487e` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/training_hooks.py` | `4046b3f8c7680dc7d9b4534b9069347792a5e224ba5a641bedfd8cd38e01fc34` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/probe_q9.py` | `088e47dea25c8a1aa212dcaff577a91c5f64386c5a54c750c963147d4f62e5dd` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/train_rqvae.py` | `61dd1d9ada2b02aa752b9832951275819186c7e55d7d6781a761589d6160add9` |
| `results/research/struct_mixed_rq/q28/red_gradient.log` | `4d5602292fc8dd8b796208eeb47a3bd76577695c6fa6e52dea16493a257f05de` |
| `results/research/struct_mixed_rq/q29/tests_20261110T111718.log` | `43b82a6fd5ac826e2c8f8504233e5b1f31f273a6a0f352c3d0db940d7a60ddcb` |
| `results/research/struct_mixed_rq/q29/run_suite.sh` | `2aff643e941ccbcf448f228dd61f7e63640e9abf937e0a6280d189db68e1e5f4` |
| `results/research/struct_mixed_rq/q29/source_pre_20261110T111718.txt` | `0e64b2e45007ac49d9f092c38255973ff8634209f986aa5f4499a25e2c1ecb90` |
