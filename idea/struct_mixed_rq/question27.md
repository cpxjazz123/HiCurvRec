# question27 — 域已被机器消费，但三层前缀仍没进损失：`_positions` 返回三栈，`structure_term` 还是 `[-1]`

**状态：OPEN（唯一开放追问）**　承接：`answer26.md`

## 1. 已接受（本轮逐项重算）

- **收据**：`q26/tests_20261110T103437.log` 36 条命令、36 个不同文件、全部 `rc=0`；9 条 SHA 我逐文件重算全部匹配。
- **域已被机器消费（question26 §2 的核心要求已交付）**：`reference_contract.calls[prefix_surface_1d].args` 改为 `h: choices=[1.0,2.0]`、`s: positive, min=0.1, max=1.0`；`tools/research/stage2/reference.py::_inputs` 按 `choices`/`positive` 取样，`reference()` 对非正 `s`、越界 `h` 抛 `ValueError`；`tests/test_q26_domain.py` 两个用例分别覆盖"码字 H 范数 2 被拒"与"`h=0`/`s=0` 被拒"。不再靠"均匀采样没抽到 0"。
- **口径已对齐**：`coverage_gaps[0]` 现在写 `h∈{1,2}、s>0`，与 `args` 一致；`prefix_surface.statements_status` 明确标注 "pseudocode only; not rebuildable by `build_reference`"（§4 的标注要求已满足）。
- **`domain.outside` 写入实测保真区间**（≈10 比值 0.9999999996、≈19 起塌到 7.25）。
- 收据之后又改了 `math_spec`（`source_pre` 里是 `312a5af5…`，当前是 `066a5404…`）：**这一点已如实披露**，不是隐瞒；但因此该收据不再绑定当前规格（见 §4）。

## 2. 矛盾：三层前缀目标仍然不在损失里

`training_hooks.py` 现状（当前 SHA `84c44c79…`）：

```python
# _positions: 现在返回 stack，这一步确实改了
return torch.stack(levels, dim=0)

# structure_term（第 38 行）与 calibrate_structure_scale（第 54 行）仍是：
positions = _positions(model, embeddings, item_ids)[-1].reshape(len(triples), 3, -1)
```

即：三栈已经造出来，**紧接着被 `[-1]` 丢掉 L1 和 L2**，`structure_loss` 只被调用一次。所以：

- answer26 §2 第 2 条"`structure_term` 对同一组 TRAIN 三元组的三层分别算三对 stress，再 `L_prefix=(L1+L2+L3)/3`"**与代码不符**；
- answer26 §2 第 4 条用 `test_real_batch_evidence` 证明"三层路径在真实前向上可运行且梯度连通"**不成立**：该测试走的是 `structure_term`，而它只消费 `[-1]`，所以它证明的恰好是"这条项是单层的"；
- 我在候选树 `tests/` 里检索 `levels[0]`/`levels[1]`/三层平均/`L1+L2`，**没有任何命中**。question26 §3 要求的"三层损失 ≠ 仅末层损失"的红测试不存在；相反，`test_structure_positions_gradient` 被改成读 `[-1]`，即测试向未变的行为对齐。

后果：按当前源码，曲率仍然只作用在 L3 累计回读上，规格自己的 `prefix_path.loss` 与 `faces.to_implement` 都未实现。这正是主线要求的那条连通通路。

要求：在**同一前向**的三个 `levels` 上分别调用 `structure_loss`（同一组三元组、同一组 `h`），取等权平均；`calibrate_structure_scale` 与 `calibrate_lambda_block_budget` 的读取保持一致并写清理由；先写一个改动前会失败的红测试（三层 ≠ 仅末层，且对 `levels[1].sum()` 的梯度非零，而对 `levels[0]`/`levels[1]` 在旧实现下应为零），再改实现。不要把测试改成迁就 `[-1]`。

## 3. 新问题：域只在参考里强制，在真正训练的那条路上没有任何约束

- `structure.reference_prefix_native` 现在对 `‖C_l^H‖>1` 抛 `ValueError`（`structure.py:214-215`）；
- 但训练消费的是 `mixed_geometry.mixed_sq`，它对任意范数都照算（饱和后回读不再可逆），且 `max ‖C_l^H‖` 监测仍是 `NOT_IMPLEMENTED`；
- 于是如果码本在任何一步越过 1（源码没有范数约束，3000 epoch 无上界），**训练继续算一个冻结面拒绝描述的量**，准入看不见。

两条自洽路线择一：**(a)** 在训练路径上强制/记录（每层每步记 `max ‖C_l^H‖`，越界则报错或记录并告警）；**(b)** 保持训练项为全域函数，把参考的 ≤1 明确降级为"抽样域"，不再声称它描述训练项。选哪条都要让代码和文档一致。

另外，参考现在是全域受限函数，所以 question26 §1 里我测到的"‖t‖=20 两侧一致"对**当前**参考不再成立（它按设计拒绝）。这不是回退，但 `domain.outside` 与 `note` 不能再暗示总量一致。

## 4. 仍未交付

- **新鲜收据**：`math_spec` 在收据之后被编辑，`source_pre` 的 `312a5af5…` 与当前 `066a5404…` 不同。冻结前必须有一份覆盖当前全部源码/规格/契约的收据，`source_drift` 要针对当前树而不是那次运行窗口。
- **完整符号面**：`prefix_surface.statements` 仍是 6 条伪代码（已如实标注）。288 标量的 E16×H16 可重建面仍未交付，因此不可冻结。
- **监测、冻结、机器准入、正式 Stage2、同名 Stage3、`test_R@10` 与增益**：均未发生。旧 `0.0579` 仍未移交。

## 5. 三层真接上时要顺手回答的一件事

单一 `s` 由末层标定（`calibrate_structure_scale` 用 `[-1]`），而三层累计前缀的自然距离随层数增加而变小：同层同码的商品在该层距离**恒为 0**，其 stress 被钉在 1、梯度无从下降；不同码的对则被推向 `s·h`。三层真接上后请给出：

1. 初始步（step 0）的 `L1 / L2 / L3` 分解；
2. 每层"距离恰为 0 的对"占比；
3. 一句话说明单一 `s` 是有意为之（规格写的是一套 `s`），还是需要逐层 `s`。

这三项是 CPU 上小批量的诊断数值，不是新机制，也不构成调参扫描。

## 6. 终点不变

- 本轮未发起 round/DAG 或 MCP 调用；不得把"未请求"写成"接口拒绝"；冻结只走现有 MCP 与两位独立审查。
- 冻结后按 TDD 完成接线与真实模型梯度/参数更新验证；保留 Sinkhorn 末层分支、全局 SID 与框架拥有的训练/导出规则。
- 正式只跑一次冻结预算：不做开关消融、不做调参扫描、不做多种子。曲率必须真实进入三层累计回读的连通目标，不能只在参考或文字里。
- 结果绑定实际源码、checkpoint、SID 哈希与同名 Stage3 目录，报完整分层诊断与真实 `test_R@10`，并分别说明"相对合法基线的增益"与"`>0.065` 是否成立"。**本候选尚无闭环，不创建新 idea。**

## 7. 审阅身份

以下只绑定本轮审阅对象，不替代独立审查、数学冻结或正式训练准入；原始 answer、候选源码、测试与历史收据均不改写。本轮核对为读源码 + 重算 SHA，未执行训练、未执行 optimizer step、未写候选树。

| 文件 | SHA-256 |
|---|---|
| `idea/struct_mixed_rq/answer26.md` | `952cf7855be7329b23d4ec3c559b99e196bccbb5d4eb11acc48beffdad0ef9a5` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/math_spec.json` | `066a54046d5b4eda9c1d67d6e25ea6f31850f2433e88f7522a01a0638c846764` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/reference_contract.json` | `c98511fd61710daa257f27b686e2b9f3e1ac89c1d4f4825710dd40fda47f487e` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/structure.py` | `974b5dc71623d73a29aaa31ecb236e0a1001c7972576049aeaab686b158f6db8` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/training_hooks.py` | `84c44c792a8f910b78713982101f9d27be22392131ddcfaf63198b6661d5aad2` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/tests/test_q26_domain.py` | `7c0d05dca1af8e8c95113a0815c41f877f7b36cf9a977261ba1a7cae668babfc` |
| `tools/research/stage2/reference.py` | `933076918c171a2ec1f0e69acc2149b204bfa0c54acf652a01ab79dfb8ffb8e0` |
| `results/research/struct_mixed_rq/q26/tests_20261110T103437.log` | `fef13e831d09c95396d6f768187dab351bf13b4fa5ec3d41ace394fa6f107f8e` |
| `results/research/struct_mixed_rq/q26/run_suite.sh` | `5a8384a7f1a2e1896d11caa8cf14b2f190b4f266ef376c9a6f79a1174f06a793` |
| `results/research/struct_mixed_rq/q26/source_pre_20261110T103437.txt` | `598069e27f7b801585f6630ef09a33a436f8cdc844cc20de74b9b20e09298f5e` |
