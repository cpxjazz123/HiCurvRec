# answer26 — 域被机器消费，结构项改为同一前向的三层前缀
**状态:三层前缀已接入结构项;完整 SymPy 面/冻结/正式训练仍未完成;训练/收益 NOT_RUN**

承接 `answer25.md`。接受 question26 §1 的三点差分(0.16/0.9/20 两侧一致)与收据核对。

## 1. h/s 改为构造上合法,域由参考拒绝
- `reference_contract.calls[prefix_surface_1d].args`:`h` 改为 `choices=[1.0,2.0]`(真实目标值,无符号 box),`s` 改为 `positive=true, min=0.1, max=1.0`;
- `tools/research/stage2/reference.py::_inputs` 读取 `choices`/`positive` 生成参数,不再产生 `h=0` 或 `s≤0`;
- `reference()` 对 `positive` 非正、`choices` 越界抛 `ValueError`;`structure.reference_prefix_native` 对 `‖C_l^H‖>1` 抛 `ValueError`;
- 新增 `tests/test_q26_domain.py`:`‖C^H‖=2`、`h` 含 0、`s=0` 三种输入均被拒绝(先红:重建的参考原本直接求值);
- `coverage_gaps[0]` 由“`1+tanh(u)`”改为与 `args` 一致的 “h∈{1,2}、s>0”,两份材料口径一致;
- `domain.outside` 写入实测保真区间:`‖t‖≈10` 比值 0.9999999996、`≈19` 起塌到 7.25。

## 2. 结构项改为同一前向的三层累计前缀
- `training_hooks._positions` 现在返回 `forward_with_codes` 的 `levels[0..2]` 堆叠(同一次量化、同一次回读,无第二条 gather);
- `structure_term` 对同一组 TRAIN 三元组的三层分别算三对 stress,再 `L_prefix=(L1+L2+L3)/3`;
- `calibrate_structure_scale` 仍用末层位置(`[-1]`)做 s 标定,与既定规则一致;
- `test_structure_positions_gradient` 的对照读取改为末层;
- `test_real_batch_evidence`(真实模型/真实 batch)通过,说明三层路径在真实前向上可运行且梯度连通。

## 3. 仍未交付
- `prefix_surface.statements` 仍是 6 条伪代码,`build_reference` 无法重建;已加 `statements_status` 明确标注“非可重建语句”;
- 完整 E16×H16 的 288 标量 SymPy 面未交付;
- 每层 `max ‖C_l^H‖` 训练诊断未实现;
- 冻结/准入/正式 Stage2/Stage3 均未运行;收益无结论,旧 `0.0579` 未移交。

## 4. 收据与说明
`q26/tests_20261110T103437.log`:36 条命令、36 个不同文件、**36 pass / 0 fail**,runner `rc=0`,
`source_drift: none`。**注意**:收据生成后我又在 `math_spec` 里补了 `statements_status` 字段,该编辑不在收据的前后清单内;下方 `math_spec` 的 SHA 是编辑后的当前值,收据覆盖的是编辑前版本,如实分开记录。

## 5. 摘要(编辑后当前值)
| 文件 | SHA-256 |
|---|---|
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/math_spec.json` | `066a54046d5b4eda9c1d67d6e25ea6f31850f2433e88f7522a01a0638c846764` |
| 候选 `reference_contract.json` | `c98511fd61710daa257f27b686e2b9f3e1ac89c1d4f4825710dd40fda47f487e` |
| 候选 `structure.py` | `974b5dc71623d73a29aaa31ecb236e0a1001c7972576049aeaab686b158f6db8` |
| 候选 `training_hooks.py` | `84c44c792a8f910b78713982101f9d27be22392131ddcfaf63198b6661d5aad2` |
| 候选 `tests/test_q26_domain.py` | `7c0d05dca1af8e8c95113a0815c41f877f7b36cf9a977261ba1a7cae668babfc` |
| 框架 `tools/research/stage2/reference.py` | `933076918c171a2ec1f0e69acc2149b204bfa0c54acf652a01ab79dfb8ffb8e0` |
| 证据 `q26/tests_20261110T103437.log` | `fef13e831d09c95396d6f768187dab351bf13b4fa5ec3d41ace394fa6f107f8e` |
| 证据 `q26/run_suite.sh` | `5a8384a7f1a2e1896d11caa8cf14b2f190b4f266ef376c9a6f79a1174f06a793` |
| 证据 `q26/source_pre_20261110T103437.txt` | `598069e27f7b801585f6630ef09a33a436f8cdc844cc20de74b9b20e09298f5e` |
