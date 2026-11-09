# answer27 — 三层前缀真正进入结构项，域的两条路线选定并写清
**状态:三层结构项已接入;符号面/监测/冻结/正式训练仍未完成;训练/收益 NOT_RUN**

承接 `answer26.md`。接受 question27 §1 的核对，并更正 answer26 §2 与代码不符的表述。

## 1. 更正 answer26 的错误陈述
answer26 §2 声称 `structure_term` 已做三层平均、`test_real_batch_evidence` 证明三层路径。实际当时 `structure_term` 仍写 `[-1]`，三栈被丢掉；该测试证明的是单层行为。该陈述作废，不再引用。

## 2. 三层前缀已接入(先红后绿)
新增 `tests/test_q27_layers.py`:
- **红**:`test_three_levels_differ_from_the_last_level_only` 在旧实现上断言失败(`term` 与仅 `[-1]` 的结果全等);`test_the_second_level_receives_a_gradient` 同时记录三段 `levels`。
- **绿**(修复后):`structure_term` 用 `_positions` 的 `(3, T, 3, 32)` 逐层调用 `structure_loss`(同一组三元组、同一组 `h`、同一 `s`),返回 `mean(L1,L2,L3)`;测试断言 `term != last`、捕获到 3 个 `levels`、`term.requires_grad` 且 encoder/码本参数收到非零梯度。
- `calibrate_structure_scale` 仍用末层 `[-1]` 标定 `s`,理由写入规格:规格只声明一个共享 `s`,标定定义在初始化末层 h=1 对上。

## 3. 域的两条路线:选定 (b) 并写清
- `math_spec.prefix_surface.domain.training_path` 明确:**训练路径消费 `mixed_geometry.mixed_sq` 的全域函数**,`‖C^H‖≤1` 只是参考的**抽样域**,不是运行时约束;
- 参考保持拒绝(`reference_prefix_native` 对 `>1` 抛错),训练不拒绝,二者分工写清,不再暗示“总量一致”;
- 每层 `max ‖C_l^H‖` 的训练监测仍未实现(如实保留)。

## 4. 三层刚接上时的 step-0 诊断(CPU、真实 TRAIN、256 三元组、真实模型)
`s = 0.04255551099777222`(末层标定):
| 层 | 损失 | 距离恰为 0 的对 | 占比 |
|---|---:|---:|---:|
| L1 | 1.4636057615280151 | 12 / 582 | 2.06% |
| L2 | 1.7739008665084839 | 1 / 582 | 0.17% |
| L3 | 2.039872169494629 | 0 / 582 | 0% |
**单一 `s` 是有意为之**:规格只声明一个共享 `s`;同层同码的对距离恒为 0、stress 钉在 1,本次最多占 L1 的 2.06%,不影响其余对的三层平均。

## 5. 收据
`q27/tests_20261110T104625.log`:37 条命令、37 个不同文件、**37 pass / 0 fail**,runner `rc=0`,
`source_drift: none`;绑定 `*.py` + `math_spec.json` + `reference_contract.json`。该收据覆盖当前规格(此前 q26 收据后的编辑问题已消除)。

## 6. 仍未交付
- 完整 288 标量 SymPy 面(`prefix_surface.statements` 仍为伪代码,已标 `statements_status`);
- 每层 `max ‖C_l^H‖` 训练监测;
- 独立冻结、MCP 准入、正式 Stage2/Stage3 与增益;旧 `0.0579` 未移交。

## 7. 摘要(自动计算,完整 64 位)
| 文件 | SHA-256 |
|---|---|
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/math_spec.json` | `38b6a607147345c4bcf36b9a7f4ca31d64ccdd8540dc40c3b7e23610529eb12a` |
| 候选 `reference_contract.json` | `c98511fd61710daa257f27b686e2b9f3e1ac89c1d4f4825710dd40fda47f487e` |
| 候选 `structure.py` | `974b5dc71623d73a29aaa31ecb236e0a1001c7972576049aeaab686b158f6db8` |
| 候选 `training_hooks.py`(三层) | `7a5d70bf1a0d8eba8304198836ca651fe52582c0568604e47fcb82ef271459f3` |
| 候选 `tests/test_q27_layers.py` | `f8b913ed6335311e319b6ee17a5a8412e322b979b39573f565e002e269331858` |
| 框架 `tools/research/stage2/reference.py` | `933076918c171a2ec1f0e69acc2149b204bfa0c54acf652a01ab79dfb8ffb8e0` |
| 证据 `q27/tests_20261110T104625.log` | `690f09bd63afe7441e936a2da0f4c85a4f2b6fe31c01b8eb0cd3af0ed05c4980` |
| 证据 `q27/run_suite.sh` | `d3b32ff31f46346ae0191c2950c4944b615f9254af1a8028e7366152545e138a` |
| 证据 `q27/source_pre_20261110T104625.txt` | `418933b1ac25057ca82e8dc78062339317c3894a251377da1a5d4d7bed0b6b5d` |
