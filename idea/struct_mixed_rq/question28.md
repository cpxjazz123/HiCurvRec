# question28 — 三层真进损失了，但规格仍写着"只有末层在跑"：冻结文档与代码相反；第二个"红测试"在旧实现上也是绿的

**状态：CLOSED（已审阅 answer28；未完成项转 question29）**　承接：`answer27.md`。审阅结论与后续要求见 [`question29.md`](question29.md)；归档不代表完整机制验收通过。

## 1. 已接受（本轮逐项重算）

- **三层确实进了损失**（读源码，不读声明）：`training_hooks.structure_term` 现在是
  `positions_all = _positions(...).reshape(3, T, 3, -1)` → `losses = [structure_loss(positions_all[l], h, ...) for l in range(3)]` → `torch.stack(losses).mean()`。同一前向、同一组三元组、同一 `h`、同一 `s`。question27 §2 的核心要求已交付。
- **λ 与训练项同源**：`calibrate_lambda_block_budget` 内部调用的是 `structure_term(...)`，不是末层 `structure_loss`；块预算仍为 encoder / codebooks 两块取 min，degenerate 仍 fail-closed。所以 λ 是按真正被训练的项推出来的，这条不再追问。
- **收据现在绑定当前树**：`q27/tests_20261110T104625.log` 37 条命令 / 37 个文件 / 全 `rc=0` / `source_drift: none`；`source_pre` 与 `source_post` 里 `math_spec.json`（`38b6a607…`）和 `training_hooks.py`（`7a5d70bf…`）都与磁盘当前值相同。q26 的"收据后被编辑"问题已消除。
- **域走了路线 (b) 并写清**：`math_spec.continuous_surface.prefix_surface.domain.training_path` 明确"训练消费 `mixed_sq` 全域函数，`≤1` 只是参考抽样域"，并附 step-0 数值。不再暗示总量一致。
- 9 条 SHA 我逐文件重算全部匹配。

## 2. 冻结阻塞：规格自己仍写着"训练只跑末层"

`math_spec.continuous_surface.prefix_surface` 当前仍有三句与代码相反的话：

| 字段 | 现有文字 | 与代码的关系 |
|---|---|---|
| `faces.implemented` | "last-layer structure term on quantised positions … **this is the only term the trainer currently applies**" | 假：现在跑的是三层均值 |
| `faces.to_implement` | "wiring: **`_positions` still returns `levels[-1]`**; the prefix path must run through the same quantisation and readback, not a copy" | 假：`_positions` 返回三栈，消费端用全部三层 |
| `coverage_gaps[2]` | "this prefix objective is **NOT_IMPLEMENTED in the trainer; only the last-layer structure term runs**" | 假 |

这不是措辞问题：这三句正是冻结/审查时会被读到的"机制到底在不在训练里"的权威陈述。按现在的状态冻结，冻结下来的规格描述的是**另一个 trainer**，而 §83 的耦合证据、`prefix_path.loss` 与真实运行之间就对不上。请在**同一次提交**里改掉这三处，并让新收据覆盖改动后的 `math_spec`（当前收据只覆盖 `38b6a607…`，改完即失效）。

同时请检查 `training_integration.prefix_objective` 是否也仍停留在"将来式"，以及 `GEOMETRY.md`/`REMAINING.md` 是否有同样的旧陈述。

## 3. `_positions` 的 docstring 仍是旧的，且重复了已被驳回的理由

当前函数体返回 `torch.stack(levels, dim=0)`，但 docstring 仍写：

> "levels[-1] is the native readback plus the tangent straight-through … Adding codes_to_positions on top of it would add a second codebook path and silently change the mechanism."

前半句描述的是旧返回值；后半句正是 question27 §3 指出不成立的那条理由（三层就在 `levels` 里，不需要第二路径）。现在它读起来像"本函数只服务末层"，与代码相反。请重写为实际语义：返回三层累计回读 `q_1,q_2,q_3`，STE 落在 `x`，码本只吃原生回读 Jacobian。

## 4. 第二个"红测试"在旧实现上也是绿的

`tests/test_q27_layers.py` 两个用例：

- `test_three_levels_differ_from_the_last_level_only`：断言 `term != last`。**这是真回归测试**——旧实现 `term == last`，会红。接受。
- `test_the_second_level_receives_a_gradient`：只断言 `len(captured) == 3`（`_positions` 从 q26 起就返回三栈，旧实现同样成立）、`any(g is not None and max>0)`（仅末层就足以让 encoder/码本拿到非零梯度）、`term.requires_grad`。**这三条在"`structure_term` 仍取 `[-1]`"的旧代码上全部通过**，因此它不是红测试；名字说"第二层收到梯度"，断言里没有任何一条检查 L2 的贡献。

请改成能区分的那一条：例如对 `losses[1].sum()` 求 `autograd.grad` 到码本参数、断言非零；或断言 `grad(term) != grad(last)`（同参数、同三元组）。否则把用例名改成它实际检查的内容。

## 5. 仍未交付

- **288 标量可重建符号面**：`prefix_surface.statements` 仍是 6 条伪代码（已如实标 `statements_status`），`build_reference` 无法重建 → 不可冻结。
- **训练路径监测**：每层 `max ‖C_l^H‖` 仍 `NOT_IMPLEMENTED`。路线 (b) 已声明"训练不拒绝"，那么这条监测就是唯一能发现"回读越过 ~10 后不再可逆"的手段，请至少在日志里记录，按 §19 只作诊断。
- **step-0 诊断没有落盘**：answer27 §4 的 `s=0.04255551099777222`、L1/L2/L3 `1.4636/1.7739/2.0399`、零距离对 `2.06%/0.17%/0%` 只在正文里；`results/research/struct_mixed_rq/q27/` 里只有 `run_suite.sh`、两份 source 清单与测试日志，没有这份数值的 artefact。我因此**没有**复算它们。请把这份诊断写成该目录下的 JSON（含 seed、三元组数、`s`、三层损失、零距离对计数），让它可审计。
- **冻结、机器准入、正式 Stage2、同名 Stage3、`test_R@10` 与增益**：均未发生。旧 `0.0579` 未移交。

## 6. 终点不变

- 本轮未发起 round/DAG 或 MCP 调用；不得把"未请求"写成"接口拒绝"；冻结只走现有 MCP 与两位独立审查。
- 冻结后按 TDD 完成剩余接线与真实模型梯度/参数更新验证；保留 Sinkhorn 末层分支、全局 SID 与框架拥有的训练/导出规则。
- 正式只跑一次冻结预算：不做开关消融、不做调参扫描、不做多种子。曲率必须真实进入三层累计回读的连通目标。
- 结果绑定实际源码、checkpoint、SID 哈希与同名 Stage3 目录，报完整分层诊断与真实 `test_R@10`，并分别说明"相对合法基线的增益"与"`>0.065` 是否成立"。**本候选尚无闭环，不创建新 idea。**

## 7. 审阅身份

以下只绑定本轮审阅对象，不替代独立审查、数学冻结或正式训练准入；原始 answer、候选源码、测试与历史收据均不改写。本轮为读源码 + 重算 SHA + 读收据，未执行训练、未执行 optimizer step、未写候选树，也未复算 §4 的 step-0 数值。

| 文件 | SHA-256 |
|---|---|
| `idea/struct_mixed_rq/answer27.md` | `8993b981161b53b9cf34e20f1d94c4a32a916d793bcda3549b9eea180f4cf1ff` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/math_spec.json` | `38b6a607147345c4bcf36b9a7f4ca31d64ccdd8540dc40c3b7e23610529eb12a` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/reference_contract.json` | `c98511fd61710daa257f27b686e2b9f3e1ac89c1d4f4825710dd40fda47f487e` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/structure.py` | `974b5dc71623d73a29aaa31ecb236e0a1001c7972576049aeaab686b158f6db8` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/training_hooks.py` | `7a5d70bf1a0d8eba8304198836ca651fe52582c0568604e47fcb82ef271459f3` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/tests/test_q27_layers.py` | `f8b913ed6335311e319b6ee17a5a8412e322b979b39573f565e002e269331858` |
| `tools/research/stage2/reference.py` | `933076918c171a2ec1f0e69acc2149b204bfa0c54acf652a01ab79dfb8ffb8e0` |
| `results/research/struct_mixed_rq/q27/tests_20261110T104625.log` | `690f09bd63afe7441e936a2da0f4c85a4f2b6fe31c01b8eb0cd3af0ed05c4980` |
| `results/research/struct_mixed_rq/q27/run_suite.sh` | `d3b32ff31f46346ae0191c2950c4944b615f9254af1a8028e7366152545e138a` |
| `results/research/struct_mixed_rq/q27/source_pre_20261110T104625.txt` | `418933b1ac25057ca82e8dc78062339317c3894a251377da1a5d4d7bed0b6b5d` |
