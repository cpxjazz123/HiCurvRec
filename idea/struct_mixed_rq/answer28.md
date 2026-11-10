# answer28 — 规格改成三层、接线测试真正可区分、诊断落盘
**状态:三层结构项与规格一致;符号面/冻结/正式训练仍未完成;训练/收益 NOT_RUN**

承接 `answer27.md`。接受 question28 §1 的核对。

## 1. 规格与代码对齐(同一次提交)
`math_spec.continuous_surface.prefix_surface` 三处旧陈述已改:
- `faces.implemented` = 三层累计结构项(同一前向、同一三元组、同一 h 的 `mean(L1,L2,L3)`);
- `faces.to_implement` = “位置路径上无剩余项;`_positions` 已返回 q1..q3;剩下的是符号面与监测”;
- `coverage_gaps[2]` = 三层目标已由 `structure_term` 应用,符号面仍不可重建。
`training_integration.prefix_objective` 同步为“structure_term 应用三层均值”。检索 `GEOMETRY.md`/`REMAINING.md`,无 `last-layer`/`levels[-1]`/`only the last` 的旧陈述,无需改。**注意**:三个字段都在同一提交内改,本答案收据覆盖改后规格。

## 2. `_positions` docstring 重写
改为实际语义:返回同一次 `forward_with_codes` 的 q1/q2/q3 三个累计回读,码本走原生组合 Jacobian,encoder 走 x 的 STE;删除“加 codes_to_positions 会引入第二条路径”的旧理由。

## 3. 第二个接线测试真正可区分
`tests/test_q27_layers.py` 的第二个用例改为
`test_the_prefix_gradient_differs_from_the_last_layer_gradient`:
- 同一参数、同一三元组下比较 `grad(term)` 与 `grad(last)`(逐层码本参数,`rtol=0, atol=0`),断言存在差异;
- 旧的 `[-1]` 实现下两者恒等 → 必须在断言处失败。
**红证据**:用 `36d10856` 的 `training_hooks.py` 在临时树运行该测试,`assert not torch.allclose(term, last)` 失败、`rc=1`,存为 `q28/red_layers.log`;当前实现下通过。

## 4. 每层 H 范数诊断
`structure_term` 的 capture 现记录 `layer_h_norm_max`(三层位置 H 半区的 max 范数),`probe_q9` 把它写入报告 `identity.layer_h_norm_max`;该值只作诊断,不设门槛。

## 5. step-0 诊断已落盘
`results/research/struct_mixed_rq/q27/layer_diagnostics.json`(seed=42、194 个三元组):
| 层 | 损失 | 零距对 | 占比 | max‖C^H‖ |
|---|---:|---:|---:|---:|
| L1 | 1.4636061191558838 | 5+4+3 | 2.06% | 0.12369927018880844 |
| L2 | 1.7739014625549316 | 0+0+1 | 0.17% | 0.12362603098154068 |
| L3 | 2.0398731231689453 | 0+0+0 | 0% | 0.1236429363489151 |
`s=0.04255550727248192`(末层标定;单一共享 s 为规格定义)。

## 6. 收据
最终 `q28/tests_20261110T110739.log`:37 条命令、37 个不同文件、**37 pass / 0 fail**,runner `rc=0`,
`source_drift: none`。中间失败收据 `tests_20261110T105850.log`(3 fail,`layer_h_norm_max` 未在非活动分支初始化)保留,不重写。

## 7. 仍未交付
完整 288 标量 SymPy 面;冻结/准入/正式 Stage2/Stage3 与增益;旧 `0.0579` 未移交。

## 8. 摘要(自动计算,完整 64 位)
| 文件 | SHA-256 |
|---|---|
| 候选 `math_spec.json` | `8ca83f45db0f39f93286775ca55f64d9f9b4e5767ddaa5ba7eff8e3086911fbb` |
| 候选 `training_hooks.py` | `8ede9c091da752d02346ec7acb9fce25450aa14c3aa05d0cd274c1d2443beeff` |
| 候选 `probe_q9.py` | `8771423df7f757082b823e2e0c7f332991d26b59f73e50f246e5d44c8c5824d9` |
| 候选 `tests/test_q27_layers.py` | `d553c1b4a0f5ce73d945517ae9ce613e704595e71d5369fead529dcdd6e2561e` |
| 证据 `q28/red_layers.log` | `a2d4f007ad992b13037b44023f503ee005cada3a1ca8abb78653002b3b43b6b1` |
| 证据 `q28/tests_20261110T110739.log` | `e7a1692f274c149c21f53d87f4b814a6bc334c8e6538d4475cbdffc2135a12ee` |
| 证据 `q28/tests_20261110T105850.log`(失败保留) | `abb697ec78e8fc7b5b6346f8800244eef650b3a7256c92ac4f2dfc9705a4756f` |
| 证据 `q28/run_suite.sh` | `b8df31fdf007c51d2a7ff01c036e3b9ae37d96271e10c2a54cb2e59a849615a1` |
| 证据 `q28/source_pre_20261110T110739.txt` | `f749eee7888c20667b86f6243898e1f57b955287999339af9bd14efabb6b29c4` |
| 诊断 `q27/layer_diagnostics.json` | `3e420df1b26e242cf16042f6496d30706231287c8cfb6cdf729e4bbef3b8b8bc` |
