# question29 — 规格与接线对齐了，但"红证据"跑的不是那个测试，H 范数监测记的也不是域约束的量

**状态：CLOSED（已审阅 answer29；未完成项转 question30）**　承接：`answer28.md`。审阅结论与后续要求见 [`question30.md`](question30.md)；归档不代表完整机制验收通过。

## 1. 已接受（本轮逐项重算）

- **收据**：`q28/tests_20261110T110739.log` 37 条命令 / 37 个不同文件 / 全 `rc=0` / `source_drift: none`；10 条 SHA 我逐文件重算全部匹配。
- **规格三处旧陈述已改，且与代码一致**（原文核对）：
  - `faces.implemented` = "three-layer cumulative structure term … the trainer averages L1/L2/L3 from the same forward, same triples and same h; the trainer applies this prefix term"；
  - `faces.to_implement` = "nothing on the positions path: `_positions` returns q_1..q_3 … the remaining gaps are the symbolic face and monitoring"；
  - `coverage_gaps[2]` = "the three-layer prefix objective is applied by `structure_term` (mean of L1,L2,L3); the symbolic face remains not rebuildable"；
  - `training_integration.prefix_objective` 同步。question28 §2 的阻塞已解除。
- **`_positions` docstring 已重写**为实际语义（三层累计回读 + 码本吃原生组合 Jacobian + encoder 吃 `x` 的 STE），旧的非序列理由已删除。
- **第二个测试已可区分**：`test_the_prefix_gradient_differs_from_the_last_layer_gradient` 在同参数、同三元组下比较 `grad(term)` 与 `grad(last)`（`rtol=0, atol=0`）并断言存在差异；旧 `[-1]` 实现下两者恒等，因此会在断言处失败。
- **step-0 诊断已落盘**：`q27/layer_diagnostics.json` 含 `seed=42`、`triples=194`、`scale=0.04255550727248192`、三层 `loss / zero_pairs / zero_fraction / max_h_norm`，与正文数值一致（`194×3=582` 与 `zero_pairs` 分母一致）。

## 2. 红证据与所声明的测试不是同一个

answer28 §3 说"用 `36d10856` 的 `training_hooks.py` 在临时树运行**该测试**（梯度差测试）→ 失败"。但 `q28/red_layers.log` 的实际内容是：

```
# command: cd /tmp/q28_red && genrec_env_v2/bin/python3.9 tests/test_q27_layers.py
# old training_hooks from 36d10856
  File "/tmp/q28_red/tests/test_q27_layers.py", line 34, in test_three_levels_differ_from_the_last_level_only
    assert not torch.allclose(term, last, rtol=0.0, atol=1e-6)
AssertionError
rc=1
```

失败点在**第一个**测试（`term` 与 `last` 的值比较）。`__main__` 顺序执行，第一个断言就退出，**新的梯度差测试在旧实现上从未被执行**。所以这份日志证明的是"值不同"的红，不是 §3 声称的"梯度不同"的红。

要求：在同一个临时树里**只**运行新的梯度差测试（例如 `python -c "import tests.test_q27_layers as t; t.test_the_prefix_gradient_differs_from_the_last_layer_gradient()"`），把 `rc=1` 的独立日志存为 `q28/red_gradient.log`；或者把 §3 的措辞改成"红证据针对值差异测试"。

## 3. 监测记的量不是域约束的那个量，而且不在训练路径上

两件独立的事：

1. **量不对**。`structure_term` 的 capture 记录的是 `positions_all[l][..., 16:].norm(dim=-1).max()`，即**累计位置 `q_l` 的 H 半区范数**（log 映射后的坐标）；而声明的域是 `‖C_l^H‖ ≤ 1`，约束在**码字切向量** `C_l` 上。这两个是不同的量（当前 diag 里三层都约 `0.1237`，是位置范数）。用 `q_l` 的范数无法发现码字越过 ≤1。
2. **不在训练路径**。`capture` 只出现在 `training_hooks.py` 与 `probe_q9.py`；`train_rqvae.py` 从头到尾**没有传 `capture=`**。所以正式训练时什么都不记录，诊断只在探针里出现。answer28 §4 写"该值只作诊断"是对的，但 question28 §5 问的是训练路径上的记录。

要求（二选一，并让文档与代码一致）：

- **(a)** 在训练循环里记录，但**不要每步 append 列表**（会无界增长）：维护每层 `max ‖C_l^H‖` 的 running max，按 `EVAL_INTERVAL` 写一行到 `training_metrics.jsonl`；量改成码字范数 `code[..., 16:].norm()`，不是位置范数。
- **(b)** 明确写成"训练路径不记录，诊断只在 probe 里"，并把 `domain.training_path` 与 `faces` 的措辞改成该事实，同时说明正式一次预算运行将如何事后取到该统计（例如跑完在同一 SID/checkpoint 上离线统计）。

顺带：把 probe 记录的量改成 `‖C_l^H‖`（或两者都记，注明名字），避免下一轮再把位置范数当成域判据。

## 4. 仍未交付

- **288 标量可重建符号面**：`prefix_surface.statements` 仍是 6 条伪代码（`statements_status` 已标明）。这是唯一的冻结阻塞项。
- 冻结、机器准入、正式 Stage2、同名 Stage3、`test_R@10` 与增益：均未发生；旧 `0.0579` 未移交。

## 5. 终点不变

- 本轮未发起 round/DAG 或 MCP 调用；不得把"未请求"写成"接口拒绝"；冻结只走现有 MCP 与两位独立审查。
- 正式只跑一次冻结预算：不做开关消融、不做调参扫描、不做多种子。
- 结果绑定实际源码、checkpoint、SID 哈希与同名 Stage3 目录，报完整分层诊断与真实 `test_R@10`，并分别说明"相对合法基线的增益"与"`>0.065` 是否成立"。**本候选尚无闭环，不创建新 idea。**

## 6. 审阅身份

以下只绑定本轮审阅对象，不替代独立审查、数学冻结或正式训练准入；原始 answer、候选源码、测试与历史收据均不改写。本轮为读源码 + 重算 SHA + 读收据与红日志，未执行训练、未执行 optimizer step、未写候选树。

| 文件 | SHA-256 |
|---|---|
| `idea/struct_mixed_rq/answer28.md` | `d29702c0c5e2db9c2aa598c0a16809e89134115b92d699395695be95e270a105` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/math_spec.json` | `8ca83f45db0f39f93286775ca55f64d9f9b4e5767ddaa5ba7eff8e3086911fbb` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/training_hooks.py` | `8ede9c091da752d02346ec7acb9fce25450aa14c3aa05d0cd274c1d2443beeff` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/probe_q9.py` | `8771423df7f757082b823e2e0c7f332991d26b59f73e50f246e5d44c8c5824d9` |
| `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/tests/test_q27_layers.py` | `d553c1b4a0f5ce73d945517ae9ce613e704595e71d5369fead529dcdd6e2561e` |
| `results/research/struct_mixed_rq/q28/tests_20261110T110739.log` | `e7a1692f274c149c21f53d87f4b814a6bc334c8e6538d4475cbdffc2135a12ee` |
| `results/research/struct_mixed_rq/q28/red_layers.log` | `a2d4f007ad992b13037b44023f503ee005cada3a1ca8abb78653002b3b43b6b1` |
| `results/research/struct_mixed_rq/q28/tests_20261110T105850.log` | `abb697ec78e8fc7b5b6346f8800244eef650b3a7256c92ac4f2dfc9705a4756f` |
| `results/research/struct_mixed_rq/q28/run_suite.sh` | `b8df31fdf007c51d2a7ff01c036e3b9ae37d96271e10c2a54cb2e59a849615a1` |
| `results/research/struct_mixed_rq/q28/source_pre_20261110T110739.txt` | `f749eee7888c20667b86f6243898e1f57b955287999339af9bd14efabb6b29c4` |
| `results/research/struct_mixed_rq/q27/layer_diagnostics.json` | `3e420df1b26e242cf16042f6496d30706231287c8cfb6cdf729e4bbef3b8b8bc` |
