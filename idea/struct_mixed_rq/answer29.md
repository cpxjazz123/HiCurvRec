# answer29 — 梯度红证据单独落盘，训练路径记录码字范数
**状态:三层结构项+码字范数诊断已对齐;符号面/冻结/正式训练仍未完成;训练/收益 NOT_RUN**

承接 `answer28.md`。接受 question29 §1 的核对,并更正 answer28 §3 的红证据归属。

## 1. 梯度差测试的独立红证据
`answer28` §3 引用的 `q28/red_layers.log` 实际失败在**第一个**用例(值比较),新的梯度差测试在旧实现下从未执行。本轮补跑:
- 临时树使用 `36d10856` 的 `training_hooks.py`;
- 只调用 `test_the_prefix_gradient_differs_from_the_last_layer_gradient()`;
- 结果 `AssertionError: differs`、`rc=1`,存为 `q28/red_gradient.log`。
当前实现下该用例通过;`red_layers.log` 仍作为“值不同”的红证据保留,两者不混用。

## 2. 监测改为码字范数,并进入训练路径
- `structure_term` 的 capture 分开记录:
  `codeword_h_norm_max`(逐层码本 `‖C_l^H‖`)与 `position_h_norm_max`(累计位置 `q_l` 的 H 半区范数);
  probe 把两者分别写入报告 `identity`,不再把位置范数当作域判据;
- `train_rqvae.py` 在每个 epoch 更新每层 `‖C_l^H‖` 的 **running max**,在 `EVAL_INTERVAL` 行与末 epoch 行写入
  `training_metrics.jsonl` 的 `codeword_h_max` 字段(不 append 无界列表);
- `math_spec.domain.training_path` 写明:训练消费全域 `mixed_sq`,≤1 是参考抽样域,码字范数以 `codeword_h_max` 记录;
- `GEOMETRY.md`/`REMAINING.md` 无同类旧陈述,无需改。

## 3. 收据
`q29/tests_20261110T111718.log`:37 条命令、37 个不同文件、**37 pass / 0 fail**,runner `rc=0`,
`source_drift: none`;覆盖改后的规格、`training_hooks.py`、`probe_q9.py` 与 `train_rqvae.py`。
该收据未执行真实训练循环,`training_metrics.jsonl` 的新字段尚无实跑样本(如实记 NOT_RUN)。

## 4. 仍未交付
- 完整 288 标量 SymPy 面(`prefix_surface.statements` 仍伪代码,已标 `statements_status`);
- 冻结、机器准入、正式 Stage2、同名 Stage3、`test_R@10` 与增益;旧 `0.0579` 未移交。

## 5. 摘要(自动计算,完整 64 位)
| 文件 | SHA-256 |
|---|---|
| 候选 `math_spec.json` | `bbcb3110f2fcc3ea34d68f5b3cbc20c0655ad74591a019e6d336b5a990532ac1` |
| 候选 `training_hooks.py` | `4046b3f8c7680dc7d9b4534b9069347792a5e224ba5a641bedfd8cd38e01fc34` |
| 候选 `probe_q9.py` | `088e47dea25c8a1aa212dcaff577a91c5f64386c5a54c750c963147d4f62e5dd` |
| 候选 `train_rqvae.py` | `61dd1d9ada2b02aa752b9832951275819186c7e55d7d6781a761589d6160add9` |
| 证据 `q28/red_gradient.log` | `4d5602292fc8dd8b796208eeb47a3bd76577695c6fa6e52dea16493a257f05de` |
| 证据 `q29/tests_20261110T111718.log` | `43b82a6fd5ac826e2c8f8504233e5b1f31f273a6a0f352c3d0db940d7a60ddcb` |
| 证据 `q29/run_suite.sh` | `2aff643e941ccbcf448f228dd61f7e63640e9abf937e0a6280d189db68e1e5f4` |
| 证据 `q29/source_pre_20261110T111718.txt` | `0e64b2e45007ac49d9f092c38255973ff8634209f986aa5f4499a25e2c1ecb90` |
