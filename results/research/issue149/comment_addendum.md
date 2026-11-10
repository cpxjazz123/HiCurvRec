## 补记（针对 R2 的键名与证据，提交 `38bc5519`）

上一轮 comment 里我把描述表身份的 key 叫 `structure_sha256`；你的 R2 措辞是 `table_sha256`。语义相同但 grep 会落空，已统一改名为 **`table_sha256`**，并补上你要求的"换表后恢复必须失败"的测试：

- `MixedRQ.get_extra_state` / `set_extra_state` 使用 `table_sha256`（`model/mixed_rq.py:338,358,380,389`）；
- `MixedRQVAE.from_checkpoint` 比对 `extra["table_sha256"]` 与所载表的 SHA，不符抛 `ValueError("this checkpoint was trained against another structure table")`（`model/model.py:114`）；
- 新测试 `tests/test_gated_trainer_wiring.py::test_the_checkpoint_binds_its_frozen_table`：同一 state 两次 `from_checkpoint` 的逐商品 α 完全相等；换一张**manifest 合法但 SHA 不同**的表必须抛错。

全量套件现为 **161 passed / 10 failed**（10 个失败仍为冻结基线原样复现的三类）。仍**未启动 Stage2**，等你的审查结论。
