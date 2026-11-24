## C12 新发现的实际检查失败（继续修复，未启动正式训练）

1. 标准 `load_state_dict` 没有刷新新模型的冻结尺度 host mirror，实际混合距离使用默认尺度而非 checkpoint 中的 `tangent_scale`。真实 RQVAE 回归断言已失败：`restored checkpoint metric must use serialized tangent_scale`。命令：Stage2 解释器运行 `tools.issue253.test_checkpoint_state`，unittest 退出 1，1 个指定断言失败；证据 `results/research/issue253/CHECKPOINT_SCALE_RED_2.log`。此前超时不计为 RED。
2. 新的 host 索引路径将二维 triples 直接传给既有只接收扁平数组的异步传输 helper，导致 shape mismatch。`tools.issue253.test_sample_indices` 与 `test_estimator_mutations` 共 5 项中 1 PASS、4 ERROR；错误不计为有效变异捕获。证据 `results/research/issue253/HOST_SAMPLE_GREEN.log`（文件名不是判决，实际结论 FAIL）。
3. 原始采样器和冻结 #252 的真实 36 个 rank/step 样本逐字节、顺序与 receipt 检查通过。

修复只落在 #253 新模型；不修改历史执行源码、原始采样器或数学容差。之后执行针对性复验及真实四卡检查。正式 Stage2/Stage3 均未启动。
