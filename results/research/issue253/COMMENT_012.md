## C12 两个新故障的针对性修复复验 PASS（未宣称整体准入通过）

`tools.issue253.test_checkpoint_state`、`test_sample_indices`、`test_estimator_mutations` 合计 **6 tests PASS，0 skipped，退出 0**。命令为 Stage2 解释器执行上述三个 unittest 模块，输出 `results/research/issue253/CHECKPOINT_HOST_GREEN.log`。

- 实际标准 checkpoint restore 后，距离函数使用保存的尺度，不再使用默认 host mirror。
- host 输入在传输前展平、传输后恢复原形状；已知答案、全部梯度、重复 reference occurrence、空/非法输入通过。
- 采样器与不可变 #252 的 36 个真实 rank/step 抽样结果及顺序一致。
- 6 个 estimator 变异均被指定断言抓到；先前的 shape 异常不算捕获。

生产修改限于新树 `model/mechanism.py` 和 `model/estimator.py`。接着跑最终源码绑定全量测试、真实四卡 kernel/两分支等价验证，再测完整 60 epoch inclusive 性能；正式训练仍未开始。
