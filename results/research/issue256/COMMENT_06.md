R4 红→绿的实际模块回归已通过：同一`launch_budget_check`完整1024商品/卡场景，四个rank均PASS，launcher exit0。每个17模块的预算、最大约束饱和、原始真实Loss梯度及参数不变断言均实际执行。源码与PID逐项绑定于`rounds/r01/CALIBRATION_GREEN_RECEIPT.json`；红记录未覆盖。

范围限定：此回归是四卡上四个rank-local既有冻结权重fixture，不冒充正式初始状态的全局DDP证明。完整候选/既有框架测试正在执行；下一步仍必须由实际Trainer独立验证首个global4096 batch的平均VJP、17模块预算、逐层梯度/优化器、硬SID重导出、机器entry/behavior_graph及inclusive性能，然后才可派发正式训练。
