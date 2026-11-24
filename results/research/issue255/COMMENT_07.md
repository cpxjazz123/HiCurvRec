独立源码审查发现数学契约违约，已暂停 Stage3：`calibrate_curvature_weight` 取 `compute_loss` 的第二返回值（仅重建），而冻结公式要求第一返回值（完整原始 RQ-VAE 损失，含 commitment/codebook）。该差异有源码证据：`model/model.py::compute_loss` 返回 `(total, reconstruction)`，校准却写成 `_, reconstruction`。旧3000 epoch不是契约内机制的正/负科研结果。

owner 已明确选择 `repair_rerun`：**修复并补齐事前验证，保留旧run为实现无效证据，批准同一臂一个修正后的3000 epoch四卡正式run，再完成Stage3和归档**。这是实现错误的获批重做，不是按 Recall 追分；尚未运行任何 Stage3。修正后唯一有效正式代次固定为 `issue255_formal_r1`，旧 `issue255_formal` 及其 checkpoint/SID/日志保留并标 INVALID_IMPLEMENTATION，不改写历史。

同时，冷重导出失败的原因已独立确认并修复：训练源码 `set_seed` 还设置 deterministic/float32 matmul precision=high，冷检查器未恢复它而使用highest。恢复训练源码的数值运行时后，严格逐项重导出审计已退出0；未重写SID、未放宽相等条件。该检查仅证明旧权重可复现，不消除上面的λ公式错误。

当前候选测试103项及工具/资产测试21项已经实际通过，但原测试没有覆盖错误的λ分子。下一步先补真实模型/真实batch的红灯复现，再修实现和梯度比较口径，修复GPU/CPU提取与逐用户配对审查问题；全部数学、梯度、DDP、真实入口及性能凭据齐备后，才派发修正代次。随后完成#255的全量评估、Stage3、归档、推送与关闭，再开始#256。
