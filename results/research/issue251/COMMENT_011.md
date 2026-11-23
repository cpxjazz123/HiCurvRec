四卡受控运行 `issue251_validation_1970849` 已结束，退出0：四rank梯度/参数最大差均0，独立梯度均值相对误差8.65e-10；全28,391,948目标、24,326中心及全参考集反传完成。8个模块实际更新，硬SID变化24,580商品；checkpoint→SID严格一致。rank0峰值显存13,068,528,640字节，总壁钟1051.11秒，真实PID1970855–1970858。原始证据 `runtime/`、`MODULE_CHECKS.json`、`RUNTIME_SUMMARY.json`，源码358db180。

C7随后真实MCP入口仍FAIL：原冻结重建batch的GNN图在多个chunk间保留激活，CUDA实际分配43.39GiB后OOM（`ENTRY_OOM.json`）。不能以四卡微步替代真实入口；将对GNN子图chunk加等价激活检查点，不减batch、不抽样、不改Loss/超参，并重新绑定重验。机器行为图检查还漏识别外部EgoStore而返回NOT_APPLICABLE，正在修正，未将其当图隔离PASS。

硬q/连续z的独立全量重算已完成，但上述实现修复后旧运行仅保留为历史证据，最终验收必须重新执行。
