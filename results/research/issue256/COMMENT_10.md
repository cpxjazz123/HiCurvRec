R4 entry 已从FAIL→PASS，没有修改/绕过机器闸门。新回归先得到2 failed/1 passed（RandomSampler与显式generator模式复现None故障），修复后实际首batch/全局RNG/显式generator恢复及既有启动回归共5 passed。`ENTRY_CALIBRATION_RED.log` / `ENTRY_CALIBRATION_GREEN.log`保留红绿。

统一改为`_calibration_batch(loader, device)`窥视真实DataLoader首batch并恢复随机状态，返回实际features及原CPU商品ID；删除旧IDs-only接口，所有调用迁移。数学系数、GNN/门控/实际分配和冻结配方未变。

重跑`python3.9 -m tools.issue256.admission`：entry=PASS、behavior_graph=PASS；真实entry完成1epoch并导出SID。训练源码SHA256 `b96cf0c322db714b69bb5246b60c623a5e0c0f0b48905d6b1b63f7f4558afe1b`，凭据`rounds/r01/ADMISSION.json`。smoke碰撞0.051857只是入口工件，不是3000epoch最终结果，未启动Stage3。旧凭据完整保存在`verification_generations/pre_loader_fix/`；新源的全量测试、四卡模块/导出与性能代正在重新生成，全部通过再正式派发。
