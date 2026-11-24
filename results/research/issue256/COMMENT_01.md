冻结 #256 验收清单：17 条，逐条原文见 `results/research/issue256/ACCEPTANCE.json` / `RUN_LOG.md`。Issue 正文 SHA-256：`32a2764070e8502926965d355a3ef6ff7619d45881abd0ffa575cb7cef73c5a8`。

#255 已 CLOSED（BLOCKED/NOT_PLANNED）；合法修复运行完成3000epoch，但前三码碰撞0.9814129418，Stage3机器拒绝，无Recall/R2完整包。证据commit `0d1b61101dbbf41374ac33571df959e88cfd00bf`，关闭回执commit `c4ee00cdc4314b7fcab882acd939bf18ee3b0824`。不将其冒充完整通过。

本Issue必须训练：总目标、R6与最终验收要求实际合法完整Stage3 `test_R@10 > 0.065`，现有结果不足。每个明确、预注册的新假设一次正式3000/150训练；正文明确授权后续版本迭代。负结果不结束本Issue，不绕过碰撞门槛，不同机制追分重跑。下一节点先归档已停止的#255源码、核对已有assets和历史来源，再以训练/validation证据登记单一改动。
