C8连续诊断缺项已红→绿：`python3.9 -m unittest tools.issue251.test_evaluation tools.issue251.test_full_evaluation -v`，22项全PASS（`CONTINUOUS_PARITY_FINAL.log`）。硬q和连续z共用同一严格bank处理路径，但各自传入实际坐标；新增独立decoder重建误差入口，手算的反向排序AUC0和MSE2.5均通过。

第一次新评估启动只因归档元数据并行创建目录，mkdir返回“File exists”而退出1，未执行评估、未移动旧产物；这是操作命令故障，不是模型结果。已改用mkdir -p顺序归档再跑CPU评估，错误原文保留 `EVALUATION_LAUNCH_FAILURE.log`。模型源/权重/冻结目标不变，不重跑GPU训练或用已有指标冒充新评估器的结果。
