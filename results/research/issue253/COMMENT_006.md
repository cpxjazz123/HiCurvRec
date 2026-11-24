C12准入注册回归已转绿：`python3 -m unittest tools.issue253.test_admission_policy tools.issue252.test_training_admission -v`，18tests/0fail/0error，exit0。保留源码/资产漂移、伪造性能、缺rank、缺真实loss/更新/开关/导出均拒绝的旧负例；三专家候选现要求本issue的四ranksource-bound凭据，不走旧单卡路径。证据 `ADMISSION_POLICY_GREEN.log`。

已增加仅用于精确profiling的GNN边界（第一affine、第二方向聚合、第二affine/activation、pooling与static gather），无数值公式变更；5个相关数值/全部GNN参数梯度/全ego及三专家检查PASS，exit0，`PROFILE_BOUNDARY_GREEN.log`。

已通过唯一launcher启动真实四卡、global4096、6step/epoch的三专家原serial kernel/原双分支profiling；这是3epoch诊断，不是正式训练或性能完成。实际rank PID将以该次报告为准。正式3000/150尚未启动。
