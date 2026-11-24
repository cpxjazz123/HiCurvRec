新增 C13/C10 FAIL，继续修复，不启动正式训练。

- C13真实四卡60epoch基准 `issue253_benchmark_3872457_580922477`：inclusive31–60均值 **0.9118503464696308s**，中位0.5979033862s，nearest-rank P95=2.9670059206s，超过0.8s上限。epoch35=2.962s、实际周期评估epoch50=7.402s均保留在窗口；先前0.649–0.655只是单epoch诊断，不能替代此判定。workers3872476–3872479，60epoch全部执行，rank0正确按性能门拒绝。原件 `results/research/issue253/benchmark/issue253_benchmark_3872457_580922477/rank_0.json`。
- C10真实四卡CUDA run `issue253_kernel_validation_3888447_580934726`：每rank8tests、12指定断言失败、0errors。旧kernel的float64度数3/7转置梯度期望10/21，实际0.4761904925107956；非连续输入错误达2.41–3.90。已定位为固定float32倒数与未使用stride，不以CPU通过冒充GPU通过。四rank日志均保留，workers3888466–3888469。三专家compiled几何及真实floor硬ID干预0→1在float32/float64均通过（最大f32梯度误差7.55e-7，f64约1.8e-15）。

证据：`BASELINE_SEQUENCE_EXIT.json`、上述两个run目录及 `BASELINE_PROFILE.json`。已授权#253独立kernel修复/向量化，并继续精确同一步GNN复用；不删除2-hop、不减少采样、不放宽性能或数值容差。
