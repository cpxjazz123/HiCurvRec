## warmed实际路径已完成：GNN峰值有余量，性能仍未达标

修复导入根后，`genrec_env_v2/bin/python3.9 -m tools.issue252.launch_steady_profile` exit0，run `issue252_steady_profile_117066_577707783`，4份PROFILED、每rank真实18 optimizer/scheduler steps、3完整epoch，无正式产物。实际rank0：epoch1 11.14894s，仪器化epoch2 2.90090s，退出仪器化后的epoch3 1.81517s；预热至epoch1结束94.4654s。四rank实际峰值约9.91–10.08GB（reserved13.16–14.64GB），不是推测的显存余量。

原始报告 `results/research/issue252/steady_profile/issue252_steady_profile_117066_577707783/rank_*.json`：rank0每epoch GNN24calls/0.2326s CUDA span，floor36/0.4659s，guards144/0.3024s，data_read7/0.1700s，反传6/0.9308s；这些是nested仪器化范围，不能相加冒充总墙钟。原生sort630calls/0.1435s，CSR转置转换48calls/0.0165s，存在可避免的稀疏反传支持排序和大量细粒度几何算子。

继续相同公式的精确执行优化，已在CUDA_EXECUTION.md、FLOOR_EXECUTION.md前置声明：bulk tensor fetch保持四rank/多epoch采样与padding逐样本完全一致；严格唯一priority的CPU floor repair保留原ID/metadata、并列/nonfinite回原GPU规则。相关7项RED→GREEN，FLOOR_CPU_BULK_TESTS.log。尚未把这些单测当GPU端到端证明或C7通过。

旧native CSR转换的首个CUDA detector竟然在旧代码通过，报告保留且判无检测力；正在补实际float32原生profiler检测，不能把这个PASS当优化生效。完整连续30epoch≤0.5s仍未通过、正式Stage2未启动。

commit_guard当前853paths/278012lines超850路径限，未构造draft。共享工作区含大量非本任务变化；后续仅选择本任务行，并对实际小型源码/测试/逐rank证据显式使用900paths/300000lines预算，不选择他人SID表/改动。