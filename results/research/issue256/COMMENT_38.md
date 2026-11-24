R01已按规则在新机制设计前完整归档：`git mv stage2_RQ-VAE/TIGER_module_budget_prefix_rq stage2_RQ-VAE/_archive/TIGER_module_budget_prefix_rq`退出0，含未跟踪诊断入口一并保留。归档前同UID进程扫描无本臂训练/验证/控制器依赖、无不可读PID；所有正式及诊断作业均已退出0。

前向引用`tools/issue256/identity.py::SOURCE`已迁移到归档路径，独立进程实际import与文件位置检查通过。Stage2顶层现只有baseline/和_archive/；不可变`_runs/issue256_r01_formal/.../src`与其checkpoint、SID、R2/102、冻结合同和历史证据均未移动/改写。证据`rounds/r01/ARCHIVE_PRECHECK.json`和`LIVE_ARCHIVE.json`。核对实际训练源码后，确认校准rank_local_checks只影响诊断检查，未替换实际硬分配/损失路径；后期失衡是初始化保证范围之外的实测结果，不伪报实现故障来重训。

R01完整负结果会封存并推送；随后在同一#256登记下一独立假设，仍从只读冻结baseline建新臂，不从旧训练权重初始化。
