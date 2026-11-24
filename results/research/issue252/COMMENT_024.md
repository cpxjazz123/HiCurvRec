## C7 仍 FAIL：新完整四卡窗口实测，不能启动正式训练

命令 `genrec_env_v2/bin/python3.9 -m tools.issue252.launch_benchmark` 退出1，完整运行60个诊断epoch（冻结3000epoch调度、global4096、seed42），第50轮真实全目录评估包含在窗口内。run_id `issue252_benchmark_669470_578127856`。

| 连续epoch31–60口径 | 实测 |
|---|---:|
| 总墙钟秒 | 32.3996938094 |
| 平均秒/epoch | 1.0799897936 |
| 中位数秒/epoch | 0.9042086266 |
| P95秒/epoch | 1.7783592449 |
| 初始化秒（单列） | 28.2181958472 |
| 按该窗口推算3000epoch秒 | 3239.9693809450 |

证据 `results/research/issue252/benchmark/issue252_benchmark_669470_578127856/rank_0.json:3399ddb1f0c27adc50e1c4ffe70090c620737e8caf03e5728851a47843254232`，其余3卡实际完成报告也保留；每epoch记录四卡结构覆盖/更新。这里不减去评估、编译或慢epoch；该值超过0.5秒上限，不能拿算子加速或诊断入口通过代替性能准入。

当前源码的完整数学套件另已实际退出0：`contract_tests/1795478044308266055/` 下 deterministic60、reference11、mutation2方法/15指定变异全部通过，与实际一轮模块/DDP/导出证明同源。下一步已通过唯一launcher派发3个实际epoch的暖态逐阶段profiling，定位剩余代价；不调权重、不改SID/全局batch/维度、不开始正式3000轮。