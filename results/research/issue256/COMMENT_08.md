R5 已过既有inclusive均值上限，未达到目标速度。真实四卡连续epoch31–60（30epoch，含epoch50周期工作、通信/优化器/结构Loss及epoch间开销）：mean=0.7099018641s，median=0.6787123508s，P95=0.8969776072s；初始42.45023085s另报。0.4–0.5s目标未达；不能把P95说成低于0.8。四rank稳态epoch2 profile完成并保留各模块/卡时间与峰值内存，未删2-hop/数据、未冻结GNN。

证据：`rounds/r01/PERFORMANCE.json`、`PROFILE_SUMMARY.json`、原始benchmark/profile四rank文件（SHA绑定）。完整实际模块凭据`MODULE_CHECKS.json`含18组真实更新/梯度与17组初始global预算；五个新系数变异的指定AssertionError已真实捕获到`BUDGET_MUTATIONS.json`，另保留继承的9+3数值/结构/路由变异。正式训练尚未启动；机器entry/behavior_graph正在执行，必须通过后才派发。
