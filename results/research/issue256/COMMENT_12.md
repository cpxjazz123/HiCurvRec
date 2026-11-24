Profiler修复先红后绿：真实SteadyTracker+真实DataLoader的隔离回归，以及修改实际profiling.py/profile_units.py必须使凭据失效的回归，先2 failed、修复后2 passed（`PROFILE_LOADER_RED.log`/`PROFILE_LOADER_GREEN.log`）。包装器透明委托真实loader属性；profile源码身份额外绑定两个实际执行依赖，没有在训练里隐藏/绕过异常。诊断控制器先跑profile以尽早暴露测量接口错误，再跑模块与inclusive基准；机制、数据、训练预算不变。

历史检查轮已完整落盘：CHECK_1（sampler-None入口失败）和CHECK_2（测量包装器失败）各12 PASS/1 FAIL/4 NOT_RUN；两轮都没有正式派发。文件明确为现在对既有执行凭据的复核，不倒填前置时间。CHECK_3_ISSUE.json重读正文SHA与冻结清单一致。当前源的全量测试与全部真实凭据仍在刷新；完成后才可正式启动。
