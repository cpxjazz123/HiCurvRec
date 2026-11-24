## C8 执行优化：同一50次软分配，实际CUDA kernel由1773降至420

先红后绿，四卡均有原生profiler结果：

- RED：`geometry_validation/geometry_1795479503061091119_846241_cfbd895d0519/rank_{0,1,2,3}.json`，18项概率/独立参考VJP检查通过，但真实1024×256、50次迭代的forward/backward有1773个compute kernel，超过预先固定的464预算；有限梯度，不是崩溃。
- GREEN：`geometry_validation/geometry_1795479619811956729_860937_0cddf4cfa357/rank_{0,1,2,3}.json`，每卡148项数值/梯度、16硬选择、18原floor对照通过；同一软分配420个实际compute kernel，独立概率域参考的VJP最大相对L2误差2.492723e-5，小于原rtol1e-4。原50次迭代、0.003温度、硬Sinkhorn/floor/容量和SID路径不变。

新增准入回归另发现：候选树外的估计器变异测试、测试执行器和独立reference改动，旧报告曾能被接受。已观察指定拒绝断言RED，随后将实际测试源码/执行器/reference纳入同一canonical source binding，benchmark/entry不再复制一份inventory逻辑；36项凭证/不可变快照/驱动回归GREEN。证据 `EXTERNAL_TEST_BINDING_RED.log`、`INDEPENDENT_REFERENCE_BINDING_RED.log`、`COMPLETE_TEST_BINDING_GREEN.log`。最终数学runner将实际执行候选树及本issue相关框架测试，不只记录新增测试文件名。

当前在独占诊断任务下重新运行完整60epoch速度窗口。算子成功不代表<=0.5秒/epoch；新的完整测试绑定也要求最终真实入口/数学凭证重跑后才可正式准入。没有正式训练启动。相对证据路径均位于 `results/research/issue252/`。