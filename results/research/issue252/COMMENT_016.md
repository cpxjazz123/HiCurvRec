## CHECK_1：4 PASS /2 FAIL /8 NOT_RUN，继续执行

CHECK_1.md逐条重抄14条issue原文、阈值和实际path:SHA，非最终准入：C1/C2/C4/C6已完成；C7完整窗口2.2026225s超限；C8同代全量数学检查保留1个缺资产exception fixture ERROR，其余生产数值/参考/变异已执行。其余尚无正式训练/导出/Stage3/R2结果，均NOT_RUN，不因单模块测试绿而视为完成。

新增真实运行故障：warmed profiler首发4rank在导入tools前崩溃，未有optimizer step；launcher_93349/logs/steady_profile.log保留。根因标准launcher将cwd设结果目录，新driver未先解析repo/sys.path。新独立子进程回归从无PYTHONPATH的结果cwd执行真实driver并核对epoch1/2/3跟踪，RED_DRIVER_ROOT.log→DRIVER_ROOT_TESTS.log（1PASS）。已修repo标记定位/导入顺序后重发诊断；未更改训练配方或重跑正式实验。

下一轮先取得warmed逐阶段实测，再继续精确优化到完整性能门达标；没有以断点或本轮FAIL收尾。