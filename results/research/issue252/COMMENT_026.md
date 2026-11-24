## 进度：真实路径已通过，性能门槛仍未达到

代码/证据已提交并推送 `4be8600453f28fb4c7d08536f16e4d212e0aae8c`，实测本地main与远端main同SHA。CHECK_2仍是PASS6/FAIL1/NOT_RUN7；正式Stage2与Stage3均未启动。

最近完整60epoch诊断窗口平均1.0799897936秒，上限0.5秒，不能开正式训练。新暖态四卡profiling `steady_profile/issue252_steady_profile_689572_578144683/rank_{0,1,2,3}.json` 指出：每卡每epoch144次assignment guards耗CPU0.299–0.312秒；36次soft allocation耗0.099–0.103秒；24次静态operator拼装耗0.172–0.175秒（均为嵌套仪器跨度，不可相加）。

已定位guard中的逐码字Python循环：对256码本逐行发出256次对角写入。新独立回归先观察256次写入RED，再改为同一对角视图的一次fill，GREEN；跨块/末尾块、重复码字、空/单元素和手算最小距离保持原值。只改执行，不关闭安全检测、改阈值或改分配。证据 `SEPARATION_MASK_RED.log` / `SEPARATION_MASK_GREEN.log`。这代源码正在完整60epoch重新计时，不能凭微检查声称达标。

以上相对证据路径位于 `results/research/issue252/`；后续生产变更会使旧准入凭证失效，正式开训前必须重新生成。