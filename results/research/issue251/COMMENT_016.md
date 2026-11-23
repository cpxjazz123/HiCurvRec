四卡验证实际进程已确认：controller2650570、torchrun2660533、workers2660570–2660573，`--master_addr=127.0.0.1 --nproc_per_node=4 --master_port=50202`。原单卡入口2456764仍在计算。

harness曾提前通知bg_76“finished(exit0)”，但进程设备和真实进程均表明它还在启动/运行；据此启动的收集器因尚无rank报告退出1，未计作模型证据，也未重复派发GPU任务。已向工具QA报告，并保存 `PREMATURE_COMPLETION.json` 与原始错误日志。后续只在真实controller结束、四份完整PASS报告和退出码对齐后收集；任何提前读产物的结果均不接受。
