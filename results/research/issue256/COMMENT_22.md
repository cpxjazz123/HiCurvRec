Stage3已实际进入冻结四卡Trainer，不只是launcher存活。2026-11-24T19:42:11Z已完成28/150epochs，loss=2.2251308250427244；真实GPU worker为858572/858573/858574/858575，每卡约28370MiB。Trainer报告ddp4/world_size4、每rank训练4096/global16384、infer1024，和MCP派发一致；error_markers为空。

该运行真实目录：`results/stage3_T5Train/TIGER_module_budget_prefix_rq/logs/Amazon_2023_Instruments/Nov-25-2026_06-36-15/`，前一初始化阶段的慢等待来自已观察到的Lustre文件读取（cl_sync_io_wait，打开torch FSDP pyc），未中断或重启作业。实测训练记录保存在`rounds/r01/STAGE3_FIRST_LIVE_TRAINER.json`和当前`STAGE3_HEALTH.json`。最终test仍未产生；继续完整150epochs。

本轮Stage2精确审计、全量诊断及独立复算已推送commit`9a96a9cf5229ded0e34f9a25ed774dfe1564e24e`，本地/远端main SHA相同；正式训练生产源码仍是`0cdfec9b024e568b5b08d96e777033844d66635e`，不混淆证据提交和生产源码身份。
