R01正式训练已通过唯一launcher启动，非smoke：run=`issue256_r01_formal`，arm=`TIGER_module_budget_prefix_rq`；冻结预算四卡/global4096/seed42/3000epoch、端口50202。代码与准入已提交推送，local/remote main均`0cdfec9b024e568b5b08d96e777033844d66635e`。

真实torchrun PID434483，四worker PID435981/435982/435983/435984；18:48:45Z观测处于初始化、已记录execution_context，epoch0，checkpoint尚未产生，不把初始化说成完成。正式不可变源码/结果：`results/stage2_RQ-VAE/_runs/issue256_r01_formal/TIGER_module_budget_prefix_rq/{src,result}`。`FORMAL_RUN.json`绑定注册、完整派发凭据与快照；`FORMAL_START.json`保存PID/GPU/真实观察。后续只监督这一轮，不另开同机制追分运行。
