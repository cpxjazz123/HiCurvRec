最终模型源码cf0fd50c的独立四卡受控验证已通过正式 launcher 入口派发：`/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 -m tools.issue251.launch_validation`，监督作业bg_76，输出 `results/research/issue251/runtime/`。只做冻结的校准、梯度对账、单微步、全28,391,948目标回传和checkpoint→SID重导出，不是正式Stage2/Stage3。

该验证与已在运行的单卡真实入口smoke并行；两者不共享模型状态或结果目录，静态资产只读。此次并行耗时只作实际诊断记录，不与先前独占GPU的耗时作性能比较。GPU分配、四rank和50202仍由现有launcher执行，未手工钉卡或改冻结优化配方；若资源/端口闸门拒绝，按实际结果诊断，不绕过。真实run_id、worker PID、显存与退出码在完成后的原始rank报告中对账。
