C8报告覆盖复核发现可修复缺项：上一轮已从原始checkpoint/SID独立算完硬q和连续z的全28,391,948行曲率/距离，但z尚未列出自己的strict held-out和decoder重建诊断；不以硬q数值替代，也不把缺项标NA/PASS。冻结损失、图、目标、尺度和模型源都不改。

先补两条手算红测试：z=[0,3,1,5]的正/负pair距离3/2、AUC0，与硬q=[0,1,3,5]的AUC1相反，可抓到错用硬q；独立重建误差应MSE2.5、meanL2=1.5且拒绝NaN。命令 `python3.9 -m unittest tools.issue251.test_evaluation.EvaluationTests.test_continuous_heldout_uses_actual_z_not_decoded_sid tools.issue251.test_evaluation.EvaluationTests.test_continuous_reconstruction_has_its_own_actual_errors -v` 退出1，当前缺少这两个生产评估入口（`RED_CONTINUOUS_PARITY.log`）。继续补齐同一严格评估路径并重跑全量评估；此前四卡/模型执行证据不受评估器补项影响。
