C2/C5/C6 从NOT_RUN→PASS（最终模型源码cf0fd50c，未复用旧358db180执行证据）。四卡控制器实际退出0；run `issue251_validation_2660533`，torchrun2660533，workers2660570–2660573，world4/port50202/seed42；四rank均PASS。监督程序另打印 `REAL_CONTROLLER_EXIT_AND_FOUR_RANK_PASS`，不接受此前空的提前通知。

C2：24587商品全部产生两个完整ego token，覆盖1.0；持久化attention[24587,5,3]与w[24587,5]，两个轴最大归一误差5.96046e-8/1.78814e-7，padding null权重逐位0。C5：rank0完成全部28,391,948合法三元组、24,326中心及完整组件参考的真实端到端回传，Loss0.9066628217697144，208.705182839185秒，零优化步；没有抽样或batch内参考替代。四rank不同中心的真实DDP梯度另做独立均值对账。

C6：真实原损失/曲率损失逐模块32行证据，8组参数均更新；曲率到Encoder/码本/GNN/Q/K/V/score均非零，到Decoder为0/8个None（预期）；切断实际代价后门控路径为0。lambda=0.001056040870025754，来源真实初始encoder范数比。微步后24580项硬SID变化，gate变化0.0001247876789420843，实际代价变化9.902761459350586。rank间梯度/参数最大差0，DDP对独立均值相对误差1.1781930986387579e-9；checkpoint→SID重导出精确相等，rank0推理无collective。

命令 `/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 -m tools.issue251.collect_evidence`，退出0。绑定：`RUNTIME_SUMMARY.json:d00f5d545e7f6cf2777378fcc17a8326829747ed2e4a9b5dedcd3ae4738a5e72`；`MODULE_CHECKS.json:815c8baf6de3b1907490dd760a9abb2e0ad4150e7b4509b56e40f9d21de1e0e8`；checkpoint `e43cdafb75f45b3d1a960f6d2d42e517dc4a0b45751aad33e1dced35c5af7018`；SID `8ce6603faf07d7c0519722525d37e3c65c25fe42d092746400a726769835a339`；attention `e125533247976777f32156366a0e79d4ccae44dda8ba1e90130fa3d39fa6f0c6`。逐源码SHA在summary/source和`FINAL_SOURCE_BINDINGS.json`。总rank0耗时1085.9484秒，CUDA峰值9,144,988,160字节；因与入口并行，只作诊断成本，不作性能横比。

独立硬q/连续z全量评估已开始；C7仍等待新公开回执并受冻结图合同冲突阻断，不因四卡通过而写成准入通过。无正式训练。
