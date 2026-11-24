## C8 新真实入口 FAIL：paired曲率编译在校准阶段失败

`genrec_env_v2/bin/python3.9 -m tools.issue252.launch_validation` exit1，run `issue252_validation_379303_577903371`，67.91s；rank0原始报告保留。失败发生在真实Trainer的 `calibrate_curvature_weight → sampled_curvature → fixed_squared(x[:,None,:],y[:,None,:])`，TorchInductor/Triton `InductorError: RuntimeError: PassManager::run failed`。尚未进入正常optimizer epoch，不能归因于科学结果、不能开始正式训练。

首中心15918的完整目标28条，实际paired布局为28×1×32；之前CUDA独立proof覆盖外积prebroadcast，没有覆盖该真实逐对可微布局。因此即使原86项数值/16分配/18floor和64-kernel proof全绿，也不替代真实入口。

正在补逐对可微输入独立参考/全部输入梯度的CUDA回归，复现后仅调整单位轴的等价执行布局，不改度量、loss、参考集合或容差；不使用exception fallback、不恢复NxN距离取对角的重复计算。

同代完整数学套件已实际完成：60确定性+11参考+2变异测试全PASS，全部15有效变异被指定断言抓到，CONTRACT_TEST_RUN_2.log，source-bound receipts在contract_tests/1795475842032520883。入口修复若改源码，这些最终凭据将失效并重新生成，不能沿用旧PASS。