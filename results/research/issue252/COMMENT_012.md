## C4：原路径与优化路径四卡 profiling 已实测

`python3.9 -m tools.issue252.launch_profile` 原路径实际四卡/global4096，仅1个真实 optimizer step、0个完整 epoch：初始化211.38–211.41s，仪器化一步202.74–202.75s，峰值约9.164GB。BASELINE_PROFILE.json 绑定保留的真实源码快照与4份原始 rank 报告；旧 producer 的 native device 微秒标签错误只在 collector 中按1e6更正，未修改原始数据、未重跑。

`python3.9 -m tools.issue252.launch_optimized_profile` 同配方优化诊断：初始化64.71–64.75s，冷启动仪器化一步4.986–5.066s，峰值8.266–8.494GB；逐阶段真实耗时、调用次数、形状/重复量及 allocator 峰值在 PROFILING.md / OPTIMIZED_PROFILE_INTERMEDIATE.json。rank0 guard 含2.944s NCCL 等待其他 rank 首次装载静态布局，不能当稳态成本下界。原路径只记录 rank-global 显存上界，不虚构其逐阶段峰值。稀疏 logical numel 不当作物化空间/实际工作量。

历史30 epoch日志的计时边界和缺失来源见 HISTORICAL_TIMING.json：旧 timer 不含 post-loop通信/评估，旧硬件/执行源码不完整；不作为本臂速度凭据。

C4诊断已完成；这不是C7通过。当前优化画像为中间源码代，仅1步，仍须最终源码60完整epoch、31–60连续窗口（含epoch50全评估）实测平均≤0.5s。未开始正式训练。

实际修复检查另见 REVIEW_CONTRACT_ASSETS_TESTS.log（21 PASS）和 R2_VERIFICATION_TESTS.log（10 PASS）；后者是离线生产路径回归，不宣称已经上传R2。