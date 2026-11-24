## C6 前置：修改生产训练路径前冻结估计器

`CONTRACT.md:4ca0c421b397b20e13c0ab970a7e69cdf9752b7a947fb4d8f5ee2986f843bb92` 的 phase2 已在实现前冻结。独立 reference.py 与4项新测试先写；`RED_ESTIMATOR.log` 实际红灯（生产 estimator 尚不存在），未把未来实现输出当答案。

决定：保留完整1/2-hop诱导 ego 与可学习 GNN，先用静态局部稀疏算子做数学精确共享；不采样/冻结 GNN。每步每 rank 从全 valid-centre 置换循环取1中心，从原全表均匀无放回取最多32 triple，从整个原分量均匀无放回取最多32 reference；真实硬选码/软梯度照旧，辅助商品集合补到256满足原 floor。所有真实重建 batch/训练预算不变。结构 Loss 每个优化器步真实反传，lambda 仅初始实际梯度比冻结。

明确偏差：有限参考均值之后的 Huber 非线性、辅助有限集合的硬分配/floor/全局归一化均不同于原全目录目标。仅对固定外部 q 的原始 kappa 证明参考均值无偏，**不声称实际全 Loss/梯度无偏**。完整28391948行及全参考最终评估保留。记录每 epoch 真实覆盖与更新次数，不冒充全表训练。

选择依据是已完成的原路径四卡 profile：每 rank 真实1024-item步202.74–202.75s；嵌套 ego 构建约152s、GNN24–25s、全目录88–89s，不能相加，也不是 epoch。原始4份报告、真实 PID/配方与源码快照绑定于 `BASELINE_PROFILE.json:575a96366375d95b9799d439c1f4ac218e12ebc05ff8abdaa658da7f8aece0ed`。尚未达到性能门、未启动正式训练。
