R02 静态缓存修复已绿：原红断言对应的目标CUDA回归4/4卡exit0；完整unit142/64/15逐例通过且无跳过；真实四卡独立raw-replica的17模块初始全局预算现在4/4 PASS，实际lambda1.0698641972339828e-6。`STATIC_CACHE_GREEN_RECEIPT.json` sha256 `278fc4f60f265b106f3c2e024101b5a2c89bd156da0fa79e24c6ab65bd03214b` 绑定生产源与原始报告。故障得到定位并修复，没有放宽2e-6预算容差。

bg_509在更晚的共享/独立两次GNN门控比较失败：4/4 `shared and independent geometry gate values differ`。实际一个epoch的前向、反向和参数更新均已记录，但epoch51、inclusive性能、机器入口仍未执行；不把unit或预算绿当成完整准入。继续用同权重、同数据的逐层batch/精度辨别诊断排除数值路径问题，不进行正式训练或追分。
