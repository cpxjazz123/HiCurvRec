C5 校准公式缺陷已按真实模型/真实训练 batch 完成红→绿，不调整契约或根据 Recall 选权重。

- 新回归：`stage2_RQ-VAE/TIGER_prefix_curvature_gate_rq/tests/test_calibration.py`。使用实际768维商品嵌入、合法训练target、真实三层256码模型、冻结训练采样器和实际原始 RQ-VAE/截面损失；独立求 encoder 梯度比。测试仅隔离λ分子，完整目录距离尺度估计保留在真实入口/四卡诊断中。
- 红灯（`CALIBRATION_RED_BATCH.log`）：原实现λ=`0.0008543586591258645`，完整原始损失梯度比应为 `0.000805870073939786`；指定相等断言失败。数值是该回归batch，不冒充正式初始化值。
- 修复：校准分子改为完整原始损失梯度；另行真实记录重建梯度、原始/重建各自与结构梯度的夹角。诊断中“相对重建”的逐层夹角也改为真正重建项，不再用总原始损失冒名。
- 绿灯（`CALIBRATION_GREEN.log`）：同一回归1 passed，并断言校准没有更新任何模型参数。模型源码SHA `7cea8be51adcaee041da2035d2b8285d46488f8197f4460726a13d0c92da1c28`；测试SHA `52e70e6ac2fc4f098da5ccab382896d646703a3993d02320f6377c324544ce0e`。
- 修复后的只读评估/配对/派发凭据工具回归：54 passed（`TOOLS_REPAIRED.log`）。包含CUDA提取凭据绑定、nullable零基rank边界、同源Stage3及逐用户账本、过期/缺rank凭据拒绝等。

完整候选+工具源码绑定测试正在执行；接着重做真实四卡模块/DDP/入口、稳态性能与剖析，凭据齐备前新派发器会拒绝正式运行。获批的修正代次仍固定为 `issue255_formal_r1`，尚未派发；旧run保持INVALID_IMPLEMENTATION，未启动Stage3。
