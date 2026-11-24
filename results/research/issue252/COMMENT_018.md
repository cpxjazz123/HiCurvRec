## C8 子检查：实际四卡 CUDA 数值/梯度/码字与资源 RED→GREEN

几何资源RED `geometry_1795475123987140614_261598_12820a31ab0f`：四rank独立数值/梯度、16组真实分配及18组floor对照均通过，实际1024×256×32的forward/backward为551个CUDA compute launches，指定<100断言失败。随后仅融合相同纯算术，不改dtype/度量/温度/硬分配/学习权重。

GREEN `geometry_1795475478322520704_305462_49c1d20a2c26`：四rank全部PASS、源前后相同。每rank86项独立数值/梯度检查、16项真实硬分配检查、18项与原始源码快照floor逐ID/metadata对照通过；9次实际CUDA测量：expert64 kernels、mixed64、fixed63（各3次），梯度有限。原容差F32 1e-5/1e-4、F64 1e-9/1e-8未改。冷编译诊断总308.88s明确保留；所有Inductor/Triton/temp缓存位于仓库results/research/issue252/compiler_cache。不是epoch速度凭据。

稀疏资源修正后的RED `issue252_sparse_validation_239075_577802695`：真实F32 CUDA每rank指定断言抓到8次native CSR重建；精确复用反向support的adjoint后，`issue252_sparse_validation_367173_577893012`四rank各1项生产backward测试PASS，原生转换计数必须为0，数值/全部参数梯度仍一致。CPU独立dense/gradgrad及静态均值16项PASS，另entry拒绝单卡旁路1项PASS，日志均保留。

新源的behavior_graph独立诊断PASS，OPTIMIZED_BEHAVIOR_GRAPH.json，identity仍fe87a3da4a5afcb7616773950b7b57ad4df5db1adcb357e59e7122c89c544212。当前正在执行最终源码完整数学套件与真实loss逐模块/DDP/同checkpoint导出诊断；尚不能把C8整体或C7/正式训练判PASS。