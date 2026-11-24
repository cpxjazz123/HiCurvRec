四卡最终权重判别诊断已真实执行并退出0：4×1024=4096训练样本、同一冻结lambda=1.6054239040386165e-6，实际original/C VJP先SUM/world再取范数/方向。17耦合组中13组超出初始化预算；最大比值为Query0的9.0859271222。global sharedScore的归一化original-dot-combined为-0.6988326877，存在该batch上的原始梯度反向冲突。

重要反证：rank0局部显示的sharedKey/Query0原始上升方向，在全局平均后均消失（归一化dot分别+0.2886/+4.0862）；不能把局部符号直接外推。全局仍有时变失衡，但不是已经证明的Recall根因，也不是R01违约：R01只承诺初始化预算。测量只覆盖一个fresh诊断RNG的最终训练batch，不代表所有3000epochs或AdamW实际步。

没有optimizer、重新校准、额外正式训练、SID改写或test访问。实际worker1300364/1300365/1300366/1300367，sampler step17994，seed42/43/44/45。证据`rounds/r01/final_global_gradient/GLOBAL_FINAL_GRADIENTS.json:2f8ab0d79bb61ea4989fe44c8c2f8ca981f6cdd2d1b00117134663afe94571ee`；命令Stage2解释器`-B -m tools.issue256.final_global_gradients`。这使“初始化约束不覆盖后期全局梯度”成为实测线索，而非凭test错误商品猜原因。
