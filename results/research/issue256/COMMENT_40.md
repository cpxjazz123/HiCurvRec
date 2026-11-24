R02 单一修改已在实现前冻结：沿用原始O/C及17模块全局VJP预算，仅将一次性lambda改为冻结原生EVAL_INTERVAL=50的块边界重算（epoch1,51,...,2951）。不新增可调周期/损失/归一化，不改变任何训练或推荐协议。

训练侧依据：R01最终四卡13/17组超初始预算，最大9.0859271222，仅全局sharedScore出现负原始点积；局部Key/Query0冲突不成立于全局。完整validation的L0前缀失配定位并不证明因果，根因仍未知，备选解释已列明。

旧活跃树已完整归档后，才从唯一冻结baseline做15文件完全一致的源码副本并只改ARM_NAME；无旧权重/cache/results。复用109项assets，manifest sha256 cfbd9d9c0d3c5282f9fdd0c0b28798cbf7a86479c2626b702e499b589acc5b6b；无新静态资产。

契约 `results/research/issue256/rounds/r02/CONTRACT.md` sha256 `f80ae1d7b07268cf53d189e10fb447bf483aef5da913f2fd0b1faf3230fc519b`；DIAGNOSIS.md与BASELINE_COPY.json同目录。独立LP/手算/变异/真实loss四卡与epoch51接线、含50/51周期工作的30连续epoch<=0.8s均须通过才派发。每次anchor预算、最终全局方向、全量validation及严格test>0.065为分别可证伪预测。
