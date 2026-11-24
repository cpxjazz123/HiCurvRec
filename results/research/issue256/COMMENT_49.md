R02 最小组件辨别已得到实测故障：相同冻结特征、拓扑和权重的新消费者，4/4卡的静态入/出均值不逐位相同；仅2-hop缓存每卡约518万项差异。6组组合affine参数和mask均相同，同一消费者重复tokens逐位相同。该探针使用highest计算策略，不能把其token差1.7881393432617188e-7直接当成Trainer high策略的gate差量。

证据 `results/research/issue256/rounds/r02/STATIC_CONSUMER_DIAGNOSIS.json` sha256 `37e70c7337fed6e0672918f009ac17eff65c1c9f2c2d999498247840dd6b363e`，命令 `python3.9 -m tools.issue256.r02.diagnose_static_consumers`，标准四卡/50202，exit0、353.12s；全四种完整冻结ego缓存逐项比较，没有重建资产。现已证实静态均值构造不可复现，但尚不能断言它解释了全部预算失败。下一步先为真实消费者补红测试，再将CUDA静态均值改用已有固定逐行归约核；公式、完整边集、dtype、阈值和训练协议不变。实际四卡预算、epoch51与全部准入必须重新通过后才派发。
