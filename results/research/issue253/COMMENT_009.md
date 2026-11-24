C1独立候选及冻结协议核验PASS（本次原型源码代）：`C1_PROTOCOL.json:1e2d978bc371a823f0e7add2116acfae5dcdc0127e3ffff3c675f6948485c52f`，真实CPU introspection exit0，未使用CUDA/未训练。输入768、latent32、三本[256,32]、Query[3,32]、曲率[-1,0,1]、floor1、仅第3层use_sk；源码常量3000/rank1024/seed42，config与冻结TIGER逐字节一致，assignment_floor与#252实际执行快照SHA完全一致。初始复制来源及15文件一致证据见source_provenance.json；旧#252目录/结果未覆盖。

真实四卡profile另证world4/6steps；不声称三层Sinkhorn。后续同一步复用/kernel改动不改该协议，最终源码冻结后仍须重新绑定对应证据，当前不是正式训练准入凭据。
