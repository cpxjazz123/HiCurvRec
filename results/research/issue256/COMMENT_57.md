R02完整143/64/17逐例unit已PASS（包含生命周期、安装幂等性、相同实例及新门控回归），无skip；`FLOOR_LIFETIME_GREEN_RECEIPT.json`绑定原始输出。实际四卡随后到达更晚的native replay，4/4报 `recorded hard IDs do not replay the frozen native allocation`。

本次已定位为验证器模式错误：实际辅助population以balanced=False选码，quantize随后恢复model.train()；旧验证器错误地用恢复后的全局True重放两种population。2x2真实VQLayer反例给出aux=[1,0]、balanced=[0,1]，旧helper被同一指定断言捕获（exit1、3.17s），见 `AUXILIARY_MODE_RED_RECEIPT.json` sha256 `eb12d463cbd8329942b33e4eb7822153d2a3d7b0ea604d79a3c30e7a28a6063d`。初次reproducer根目录标记错误的import失败另存，不冒充红测试。

现在验证器显式接收每条捕获记录的balanced标志，移除隐藏全局模式；新增false/true两侧实际硬分配回归。生产训练/损失/分配/容差均未改，bg_514正在重新完整检查；仍未正式开训。
