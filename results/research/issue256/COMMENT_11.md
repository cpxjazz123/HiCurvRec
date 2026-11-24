R4/R5刷新中的新工程FAIL：机器entry/graph已PASS、候选124+框架64测试通过、真实四卡验证PASS、30epoch inclusive均值0.7077311168s；但实际steady_profile退出1。原文：`_calibration_batch`读取`loader.generator`时，`AttributeError: 'MeasuredLoader' object has no attribute 'generator'`。该测量包装器没有透明暴露真实DataLoader的sampler/generator。四rank失败记录：`steady_profile/issue256_r01_steady_profile_228840_584776074/rank_*.json`。没有拿前一代profile当当前通过凭据；collector未执行，现有旧汇总不用于派发。

继续修：先用真实SteadyTracker包装真实DataLoader补可执行回归，再补包装器属性委托；不在训练里特判/绕过包装器。同时恢复profile独有执行依赖的源码哈希覆盖（profiling.py/profile_units.py），加真实漂移检测回归。属于测量层兼容/身份修复，机制与冻结协议不改；保留失败代，再生成完整当前源凭据。
