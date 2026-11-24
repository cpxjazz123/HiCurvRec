R4 红阶段已取得实际断言失败，而非把超时当红：经唯一launcher运行`python3.9 -m tools.issue256.launch_budget_check`，四张卡各用真实1024商品batch、实际原始RQ-VAE/曲率Loss；四个rank均在`module budget exceeded`断言失败，launcher exit1。旧校准虽然encoder比值约1，Query2结构/原始范数比约7.84、Query1约5.62、codebook2约4.18。真实日志：`rounds/r01/budget_regression_launcher/logs/budget_regression.log`；逐rank记录在`budget_regression/`；红阶段源码冻结在`red_source/`。这些是CPU诊断fixture以既有冻结权重验证实际生产校准器，不冒充新候选正式初始化或新实验结果。

此前CPU/full1024与第一次GPU启动各180秒超时，均不计为红/通过；保留原日志及`budget_regression_launcher_timeout/`。改用带调用栈诊断的完整四卡场景后得到了指定数值断言失败。现在按已冻结CONTRACT实现，不放宽预算断言；正式初始化/全局平均VJP还需实际Trainer独立验收。
