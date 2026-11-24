R02 静态缓存故障已先补真实红测试：`python3.9 -m tools.issue256.r02.check_static_cache` exit1、17.70s，4/4卡均在调用阶段被指定断言 `fresh frozen feature means must be bitwise identical: incoming` 捕获；不是崩溃、导入失败或超时。`STATIC_CACHE_RED_RECEIPT.json` sha256 `93650b91e6b43703eb98740f758c559b13f9fcaf14c77666f3df1a505c1de658` 绑定原源码、四卡报告和失败用例。

实现现在仅将CUDA六特征静态均值改用已存在的完整逐行确定性归约核，CPU路径不变；不减边、不建资产、不改公式/损失/权重规则/容差/配方。旧生产文件逐字保存在 `rounds/r02/verification_generations/pre_static_cache/`。当前运行目标绿色回归和完整source-bound准入流水线（bg_509）；在实际四卡、epoch51、inclusive性能和真实入口全部通过之前，不宣称修复完成，也不正式开训。
