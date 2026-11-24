## C12 新一轮检查 FAIL：公开入口环境与两个旧 fixture

真实四卡 `launch_validation` 在结果目录启动时，4 rank 均在导入新 `model/estimator.py` 的 helper 处报 `ModuleNotFoundError: No module named 'tools'`；尚未进入训练。证据 `results/research/issue253/validation/launcher_4135249/logs/validation.log`，launcher 退出 1。这是新 import 早于模型原有仓库 root bootstrap 的工程故障，正在补独立 isolated-python 回归并修复，不设置 PYTHONPATH 绕过。

全量源码绑定 suite：deterministic **176 项中 2 ERROR**（旧测试 fixture 仍为未注册 candidate 身份及 issue252 contract），reference **11 PASS**，mutation **2 PASS**。证据 `CONTRACT_SUITE_1.log` 与 `contract_tests/1795507912998728865/`。将迁移 fixture 到新身份及新 contract，保留原来的源码漂移检测判据。

另发现继承的 sync-free tests 自行调用 CUDA，不应通过 CPU driver 直接执行。将把这两个实际 CUDA 用例纳入标准四卡 kernel diagnostic，CPU driver 只执行 CPU 路径；不把漏跑或 skip 当通过。

优化 kernel 的首次标准四卡数值复验已完成：run `issue253_kernel_validation_4117316_581099998`，8 tests/rank、0 failures/errors/skips；后续移动上述 CUDA 用例后会对最终绑定再次执行。正式训练未开始。
