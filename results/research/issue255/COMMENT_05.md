C9 检查循环发现并修复的工程问题（仍在执行完整绿灯验证，未宣称全套 PASS）：

1. `model/structure.py` 空输入返回 `(0,2,32)`，与四 token 契约冲突。两个独立回归断言实测失败，现修为 `(0,4,32)` 及对应 mask。只改活跃树；正式运行的不可变 `src` 未修改，该空输入分支不在本次真实训练人口中。
2. `benchmark_entry.source_binding` 漏掉实际 launcher 与导入的仓库依赖。已有工程回归实测捕获（改变 launcher 后凭据未失效）；活跃树补齐依赖哈希。旧性能窗口/正式训练仍绑定各自执行时的冻结源码，不把新凭据冒充旧凭据。
3. `results/research/issue255/run_tests.py` 仅靠 `stage2_RQ-VAE/` 目录寻根，误认 `results/` 为仓库根，启动验证失败。新增独立 `tools/issue255/tests/test_verification_workspace.py`，用同时存在源码/结果同名目录的真实文件树复现红灯；修为同时要求框架标记文件。临时产物均位于仓库内。
4. 旧 adjoint 数值参考缺少新增固定 token 类型编码，已扩展为四人口参考；不放宽数值容差。删除一个无效接线测试：它把每个 loss mock 为零、比较距离与自身、却断言总 loss>0；不将其改成另一种同义反复。真实逐层损失/梯度/前缀隔离由实际模型诊断和现有数学测试承担。

证据：`TEST_FAILURE_DETAILS.log`（5 fail/1 pass 的完整定位）、shadow-root 红灯输出、`tools/issue255/tests/` 工具测试此前20项通过。受影响的全套候选/工具验证正在重新执行并绑定最终源码 SHA。没有重跑正式 Stage2、调参、换种子、放宽碰撞门槛或改写冻结数学契约。
