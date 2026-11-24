R02 检查出现新FAIL：137候选+64框架+11控制器测试通过，四卡profile完成；真实Trainer校验在共享/独立分支的离散分配等价检测失败（rank0目标568.999695 vs567.841797、旧界0.0111763477；rank3重建hard IDs不同）。因此禁止正式派发，后续benchmark/epoch51尚未执行。

已定位旧检测器的错误数学前提：它把实际冻结的贪心floor repair当成全局最优匹配。2×2手算反例已由当前生产分配器及失败测试实证：两份cost只差0.001953125（逐行最大差之和），合法贪心目标却从3变为1.9990234375。指定旧断言因此错误拒绝；不能更换分配规则或放宽阈值掩盖。

修复范围仅验证器：逐份原始cost精确重放真实硬分配、记录离散边界差异；在同一已记录hard分支下独立核对共享GNN与两次GNN执行的连续loss/VJP，保持原有数值容差。真实训练loss/DDP更新、硬prefix解码及全目录checkpoint SID精确导出不放宽。数学契约、正式模型/超参数不变。

证据 `results/research/issue256/rounds/r02/SHARED_ORACLE_FAILURE.json` sha256 `ce30d6e996fb3e751e9bc762139e0d4135196f84eaf148223d193f7f5bdf277e`；GREEDY_ORACLE_RED.log及pre_shared_oracle/保留旧源码、212项通过记录和失败原文。继续修复并生成全新完整凭据，不复用过期profile/测试充当通过。
