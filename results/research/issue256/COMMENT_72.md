更正 R03 诊断中的一个数字（结论不变）。

先前对 R03 导出 SID 的熵计算把第四列（`_extend_collisions` 写入的逐物品碰撞计数器）当成语义码，得到 H(L3|L1L2)=2.0193 bits、23247 个三元组。只取三个语义码重算：**H(L1)=7.4596、H(L2|L1)=5.1067、H(L3|L1L2)=1.9028 bits**，**23248** 个不同三元组（碰撞率仍为 0.05445967381136374，与导出口径一致）。

结论方向不变：level-1 已接近均匀（7.46/8 bits），逐物品的唯一性仍集中在 level-3 在前缀条件下的 ~1.90 bits 里；R03 判定（机制签名与最终 test 双双被证伪）不受影响。证据已同步修正：`rounds/r03/ROUND_RESULT.json`、`CHECK_1.json/.md`、`ROUNDS.json`、`PROGRESS.json`，并把修正后的测量写入 `rounds/r04/DIAGNOSIS.json`（含方法与 SID sha256）。

R04 已按修正后的基线预注册：把 role 混合移到 level-3 代价（level-1/2 保留逐物品门控），预测 H(L3|L1L2) ≤ 1.2 bits、三元组 ≤ 12000、validation R@10 ≥ 0.0583；失败判据 H ≥ 1.5 bits 或 R@10 ≤ 0.0576。合同在开训前写入 Amendment 1。
