R03 正式 Stage3 已跑完并完成独立复算：150 epoch、四卡 DDP、冻结协议（beam 20、seed 42、无早停），最终 test 覆盖完整 57439 行。**test_R@10 = 0.0490085133794112**（R@5 = 0.03189470568777312），低于 0.065 阈值，也低于 R02 的 0.05266456588728913 与基线 TIGER_RQ-VAE 的 0.0557。

**本轮是负结果，机制在 Stage2 已被自己的判据证伪**（CONTRACT §3）：
- distinct two-code prefixes **9257**（预注册要求 ≤6000；R02 8840）；
- absolute L1 prefix carry **0.05961**（要求 ≥0.10；R02 0.0643；判据的硬线是 <0.08）；
- 训练中段的碰撞率确实粗过（epoch100 0.329、1000 0.535），但收敛时回到 0.0545，共享的角色度量没有把最终分配变粗。

完整 57439 行推荐复核（`recommendation_validation/VALIDATION.json`，不参与选点、不触碰 test）：validation_R@10 = **0.05430108462891067**，同样低于本轮预注册的 0.0583，也低于 R01/R02——这是第三个独立的证伪面。

**为什么没变粗（本轮新测的定位证据）**：对导出的 SID 直接统计条件熵，H(L1)=7.4596、H(L2|L1)=5.1067、H(L3|L1L2)=2.0193 bits，三码不同的三元组 23247/24587 个。即：level-1 已经接近均匀（256 码全用），逐物品的唯一性信息集中在 **L3 在前缀条件下的 2.02 bits** 里（在 256 个码里挑约 4 个）。所以"把 level-1 的度量换成角色共享"并不能改变三元组的粒度——下一轮必须把行为信号耦合到 level-3 的 refinement，而不是 level-1 的度量。

归档与状态：四件套已按 §6b 上传到 **r2:genrec/104/**（131 个文件、23 个 recall 键，台账 `results/research/issue139/r2_ledger.json`）；独立复算 `rounds/r03/STAGE3_RECONCILIATION.json` 逐键一致（strict_target_met false）；`CHECK_1` 记录 16 PASS / 1 FAIL（C17 未达阈值）。Issue 保持 **OPEN**，候选不采纳，下一轮在 level-3 refinement 杠杆上预注册单点变更。
