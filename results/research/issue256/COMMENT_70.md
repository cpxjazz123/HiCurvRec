R03 正式 Stage2 已跑完 3000 epoch（2136.841s，final-epoch checkpoint，60 次原生块刷新，24587 个商品，三码碰撞率终值 0.05445967381136374）。

**机制签名与预注册预测不符**，按 CONTRACT §3 的失败判据先记录：导出 SID 的 layered 指标（同一 `tools/issue252/layer_metrics` 口径）为 distinct two-code prefixes **9257**（预注册要求 ≤6000；R02 8840、混合族最好 4159→4151、基线 7374），绝对 L1 前缀命中率 **0.05961**（要求 ≥0.10；R02 0.0643、基线 0.0925、混合族最好 0.1599），L3 refinement 1.90 bits。也就是说：**role-conditioned level-1 没有把最终分配变粗，反而比 R02 更细一档**。

训练中段的碰撞率确实变粗过（epoch100 0.329、1000 0.535），但随后单调塌回 0.054——按 role 共享几何只在中段起作用，最终 latent 仍然可以逐物品区分，共享度量没能阻止近乎唯一的编码。这正是 CONTRACT §3.3「L1 命中率 <0.08 → 该耦合假设被证伪」的情形。

按 Issue 的 R6 规则该轮仍要跑完冻结 Stage3 并记录最终 test（不做早停、不换种子、不重跑）；现在的导出审计（同一 checkpoint 精确复现全部 SID、archive binding、分层指标落盘、post-training 梯度探针）正在四卡上执行，完成后立即 `stage3_start`。Issue 保持 OPEN。
