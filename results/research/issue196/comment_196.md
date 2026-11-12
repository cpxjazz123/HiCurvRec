## 撤销决定已执行，三项验收全部核实

1. **#195 已关闭**（owner 指令取消），并在关闭后补了一条**更正版**说明（首条因 shell 反引号展开缺字）。
2. **全仓无消融物**：`ls -d stage2_RQ-VAE/TIGER_abl*` 空、`results/stage2_RQ-VAE/TIGER_abl*` 与 `results/stage3_T5Train/TIGER_abl*` 空、`pgrep -af TIGER_abl` 空。
   三个开关仅存在于代码并各被廉价接线检查覆盖（`tests/test_sdh_off_state.py`、`tests/test_sdh_kernel.py`，树内共 65 项测试通过），**未训练任何关闭态臂**。
3. **后续 Issue 形状**已写入 `results/research/issue196/CANCELLATION.md` §5：必须单臂、自带预注册指标与关闭条件，不得以"先做个对照看看"提出。

## 一处如实交代（边界情形）

#194 验收 1 需要证明 `BEHAVIOUR_OFF` 与冻结基线一致，为此存在**一次一 epoch 的 CPU 单进程对照**（`results/research/issue194/off_equivalence.py`，各 3 次重复）。它不是 Stage2 臂：未走四卡冻结配方、未进 3000 epoch、未产生用于比较的 SID、未登记为结果；结论是步数逐位一致、数值在噪声内、**逐位相等在本 harness 不可测**。此处公开该运行的存在，避免"未运行关闭态"的说法与事实不符。

## 顺带：#196 正文里 #194 的快照已过期

#194 已按预注册收线：Stage2 `collision = 0.2119005979 < 0.40` 通过 §89；Stage3 `test_recall@10 = 0.05598983269207333 ≤ 0.056216` ⇒ 落入**关闭分支**；§6b 已归档到 `r2:genrec/72/`（14 文件，逐文件 SHA 在 `r2_ledger.json` 的 `72` 项）。Issue #194 已关闭。

## "分离问题"留档（不再用实验回答）

#193 P0-3：Pearson **0.10530**、Spearman **0.10746**、mean KL **2.03845**、零共享对 **94.95%**；单臂读数 #194 **0.05598983**、#181（缺陷版 + 4× 批量）**0.054266**、#179 参照 **0.056216**。

**依据 commit**：`5de7ebdfa`（核实记录与撤销）。
