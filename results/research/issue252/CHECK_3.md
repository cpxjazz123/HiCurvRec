# Issue252 CHECK_3 — final source and completed experiment

Issue was re-read live before this check; its original body SHA remains `3e9b8ca4018cd9d0a43648efc42cd6a12ef27bc892fdb981590030bea37849eb`. All14 original criteria follow verbatim. Judgments use only the recorded prospective owner amendments; original unmet requirements and operational recording qualifications remain explicit in REPORT.md / VERDICT.md. This check does not claim scientific adoption.

## C1

> 从原始训练交互独立重建/对账 canonical 图（323633有向边）与冻结 visible 图（261132有向边），逐类解释62501边的差异，核查用户划分、结构 held-out、推荐 valid/test 隔离及权重规则。禁止把隐藏行为用于训练。

PASS — raw graph difference 62501 fully classified; zero unexplained/visible-only edges and zero strict held-out/train overlap. VERDICT E1.

## C2

> 此 issue 明确授权修复该冲突：若 canonical 包含专门结构验证/测试所保留的训练用户行为，则以合法可见图为训练图，并修订规范/准入为可机器复验的具名图契约，严格证明来源和隔离；若冻结 visible 的差异源自错误构图，则按正确可见协议构建版本化新资产，重算受影响的 ego、特征、全量曲率和评估 bank。不可仅屏蔽检查或静默换图。保留旧资产及 provenance，正确资产不重复计算，新增数据持久保存至 assets/ 并更新 manifest。

PASS — named infra246-visible-v2 contract and immutable registered assets; new runtime-only indices/operators under assets/. VERDICT E2/E3.

## C3

> 实际 verification_check 及 stage2 正式准入必须 entry PASS + behavior_graph PASS；回归覆盖错图、混图、泄漏、旧缓存和源版本变化。

PASS for final current-source public call — entry PASS + behavior_graph PASS; source/graph/test identities E4/E5. Invocation is post-run, not backdated.

## C4

> 复核以前约0.4s/epoch的真实日志及计时边界/数据/硬件/global batch，区分单卡入口、四卡正式、校准、评估和导出；对当前相同配方进行真实四卡逐阶段 profiling：数据读取、ego 构建、GNN、Attention、五几何代价、Sinkhorn、全参考距离、三元组曲率、反传、DDP通信。报告每项耗时、峰值显存、调用频率与重复计算量。

PASS — original versus current real four-rank phase/timing boundaries distinguished in PROFILING.md and historical/profile receipts.

## C5

> 优先消除重复整图 ego 展开、重复磁盘解压、逐商品 Python 循环、重复全目录 forward 和重复参考距离；采用持久化静态图索引/稀疏张量、批处理、共享计算、精确代数化简、分块、融合算子与预取。不能把可学习 GNN 输出长期缓存为常量或冻结 GNN；门控与 GNN 必须有真实更新。

PASS — exact static layout/per-ego fusion removes repeated construction while actual learned branches update. Current real module proof E7.

## C6

> 全量监督资产继续保存，不重新抽样构建；训练计算可采用有明确数学依据的分块/随机小批量或周期性结构更新以达到可行预算，但必须在修改前固定监督估计器、覆盖策略、归一化、参考集合、梯度语义及与全量目标的差异，验证无偏性（若声称无偏）或明确偏差及其影响。禁止把 batch 内参考均值冒充全参考精确结果、未经声明改目标、长期跳过结构 Loss 或将其仅用于评估。完整全量曲率评估仍必须保留。若采用周期性更新，必须将其成本计入摊销 epoch 时间，并记录每 epoch 实际目标覆盖及更新次数；不能用不含结构计算的 epoch 计时冒充总速度。不得以追求速度为由取消端到端机制。

PASS — stochastic training estimator declared separately; full frozen 28391948-triple evaluation preserved for both representations. CONTRACT and E8.

## C7

> 在原冻结四卡配方（global batch4096、seed42、3000 epochs）上做短性能验证，排除初始化后报告连续至少30 epoch 的总墙钟/epoch（包括数据加载、周期性结构计算摊销、前后向、优化器、通信），以及中位数/P95、完整验证窗口耗时和预计3000 epoch成本。单独报告初始化及导出开销。目标平均 <=0.4 秒附近，验收上限0.5秒/epoch；若未达到，继续定位优化，不直接开始正式3000 epoch，不通过改计时方式、删数据、缩减 batch/epoch 或隐藏重计算达标。若证明目标与硬件/完整机制存在不可兼容的成本下界，报告实测及可行方案，明确未达标，不伪造结果。

PASS under recorded owner 0.8s amendment, NOT a pass of original 0.5s requirement. Final-source inclusive 30-epoch mean 0.667614755500108s; formal timing retained. E4/E6.

## C8

> 每项优化先红后绿，独立对照数值、梯度和硬SID；近似算法必须单列，不称精确等价。复核 GNN/Q/K/V/score/Encoder 梯度、权重干预影响真实码字、硬q与导出一致、固定度量不旁路、DDP一致、重建语义和碰撞。禁止为了速度改变原始768/32维、三层SID、容量及原Sinkhorn硬分配协议。

PASS under recorded last-level-only Sinkhorn amendment — 193 deterministic/11 reference/2 mutation suites, 15 caught mutations; actual module/DDP/export checks. E4/E7. Original three-level Sinkhorn claim is not preserved.

## C9

> 全部准入、性能与数学测试通过后，使用标准 launcher/MCP 正式执行一次四卡 Stage2 3000 epoch（固定原配方，不做多种子、不训额外基线、不调参筛选）。记录 run_id、源哈希、资产哈希、完整损失/梯度/时间/显存、门控/注意力分布及真实完成状态。

PASS for one completed immutable formal3000-epoch four-rank/global4096/seed42 generation. E6/E9. Formal rank PID fields are unavailable and not replaced with diagnostic PIDs.

## C10

> 从本次 checkpoint 导出真实 SID，重新编码严格对齐；记录前三码碰撞率、占用、重建、硬q与连续z曲率宏/微MAE、分符号 balanced accuracy、held-out结构保真与前缀指标。Stage3 碰撞硬门槛 <0.4，不能用其他快照、第四列计数器或软表示代替。

PASS for exact checkpoint/SID and complete measured diagnostics. E6/E8/E9. Removed legacy G3 remains MISSING; layer-record timing qualification is explicit.

## C11

> 如发生真实运行、OOM、数据、导出、碰撞或准入故障，诊断根因、以回归测试修复，并在授权范围内恢复必要运行；不得把故障恢复冒充多种子实验或以测试结果挑选模型。保留失败证据，不覆盖历史产物。

PASS — actual failures retained; real regressions and current complete suites passed. Final optional-key R2 reader fix is also exercised against the actual archive.

## C12

> 使用该次合法 SID 实际完成一次 Stage3 150 epochs，遵守冻结 T5 配方，不能替换 SID、提前停止或跳过最终 test。保存 R@5/R@10、NDCG、实际日志、checkpoint、SID哈希与 run 绑定，并按现有仓库规定归档 R2。完整训练完成与研究收益是两件事：如未超过历史 TIGER 或仓库科学阈值，如实报告负结果，不为追指标重复训练。

PASS — frozen four-rank150 epochs and final test completed; exact R2 bundle100 uploaded with100 file hashes verified. Negative R@10 is reported unchanged. E6/E10.

## C13

> TIGER 只用已冻结历史产物对比，不重新训练 TIGER；无同源几何、重建或 Recall 证据的指标标 NA，不混用其他 checkpoint。

PASS — only frozen historical TIGER SID used; unmatched geometry/reconstruction/Recall NA; no TIGER retraining. E6/E8.

## C14

> 完成最终独立指标复算、全部资产持久化、代码提交与远端 SHA 核对；提交 VERDICT、性能报告、准入证据、Stage2/Stage3 完整记录及归档索引。

PASS for independent reconciliation and observed source/evidence commit/push/SHA equality at25a5164c246fa096e9703f3e3616c9c83a28ad94. Final verdict metadata publication/closure is anchored by the closing comment, not fabricated as an earlier event.

Commands and source/output SHA descriptors are in TRAINING_ADMISSION.json, PUBLIC_VERIFICATION.json, MODULE_CHECKS.json and RECONCILIATION.json. VERDICT.md resolves E1–E10 to exact artifacts and hashes. The final complete command logs are preserved, including the earlier failed reconciliation. No remaining numerical/implementation failure or requested formal training is deferred.
