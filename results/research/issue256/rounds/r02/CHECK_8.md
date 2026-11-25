# R02 CHECK_8 — FAIL, repair and recheck

8PASS/1FAIL/8NOT_RUN. Issue OPEN; original body SHA unchanged. No formal R02 training.

## C1 — NOT_RUN

这是持续迭代的研究 Issue，不是一次实验交付 Issue。继承 #255 的三曲率分层门控与累计截面曲率监督框架；#255 当前仍在执行，先等待其真实最终结果并审计，再作为本 Issue 的起点，不并行重复执行其模型。每轮完成后必须基于实际证据定位未达到目标的原因，提出针对性的下一轮修改，验证并迭代，直到同一冻结推荐协议下合法、完整 Stage3 最终 test R@10 > 0.065（严格沿用仓库采纳阈值）。没有达到目标不得标 COMPLETED 或以负结果结束本 Issue；真实权限/资源/外部依赖阻塞如实记录，等待解决，不伪造、不绕过权限。达到阈值后仍须完成完整验收、复算、归档和来源绑定才允许关闭。

Observed: Current-source complete execution or final run/push/strict final result is not yet established; no historical proof is substituted.

## C2 — NOT_RUN

保留三种曲率空间 {-1,0,+1}、共享有向 GNN、完整 1-hop / 2-hop ego、入/出方向拓扑 token、分层 Cross-Attention 几何门控、混合曲率码字选择及累计前缀截面曲率监督；GNN 与门控联合优化。

Observed: Current-source complete execution or final run/push/strict final result is not yet established; no historical proof is substituted.

## C3 — PASS

保留768维原始语义输入、32维latent、三层256码字、统一商品SID、原始RQ-VAE重建/commitment/codebook、#255实际硬分配协议及合法选码梯度路径。只允许在现有机制内部做针对性改进，例如方向/子图readout、门控表达/梯度、几何距离尺度、截面代理的层次及归一化、梯度冲突和稳定性等；每次变更必须解释与基础机制的关系。不得引入独立行为邻域约束、InfoNCE、pair-ranking、额外行为预测任务或其他不相关Loss，不删2-hop、不冻结GNN、不替换三曲率空间。

Observed: Frozen inherited capacity/protocol, immutable asset/provenance/validation/test-access ledger, diagnosis/contract or continuing-loop action is preserved in the bound evidence.

## C4 — PASS

复用冻结训练图、完整曲率资产、基线TIGER与既有Infra；禁止重训TIGER、额外基线训练、PPR、四节点全量graphlet、多种子或盲目网格搜索。单agent串行研究决策，计算使用四卡，不创建每轮独立Issue。

Observed: Frozen inherited capacity/protocol, immutable asset/provenance/validation/test-access ledger, diagnosis/contract or continuing-loop action is preserved in the bound evidence.

## C5 — NOT_RUN

每个有明确机制假设的新候选一次正式训练：四卡/global4096/seed42/Stage2 3000 epochs，固定Stage3 150 epochs及beam20、完整test和归档；失败实现故障的修复重跑需明确区分。既有规则中“一次候选训练”仍适用，但本 Issue 明确授权在证据驱动下创建后续机制版本，不能将其误读为禁止继续迭代。

Observed: Current-source complete execution or final run/push/strict final result is not yet established; no historical proof is substituted.

## C6 — PASS

审计 #252–#255 的最终报告、真实源码、数据、checkpoint、SID与R2归档，核查#255是否完整通过。保留历史 #252 R@10 0.049287、#253 0.046432 及#254相关性结果，不把它们视作单因素因果或已核验的TIGER Recall。

Observed: Frozen inherited capacity/protocol, immutable asset/provenance/validation/test-access ledger, diagnosis/contract or continuing-loop action is preserved in the bound evidence.

## C7 — PASS

建立固定、训练外的验证划分和统一诊断口径，冻结排序/候选解码/词表派生规则、评估总体及指标定义。优先使用 validation 与已冻结的结构验证库决定机制修改、权重或候选；严格隔离 recommendation test 和 structure test，禁止将#254具体测试错误商品清单、test标签/派生特征作为监督或参数选择依据。

Observed: Frozen inherited capacity/protocol, immutable asset/provenance/validation/test-access ledger, diagnosis/contract or continuing-loop action is preserved in the bound evidence.

## C8 — PASS

用户要求以最终固定test R@10>0.065为关闭阈值。若多个候选先后查看同一test分数，该分数是反复使用的探索性闭环门槛，不再是完全独立的最终泛化估计。记录每次test访问及由此产生的选择偏差；条件允许时预留未访问的独立确认集或新数据集作一次确认，否则明确局限，不宣称无偏泛化。即使某轮test失败，下一轮具体修改应由训练/验证侧证据驱动，而非针对test样本调参。

Observed: Frozen inherited capacity/protocol, immutable asset/provenance/validation/test-access ledger, diagnosis/contract or continuing-loop action is preserved in the bound evidence.

## C9 — PASS

R1 记录上轮结果及未达标差距：R@5/10、NDCG@5/10、训练/验证行为边L1前缀保持、度匹配对照、L1/L2/L3前缀和碰撞、普通语义重建、各级硬q曲率宏/微误差与符号平衡准确率、正/零/负召回、分层gate/attention、梯度范数/方向、各卡耗时。历史缺失标NA，不拼接不一致checkpoint。逐用户/商品配对错误分析以validation为主要决策集，test仅在授权的最终测试和事后透明报告中使用。

Observed: Frozen inherited capacity/protocol, immutable asset/provenance/validation/test-access ledger, diagnosis/contract or continuing-loop action is preserved in the bound evidence.

## C10 — NOT_RUN

R2 定位最有证据的失败原因：先排除实现、数据隔离、导出、词表、训练/评估协议、尺度/梯度路径、数值和工程错误，再分析机制。区分已验证故障、统计相关、未证实假设。至少列出支持证据、反例、替代解释、覆盖率与影响量；若不能确定根因，必须明确未知，并设计能排除假设的最小诊断，而非虚构原因。

Observed: Current-source complete execution or final run/push/strict final result is not yet established; no historical proof is substituted.

## C11 — PASS

R3 冻结下一轮单一主要改动与可证伪预测：写清为何针对R2证据、数学定义、影响模块、保持不变项、训练前可验证判据、期望机制指标与Recall方向、失败判据。避免一次改变多个独立因素；若必须联合改动，明确只能归因于整个组合。不能仅凭上轮Recall低机械地更换参数。每轮方案/判据先登记，再写代码；权重/温度等原则上由训练侧统计或数学推导确定。

Observed: Frozen inherited capacity/protocol, immutable asset/provenance/validation/test-access ledger, diagnosis/contract or continuing-loop action is preserved in the bound evidence.

## C12 — FAIL

R4 先红后绿实现并独立验收：真实数学参考、数值稳定、硬SID与checkpoint重导出一致、曲率Loss经选码路径更新各层gate/Q/K/V/GNN、梯度冲突、数据/资产身份、四卡DDP与真实入口；实际machine entry和behavior_graph PASS后才允许正式训练。诊断短运行不冒充正式结果。

Observed: FAIL in collection only: real formal-entry ownership regression1PASS, complete145/64/18 units and all four-rank diagnostics/machine checks complete. Profiler subset matching falsely also admits an obsolete inventory, yielding2matches. Three semantic regressions catch removed-candidate/missing-or-changed-controller cases; exact duplicate ambiguity remains rejected. Canonical full-role identity and exact before/after equality are implemented and bg_516 is regenerating all changed-source evidence. No reservation or formal training.

## C13 — NOT_RUN

R5 保持性能与数据完整：先复用已保存资产，缺失静态数据才创建并持久化到assets/注册manifest；test派生分析留在evaluation区。真实四卡inclusive连续至少30稳态epoch计时，包含周期工作、结构Loss、通信和优化器；目标0.4–0.5s，既有授权上限0.8s。不得通过隐藏耗时、减少数据、改变训练预算或冻结GNN达标。超过上限定位并修复，不能机械派发。

Observed: Current-source complete execution or final run/push/strict final result is not yet established; no historical proof is substituted.

## C14 — NOT_RUN

R6 完成当前候选Stage2 3000 epochs、合法硬SID（前三码碰撞<0.4且精确导出）及Stage3 150 epochs，按同一协议记录最终test，独立复算并绑定R2源码/SID/checkpoint/日志。保存验证结果、分层指标、逐用户配对、训练损失/梯度/时间等。无额外多种子或同机制追分重跑。

Observed: Current-source complete execution or final run/push/strict final result is not yet established; no historical proof is substituted.

## C15 — PASS

R7 逐项验收与决策：对照预注册的预测，解释是否提高了真实L1前缀保持、几何保真、语义保真与推荐指标；若机制指标改善但Recall未改善，则该轮改善假设不成立。若 R@10 未过0.065，记录本轮最重要失败原因或剩余待证假设、下一轮方向和依据，更新本Issue进度后返回R1/R2继续。不得因单轮失败、负结果、复杂度或工作量以COMPLETED关闭。实际外部阻塞保留证据并等待解除。

Observed: Frozen inherited capacity/protocol, immutable asset/provenance/validation/test-access ledger, diagnosis/contract or continuing-loop action is preserved in the bound evidence.

## C16 — NOT_RUN

建立单一持续迭代台账（round_id、父版本/源码SHA、变更、假设、诊断/验证数据身份、完整Stage2/3 run、结果、归档ID、结论、下一轮方向），每轮以评论汇报且持久化独立目录，不覆盖历史失败/负结果。保存所有可复用静态数据及manifest，保存每轮完整模型、SID、日志、分析表、指标、独立复算、准入证据及R2归档。正式候选源码与运行绑定，推送并核对远端SHA。每轮结果不删减，不把相关性称因果。

Observed: Current-source complete execution or final run/push/strict final result is not yet established; no historical proof is substituted.

## C17 — NOT_RUN

实际一次合法且完整的 Stage3 test R@10 > 0.065；源、输入、SID、checkpoint、3000/150 epochs、评估协议、独立复算和R2归档全部匹配；基础机制仍然保留；全部必要测试与真实准入通过；所有历史轮次与未达标原因、修复依据可追溯；如反复使用固定test，最终报告明确选择偏差及是否有独立确认。全部满足后才以COMPLETED关闭。

Observed: Current-source complete execution or final run/push/strict final result is not yet established; no historical proof is substituted.

