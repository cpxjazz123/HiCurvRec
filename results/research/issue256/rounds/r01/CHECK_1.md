# R01 CHECK_1: historical source generation

Overall FAIL; no formal dispatch. Counts:{'PASS': 12, 'FAIL': 1, 'NOT_RUN': 4}. Recorded now from immutable executed receipts, not backdated. Issue body SHA256:32a2764070e8502926965d355a3ef6ff7619d45881abd0ffa575cb7cef73c5a8; rechecked unchanged against CHECK_3_ISSUE.json.

Failure: Real non-DDP machine entry rejected sampler=None

## C1 — NOT_RUN

> 这是持续迭代的研究 Issue，不是一次实验交付 Issue。继承 #255 的三曲率分层门控与累计截面曲率监督框架；#255 当前仍在执行，先等待其真实最终结果并审计，再作为本 Issue 的起点，不并行重复执行其模型。每轮完成后必须基于实际证据定位未达到目标的原因，提出针对性的下一轮修改，验证并迭代，直到同一冻结推荐协议下合法、完整 Stage3 最终 test R@10 > 0.065（严格沿用仓库采纳阈值）。没有达到目标不得标 COMPLETED 或以负结果结束本 Issue；真实权限/资源/外部依赖阻塞如实记录，等待解决，不伪造、不绕过权限。达到阈值后仍须完成完整验收、复算、归档和来源绑定才允许关闭。

Criterion/threshold:test_R@10 > 0.065; do not complete after a failed candidate. Data:ACCEPTANCE.json immutable identities. Scope:pretraining generation only; full training/final threshold remain NOT_RUN.

## C2 — PASS

> 保留三种曲率空间 {-1,0,+1}、共享有向 GNN、完整 1-hop / 2-hop ego、入/出方向拓扑 token、分层 Cross-Attention 几何门控、混合曲率码字选择及累计前缀截面曲率监督；GNN 与门控联合优化。

Criterion/threshold:three curvatures; directed shared GNN; four complete direction/ego tokens; layer gates; cumulative curvature. Data:ACCEPTANCE.json immutable identities. Scope:pretraining generation only; full training/final threshold remain NOT_RUN.

## C3 — PASS

> 保留768维原始语义输入、32维latent、三层256码字、统一商品SID、原始RQ-VAE重建/commitment/codebook、#255实际硬分配协议及合法选码梯度路径。只允许在现有机制内部做针对性改进，例如方向/子图readout、门控表达/梯度、几何距离尺度、截面代理的层次及归一化、梯度冲突和稳定性等；每次变更必须解释与基础机制的关系。不得引入独立行为邻域约束、InfoNCE、pair-ranking、额外行为预测任务或其他不相关Loss，不删2-hop、不冻结GNN、不替换三曲率空间。

Criterion/threshold:768/32; 3x256; original RQ-VAE losses; actual #255 hard assignment; no unrelated objective. Data:ACCEPTANCE.json immutable identities. Scope:pretraining generation only; full training/final threshold remain NOT_RUN.

## C4 — PASS

> 复用冻结训练图、完整曲率资产、基线TIGER与既有Infra；禁止重训TIGER、额外基线训练、PPR、四节点全量graphlet、多种子或盲目网格搜索。单agent串行研究决策，计算使用四卡，不创建每轮独立Issue。

Criterion/threshold:no repeated static assets, baseline training, PPR, grid or multiseed. Data:ACCEPTANCE.json immutable identities. Scope:pretraining generation only; full training/final threshold remain NOT_RUN.

## C5 — PASS

> 每个有明确机制假设的新候选一次正式训练：四卡/global4096/seed42/Stage2 3000 epochs，固定Stage3 150 epochs及beam20、完整test和归档；失败实现故障的修复重跑需明确区分。既有规则中“一次候选训练”仍适用，但本 Issue 明确授权在证据驱动下创建后续机制版本，不能将其误读为禁止继续迭代。

Criterion/threshold:one complete 3000-epoch world4/global4096 seed42 run per distinct hypothesis; frozen 150-epoch Stage3. Data:ACCEPTANCE.json immutable identities. Scope:pretraining generation only; full training/final threshold remain NOT_RUN.

## C6 — PASS

> 审计 #252–#255 的最终报告、真实源码、数据、checkpoint、SID与R2归档，核查#255是否完整通过。保留历史 #252 R@10 0.049287、#253 0.046432 及#254相关性结果，不把它们视作单因素因果或已核验的TIGER Recall。

Criterion/threshold:audit #252 through #255; never interpret correlation as cause. Data:ACCEPTANCE.json immutable identities. Scope:pretraining generation only; full training/final threshold remain NOT_RUN.

## C7 — PASS

> 建立固定、训练外的验证划分和统一诊断口径，冻结排序/候选解码/词表派生规则、评估总体及指标定义。优先使用 validation 与已冻结的结构验证库决定机制修改、权重或候选；严格隔离 recommendation test 和 structure test，禁止将#254具体测试错误商品清单、test标签/派生特征作为监督或参数选择依据。

Criterion/threshold:fixed train-external validation; test and #254 item-error lists excluded from supervision/selection. Data:ACCEPTANCE.json immutable identities. Scope:pretraining generation only; full training/final threshold remain NOT_RUN.

## C8 — PASS

> 用户要求以最终固定test R@10>0.065为关闭阈值。若多个候选先后查看同一test分数，该分数是反复使用的探索性闭环门槛，不再是完全独立的最终泛化估计。记录每次test访问及由此产生的选择偏差；条件允许时预留未访问的独立确认集或新数据集作一次确认，否则明确局限，不宣称无偏泛化。即使某轮test失败，下一轮具体修改应由训练/验证侧证据驱动，而非针对test样本调参。

Criterion/threshold:log each access; no claim of independent confirmation without genuinely untouched data. Data:ACCEPTANCE.json immutable identities. Scope:pretraining generation only; full training/final threshold remain NOT_RUN.

## C9 — PASS

> R1 记录上轮结果及未达标差距：R@5/10、NDCG@5/10、训练/验证行为边L1前缀保持、度匹配对照、L1/L2/L3前缀和碰撞、普通语义重建、各级硬q曲率宏/微误差与符号平衡准确率、正/零/负召回、分层gate/attention、梯度范数/方向、各卡耗时。历史缺失标NA，不拼接不一致checkpoint。逐用户/商品配对错误分析以validation为主要决策集，test仅在授权的最终测试和事后透明报告中使用。

Criterion/threshold:record all requested diagnostics; missing historical values NA. Data:ACCEPTANCE.json immutable identities. Scope:pretraining generation only; full training/final threshold remain NOT_RUN.

## C10 — PASS

> R2 定位最有证据的失败原因：先排除实现、数据隔离、导出、词表、训练/评估协议、尺度/梯度路径、数值和工程错误，再分析机制。区分已验证故障、统计相关、未证实假设。至少列出支持证据、反例、替代解释、覆盖率与影响量；若不能确定根因，必须明确未知，并设计能排除假设的最小诊断，而非虚构原因。

Criterion/threshold:support, counterexample, alternatives, coverage and measured impact; unproven root cause stays unknown. Data:ACCEPTANCE.json immutable identities. Scope:pretraining generation only; full training/final threshold remain NOT_RUN.

## C11 — PASS

> R3 冻结下一轮单一主要改动与可证伪预测：写清为何针对R2证据、数学定义、影响模块、保持不变项、训练前可验证判据、期望机制指标与Recall方向、失败判据。避免一次改变多个独立因素；若必须联合改动，明确只能归因于整个组合。不能仅凭上轮Recall低机械地更换参数。每轮方案/判据先登记，再写代码；权重/温度等原则上由训练侧统计或数学推导确定。

Criterion/threshold:one main mathematical change; falsifiable pretraining and final predictions. Data:ACCEPTANCE.json immutable identities. Scope:pretraining generation only; full training/final threshold remain NOT_RUN.

## C12 — FAIL

> R4 先红后绿实现并独立验收：真实数学参考、数值稳定、硬SID与checkpoint重导出一致、曲率Loss经选码路径更新各层gate/Q/K/V/GNN、梯度冲突、数据/资产身份、四卡DDP与真实入口；实际machine entry和behavior_graph PASS后才允许正式训练。诊断短运行不冒充正式结果。

Criterion/threshold:TDD; independent reference; mutations; module VJPs/updates; four-rank and real entry; entry and graph PASS. Data:ACCEPTANCE.json immutable identities. Scope:pretraining generation only; full training/final threshold remain NOT_RUN.

## C13 — PASS

> R5 保持性能与数据完整：先复用已保存资产，缺失静态数据才创建并持久化到assets/注册manifest；test派生分析留在evaluation区。真实四卡inclusive连续至少30稳态epoch计时，包含周期工作、结构Loss、通信和优化器；目标0.4–0.5s，既有授权上限0.8s。不得通过隐藏耗时、减少数据、改变训练预算或冻结GNN达标。超过上限定位并修复，不能机械派发。

Criterion/threshold:at least 30 consecutive steady four-card epochs; inclusive mean <= 0.8 s; target 0.4-0.5 s; full data and GNN. Data:ACCEPTANCE.json immutable identities. Scope:pretraining generation only; full training/final threshold remain NOT_RUN.

## C14 — NOT_RUN

> R6 完成当前候选Stage2 3000 epochs、合法硬SID（前三码碰撞<0.4且精确导出）及Stage3 150 epochs，按同一协议记录最终test，独立复算并绑定R2源码/SID/checkpoint/日志。保存验证结果、分层指标、逐用户配对、训练损失/梯度/时间等。无额外多种子或同机制追分重跑。

Criterion/threshold:3000/150 epochs; exact checkpoint export; first-three collision < 0.4; independent recompute and complete R2. Data:ACCEPTANCE.json immutable identities. Scope:pretraining generation only; full training/final threshold remain NOT_RUN.

## C15 — NOT_RUN

> R7 逐项验收与决策：对照预注册的预测，解释是否提高了真实L1前缀保持、几何保真、语义保真与推荐指标；若机制指标改善但Recall未改善，则该轮改善假设不成立。若 R@10 未过0.065，记录本轮最重要失败原因或剩余待证假设、下一轮方向和依据，更新本Issue进度后返回R1/R2继续。不得因单轮失败、负结果、复杂度或工作量以COMPLETED关闭。实际外部阻塞保留证据并等待解除。

Criterion/threshold:if final R@10 <= 0.065 return R1/R2; no same-mechanism score rerun. Data:ACCEPTANCE.json immutable identities. Scope:pretraining generation only; full training/final threshold remain NOT_RUN.

## C16 — PASS

> 建立单一持续迭代台账（round_id、父版本/源码SHA、变更、假设、诊断/验证数据身份、完整Stage2/3 run、结果、归档ID、结论、下一轮方向），每轮以评论汇报且持久化独立目录，不覆盖历史失败/负结果。保存所有可复用静态数据及manifest，保存每轮完整模型、SID、日志、分析表、指标、独立复算、准入证据及R2归档。正式候选源码与运行绑定，推送并核对远端SHA。每轮结果不删减，不把相关性称因果。

Criterion/threshold:all rounds, source hashes, evidence, comments, runs, failures and archive IDs preserved. Data:ACCEPTANCE.json immutable identities. Scope:pretraining generation only; full training/final threshold remain NOT_RUN.

## C17 — NOT_RUN

> 实际一次合法且完整的 Stage3 test R@10 > 0.065；源、输入、SID、checkpoint、3000/150 epochs、评估协议、独立复算和R2归档全部匹配；基础机制仍然保留；全部必要测试与真实准入通过；所有历史轮次与未达标原因、修复依据可追溯；如反复使用固定test，最终报告明确选择偏差及是否有独立确认。全部满足后才以COMPLETED关闭。

Criterion/threshold:every requested condition and R@10 > 0.065; complete source/SID/checkpoint/protocol/R2 pairing. Data:ACCEPTANCE.json immutable identities. Scope:pretraining generation only; full training/final threshold remain NOT_RUN.

## Executed evidence

- `results/research/issue256/rounds/r01/verification_generations/pre_loader_fix/CHECK_1_ISSUE.json` SHA256 `73ba834edbca1569338aa6d2385f246e8c72f8fa419a3d8d85172d369f9d4204`
- `results/research/issue256/rounds/r01/verification_generations/pre_loader_fix/LINEAGE.json` SHA256 `0b6912d764494560487191a88a8b8fc309074484ee9d980d7954d0f3483cf670`
- `results/research/issue256/rounds/r01/verification_generations/pre_loader_fix/TEST_RESULTS.json` SHA256 `b5672b298c8b800014ea30137cfd5def6a364085ec548c7a0ab9019278286e53`
- `results/research/issue256/rounds/r01/verification_generations/pre_loader_fix/PARENT_DIAGNOSTICS.json` SHA256 `4f884d6f9bbfbc439d96a914a8de81fe279f172ac2e4ff9e550212a3d50f81b5`
- `results/research/issue256/rounds/r01/verification_generations/pre_loader_fix/VALIDATION.json` SHA256 `d99f4afb0ef1c21c84fc46ae0104cbe51a94d8c608c9c209429c07a6364c3228`
- `results/research/issue256/rounds/r01/verification_generations/pre_loader_fix/LAYER_MUTATIONS.json` SHA256 `9b96ede3451eb81913dbda5bc31a54b19b19468f391618db3a585b53c1fb8056`
- `results/research/issue256/rounds/r01/verification_generations/pre_loader_fix/MUTATIONS.json` SHA256 `4c414a0ba536c64328e79e5576d4b8b6b9b3ae17fc6bb2244d64e663456acd98`
- `results/research/issue256/rounds/r01/verification_generations/pre_loader_fix/CALIBRATION_RED_RECEIPT.json` SHA256 `2b7914962139500670c0a72aae3401e23073fd0b8ec7ceb0445031c0469b82c4`
- `results/research/issue256/rounds/r01/verification_generations/pre_loader_fix/PROFILE_SUMMARY.json` SHA256 `c8ee88e9f5a57725ef7e2b676726c80c9f085cb5c4f3071969e7a679653041c9`
- `results/research/issue256/rounds/r01/verification_generations/pre_loader_fix/BUDGET_MUTATIONS.json` SHA256 `86dee901074dc79386205ea604b81b8a82328ca23916fc8b455a91fc77833cd7`
- `results/research/issue256/rounds/r01/verification_generations/pre_loader_fix/PERFORMANCE.json` SHA256 `a8d20d191bdb50f5d7cfbf980e22c693930ce576bf8a02c4fbdfa037df92dbc7`
- `results/research/issue256/rounds/r01/verification_generations/pre_loader_fix/CALIBRATION_GREEN_RECEIPT.json` SHA256 `7dbde61a7f23560d1c39a61653ec65c422155699fa0de5229131fa64dd2e7b17`
- `results/research/issue256/rounds/r01/verification_generations/pre_loader_fix/PRESERVATION.json` SHA256 `2e7fc08eb6040c915768a3fedad8f0b6cd8cfea4099b34a5940add2f75e3cea6`
- `results/research/issue256/rounds/r01/verification_generations/pre_loader_fix/ADMISSION.json` SHA256 `b590eca203f4289ac0703b4555fbb1a1101d175eb3945c710690f2bfd310ada2`
- `results/research/issue256/rounds/r01/verification_generations/pre_loader_fix/MODULE_CHECKS.json` SHA256 `b27032ab0ee8026e9deb026f65149c44b43c17c42af291bae5e0814cd9b0b2c6`
