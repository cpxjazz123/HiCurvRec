# Issue #256 — continuous research ledger

Title: Iterative improvement: preserve three-curvature GNN-attention RQ-VAE until Stage3 Recall@10 exceeds 0.065

Body SHA-256: `32a2764070e8502926965d355a3ef6ff7619d45881abd0ffa575cb7cef73c5a8`. Snapshot: `ISSUE.json`. Parent #255 CLOSED (NOT_PLANNED/BLOCKED) at 2026-11-24T17:20:17Z; evidence commit `0d1b61101dbbf41374ac33571df959e88cfd00bf`, closing receipt commit `c4ee00cdc4314b7fcab882acd939bf18ee3b0824`.

## Training decision and authority

Training is required by 总目标与关闭条件 / R6 / 最终验收. Existing artifacts cannot satisfy a legal full Stage3 R@10 > 0.065. Each preregistered distinct hypothesis receives at most one frozen 3000-epoch Stage2 and 150-epoch Stage3; implementation-failure reruns remain separately labelled. Issue256 explicitly authorizes evidence-driven later candidates. A failed round is not completion. No gate waiver, unrelated loss, blind parameter grid, multiseed or extra baseline training. Research decisions are serial and single-agent.

Assets were inspected before creation; existing 109-entry manifest is reused, not rebuilt. No new static asset has been created. Archive inactive #255 before any new mechanism design or reference/production implementation. Immutable historical runs and failed evidence will not be rewritten.

## Frozen acceptance checklist

### C1 — Final legal Stage3 R@10 and continuing loop

> 这是持续迭代的研究 Issue，不是一次实验交付 Issue。继承 #255 的三曲率分层门控与累计截面曲率监督框架；#255 当前仍在执行，先等待其真实最终结果并审计，再作为本 Issue 的起点，不并行重复执行其模型。每轮完成后必须基于实际证据定位未达到目标的原因，提出针对性的下一轮修改，验证并迭代，直到同一冻结推荐协议下合法、完整 Stage3 最终 test R@10 > 0.065（严格沿用仓库采纳阈值）。没有达到目标不得标 COMPLETED 或以负结果结束本 Issue；真实权限/资源/外部依赖阻塞如实记录，等待解决，不伪造、不绕过权限。达到阈值后仍须完成完整验收、复算、归档和来源绑定才允许关闭。

判据/阈值: test_R@10 > 0.065; do not complete after a failed candidate.

数据: identities in ACCEPTANCE.json; frozen structure-validation entries additionally audited in INFRA.json.

口径: Each immutable rounds/<round_id>/CHECK_*.json and final VERDICT.md; production source hashes and command/exit code required; final test only after legal complete training.

### C2 — Architecture and trainability

> 保留三种曲率空间 {-1,0,+1}、共享有向 GNN、完整 1-hop / 2-hop ego、入/出方向拓扑 token、分层 Cross-Attention 几何门控、混合曲率码字选择及累计前缀截面曲率监督；GNN 与门控联合优化。

判据/阈值: three curvatures; directed shared GNN; four complete direction/ego tokens; layer gates; cumulative curvature.

数据: identities in ACCEPTANCE.json; frozen structure-validation entries additionally audited in INFRA.json.

口径: Each immutable rounds/<round_id>/CHECK_*.json and final VERDICT.md; production source hashes and command/exit code required; training/validation-only research decisions.

### C3 — Capacity, assignment and allowed scope

> 保留768维原始语义输入、32维latent、三层256码字、统一商品SID、原始RQ-VAE重建/commitment/codebook、#255实际硬分配协议及合法选码梯度路径。只允许在现有机制内部做针对性改进，例如方向/子图readout、门控表达/梯度、几何距离尺度、截面代理的层次及归一化、梯度冲突和稳定性等；每次变更必须解释与基础机制的关系。不得引入独立行为邻域约束、InfoNCE、pair-ranking、额外行为预测任务或其他不相关Loss，不删2-hop、不冻结GNN、不替换三曲率空间。

判据/阈值: 768/32; 3x256; original RQ-VAE losses; actual #255 hard assignment; no unrelated objective.

数据: identities in ACCEPTANCE.json; frozen structure-validation entries additionally audited in INFRA.json.

口径: Each immutable rounds/<round_id>/CHECK_*.json and final VERDICT.md; production source hashes and command/exit code required; training/validation-only research decisions.

### C4 — Reuse and serial decisions

> 复用冻结训练图、完整曲率资产、基线TIGER与既有Infra；禁止重训TIGER、额外基线训练、PPR、四节点全量graphlet、多种子或盲目网格搜索。单agent串行研究决策，计算使用四卡，不创建每轮独立Issue。

判据/阈值: no repeated static assets, baseline training, PPR, grid or multiseed.

数据: identities in ACCEPTANCE.json; frozen structure-validation entries additionally audited in INFRA.json.

口径: Each immutable rounds/<round_id>/CHECK_*.json and final VERDICT.md; production source hashes and command/exit code required; training/validation-only research decisions.

### C5 — Candidate run authority

> 每个有明确机制假设的新候选一次正式训练：四卡/global4096/seed42/Stage2 3000 epochs，固定Stage3 150 epochs及beam20、完整test和归档；失败实现故障的修复重跑需明确区分。既有规则中“一次候选训练”仍适用，但本 Issue 明确授权在证据驱动下创建后续机制版本，不能将其误读为禁止继续迭代。

判据/阈值: one complete 3000-epoch world4/global4096 seed42 run per distinct hypothesis; frozen 150-epoch Stage3.

数据: identities in ACCEPTANCE.json; frozen structure-validation entries additionally audited in INFRA.json.

口径: Each immutable rounds/<round_id>/CHECK_*.json and final VERDICT.md; production source hashes and command/exit code required; training/validation-only research decisions.

### C6 — Parent provenance audit

> 审计 #252–#255 的最终报告、真实源码、数据、checkpoint、SID与R2归档，核查#255是否完整通过。保留历史 #252 R@10 0.049287、#253 0.046432 及#254相关性结果，不把它们视作单因素因果或已核验的TIGER Recall。

判据/阈值: audit #252 through #255; never interpret correlation as cause.

数据: identities in ACCEPTANCE.json; frozen structure-validation entries additionally audited in INFRA.json.

口径: Each immutable rounds/<round_id>/CHECK_*.json and final VERDICT.md; production source hashes and command/exit code required; training/validation-only research decisions.

### C7 — Validation isolation

> 建立固定、训练外的验证划分和统一诊断口径，冻结排序/候选解码/词表派生规则、评估总体及指标定义。优先使用 validation 与已冻结的结构验证库决定机制修改、权重或候选；严格隔离 recommendation test 和 structure test，禁止将#254具体测试错误商品清单、test标签/派生特征作为监督或参数选择依据。

判据/阈值: fixed train-external validation; test and #254 item-error lists excluded from supervision/selection.

数据: identities in ACCEPTANCE.json; frozen structure-validation entries additionally audited in INFRA.json.

口径: Each immutable rounds/<round_id>/CHECK_*.json and final VERDICT.md; production source hashes and command/exit code required; training/validation-only research decisions.

### C8 — Test access and bias

> 用户要求以最终固定test R@10>0.065为关闭阈值。若多个候选先后查看同一test分数，该分数是反复使用的探索性闭环门槛，不再是完全独立的最终泛化估计。记录每次test访问及由此产生的选择偏差；条件允许时预留未访问的独立确认集或新数据集作一次确认，否则明确局限，不宣称无偏泛化。即使某轮test失败，下一轮具体修改应由训练/验证侧证据驱动，而非针对test样本调参。

判据/阈值: log each access; no claim of independent confirmation without genuinely untouched data.

数据: identities in ACCEPTANCE.json; frozen structure-validation entries additionally audited in INFRA.json.

口径: Each immutable rounds/<round_id>/CHECK_*.json and final VERDICT.md; production source hashes and command/exit code required; final test only after legal complete training.

### C9 — R1 complete diagnostic inventory

> R1 记录上轮结果及未达标差距：R@5/10、NDCG@5/10、训练/验证行为边L1前缀保持、度匹配对照、L1/L2/L3前缀和碰撞、普通语义重建、各级硬q曲率宏/微误差与符号平衡准确率、正/零/负召回、分层gate/attention、梯度范数/方向、各卡耗时。历史缺失标NA，不拼接不一致checkpoint。逐用户/商品配对错误分析以validation为主要决策集，test仅在授权的最终测试和事后透明报告中使用。

判据/阈值: record all requested diagnostics; missing historical values NA.

数据: identities in ACCEPTANCE.json; frozen structure-validation entries additionally audited in INFRA.json.

口径: Each immutable rounds/<round_id>/CHECK_*.json and final VERDICT.md; production source hashes and command/exit code required; training/validation-only research decisions.

### C10 — R2 discriminating diagnosis

> R2 定位最有证据的失败原因：先排除实现、数据隔离、导出、词表、训练/评估协议、尺度/梯度路径、数值和工程错误，再分析机制。区分已验证故障、统计相关、未证实假设。至少列出支持证据、反例、替代解释、覆盖率与影响量；若不能确定根因，必须明确未知，并设计能排除假设的最小诊断，而非虚构原因。

判据/阈值: support, counterexample, alternatives, coverage and measured impact; unproven root cause stays unknown.

数据: identities in ACCEPTANCE.json; frozen structure-validation entries additionally audited in INFRA.json.

口径: Each immutable rounds/<round_id>/CHECK_*.json and final VERDICT.md; production source hashes and command/exit code required; training/validation-only research decisions.

### C11 — R3 preregistered single change

> R3 冻结下一轮单一主要改动与可证伪预测：写清为何针对R2证据、数学定义、影响模块、保持不变项、训练前可验证判据、期望机制指标与Recall方向、失败判据。避免一次改变多个独立因素；若必须联合改动，明确只能归因于整个组合。不能仅凭上轮Recall低机械地更换参数。每轮方案/判据先登记，再写代码；权重/温度等原则上由训练侧统计或数学推导确定。

判据/阈值: one main mathematical change; falsifiable pretraining and final predictions.

数据: identities in ACCEPTANCE.json; frozen structure-validation entries additionally audited in INFRA.json.

口径: Each immutable rounds/<round_id>/CHECK_*.json and final VERDICT.md; production source hashes and command/exit code required; training/validation-only research decisions.

### C12 — R4 actual path verification

> R4 先红后绿实现并独立验收：真实数学参考、数值稳定、硬SID与checkpoint重导出一致、曲率Loss经选码路径更新各层gate/Q/K/V/GNN、梯度冲突、数据/资产身份、四卡DDP与真实入口；实际machine entry和behavior_graph PASS后才允许正式训练。诊断短运行不冒充正式结果。

判据/阈值: TDD; independent reference; mutations; module VJPs/updates; four-rank and real entry; entry and graph PASS.

数据: identities in ACCEPTANCE.json; frozen structure-validation entries additionally audited in INFRA.json.

口径: Each immutable rounds/<round_id>/CHECK_*.json and final VERDICT.md; production source hashes and command/exit code required; training/validation-only research decisions.

### C13 — R5 full-work performance

> R5 保持性能与数据完整：先复用已保存资产，缺失静态数据才创建并持久化到assets/注册manifest；test派生分析留在evaluation区。真实四卡inclusive连续至少30稳态epoch计时，包含周期工作、结构Loss、通信和优化器；目标0.4–0.5s，既有授权上限0.8s。不得通过隐藏耗时、减少数据、改变训练预算或冻结GNN达标。超过上限定位并修复，不能机械派发。

判据/阈值: at least 30 consecutive steady four-card epochs; inclusive mean <= 0.8 s; target 0.4-0.5 s; full data and GNN.

数据: identities in ACCEPTANCE.json; frozen structure-validation entries additionally audited in INFRA.json.

口径: Each immutable rounds/<round_id>/CHECK_*.json and final VERDICT.md; production source hashes and command/exit code required; training/validation-only research decisions.

### C14 — R6 complete legally paired run

> R6 完成当前候选Stage2 3000 epochs、合法硬SID（前三码碰撞<0.4且精确导出）及Stage3 150 epochs，按同一协议记录最终test，独立复算并绑定R2源码/SID/checkpoint/日志。保存验证结果、分层指标、逐用户配对、训练损失/梯度/时间等。无额外多种子或同机制追分重跑。

判据/阈值: 3000/150 epochs; exact checkpoint export; first-three collision < 0.4; independent recompute and complete R2.

数据: identities in ACCEPTANCE.json; frozen structure-validation entries additionally audited in INFRA.json.

口径: Each immutable rounds/<round_id>/CHECK_*.json and final VERDICT.md; production source hashes and command/exit code required; final test only after legal complete training.

### C15 — R7 continue negative rounds

> R7 逐项验收与决策：对照预注册的预测，解释是否提高了真实L1前缀保持、几何保真、语义保真与推荐指标；若机制指标改善但Recall未改善，则该轮改善假设不成立。若 R@10 未过0.065，记录本轮最重要失败原因或剩余待证假设、下一轮方向和依据，更新本Issue进度后返回R1/R2继续。不得因单轮失败、负结果、复杂度或工作量以COMPLETED关闭。实际外部阻塞保留证据并等待解除。

判据/阈值: if final R@10 <= 0.065 return R1/R2; no same-mechanism score rerun.

数据: identities in ACCEPTANCE.json; frozen structure-validation entries additionally audited in INFRA.json.

口径: Each immutable rounds/<round_id>/CHECK_*.json and final VERDICT.md; production source hashes and command/exit code required; final test only after legal complete training.

### C16 — Immutable continuing ledger

> 建立单一持续迭代台账（round_id、父版本/源码SHA、变更、假设、诊断/验证数据身份、完整Stage2/3 run、结果、归档ID、结论、下一轮方向），每轮以评论汇报且持久化独立目录，不覆盖历史失败/负结果。保存所有可复用静态数据及manifest，保存每轮完整模型、SID、日志、分析表、指标、独立复算、准入证据及R2归档。正式候选源码与运行绑定，推送并核对远端SHA。每轮结果不删减，不把相关性称因果。

判据/阈值: all rounds, source hashes, evidence, comments, runs, failures and archive IDs preserved.

数据: identities in ACCEPTANCE.json; frozen structure-validation entries additionally audited in INFRA.json.

口径: Each immutable rounds/<round_id>/CHECK_*.json and final VERDICT.md; production source hashes and command/exit code required; training/validation-only research decisions.

### C17 — Final conjunction

> 实际一次合法且完整的 Stage3 test R@10 > 0.065；源、输入、SID、checkpoint、3000/150 epochs、评估协议、独立复算和R2归档全部匹配；基础机制仍然保留；全部必要测试与真实准入通过；所有历史轮次与未达标原因、修复依据可追溯；如反复使用固定test，最终报告明确选择偏差及是否有独立确认。全部满足后才以COMPLETED关闭。

判据/阈值: every requested condition and R@10 > 0.065; complete source/SID/checkpoint/protocol/R2 pairing.

数据: identities in ACCEPTANCE.json; frozen structure-validation entries additionally audited in INFRA.json.

口径: Each immutable rounds/<round_id>/CHECK_*.json and final VERDICT.md; production source hashes and command/exit code required; final test only after legal complete training.

## Rounds

R01 is preregistered as `TIGER_module_budget_prefix_rq` / `issue256_r01_formal`; CONTRACT.md and DIAGNOSIS.md were published before reference/implementation. Its single change is the maximal initial global module-gradient norm budget. No Issue256 formal training has started.

The first complete source generation passed121 candidate+64 inherited tests and four-rank module/performance checks, but the machine entry exposed a sampler-None compatibility failure. Its exact source overrides/receipts remain in `rounds/r01/verification_generations/pre_loader_fix/`. Unified actual-DataLoader peeking repairs that path without changing mathematical or training definitions; dedicated RandomSampler/DistributedSampler/explicit-generator regressions passed and actual entry/behavior_graph both passed.

The next source generation passed124+64 tests, four-rank validation and inclusive mean0.7077311168s, but its real profiler wrapper hid loader generator/sampler attributes. `pre_profile_fix/` and raw failed profile run preserve that evidence; old root summaries are not substituted for a current credential. Real wrapper and source-drift regressions first failed; transparent wrapper delegation and profiler-dependency hash coverage are being verified before a new complete credential generation. See COMMENT_01..11 and ENTRY_REPAIR.md. Training/validation remain the only mechanism-decision inputs; no test item error list is used.

Current-source CHECK_3 completed:126 candidate+64 framework tests pass; machine entry/behavior_graph pass; four-rank actual module/prefix/update/export checks, full-data30-epoch inclusive mean0.7027374609683951s, and per-card instrumented epoch2 profiling pass. `dispatch_evidence.verify` rechecked93 source/contract files and all credentials.13 pretraining criteria PASS,0 FAIL,4 full-training/final-adoption criteria NOT_RUN. This permits one formal dispatch, not issue completion. No source change is planned before the immutable R01 freeze.

R01 admitted source/evidence commit `0cdfec9b024e568b5b08d96e777033844d66635e` was pushed and matched remote main. The sole formal run is now reserved and running from its immutable snapshot; actual workers435981/435982/435983/435984, torchrun434483. At18:56:03Z it had completed629/3000epochs, first real Trainer smoke passed, and no final checkpoint existed. Formal lambda was derived once as1.6054239040386165e-6; another diagnostic generation's coefficient is not substituted. Intermediate epoch600 collision0.2037662179 is not a final SID or Stage3 admission credential. Postprocessing code may be implemented separately while the immutable training source and inputs remain unchanged.

At19:11:01.284375Z the same four workers had completed2000/3000epochs. Actual loss0.0001084160030586645 and reconstruction0.00008517928654327989 were finite; epoch2000 periodic collision0.08040834587383572. No final checkpoint/train_end existed. This is progress only, not legal Stage3 admission. The new CPU validation-only consumer reproduced all15 parent validation-prefix rates exactly without opening a structure-test bank (`VALIDATION_READER_SMOKE.json`); final-weight consumers remain unexecuted until the actual final checkpoint exists.

The sole formal Stage2 launcher exited0 after2133.41s; its actual train_end records2084.053s and all3000epochs. Final collision0.06568511815186884, checkpoint680439b57bfdbec622a5a109b0d589ab21566f71236fc007eb19d59b6108fc0c, SID201d2cc537dfdcc27060302489324d818356fd6327b88de89584943108a5758f. Cold audit exited0 with all24587items and suffix exact, no SID rewriting; layer metrics preceded Stage3 access. Quality allows entry with quant_error0.0072272333547981436/effective_rank5.390244278887607; the low-error machine warning is a diagnostic, not proof of generalization failure.

Final-weight actual-loss replay recorded18/18 combined gradients and fresh-state diagnostic optimizer updates. Its local1024-item rank0 final-weight maximum weighted C/O ratio10.45835596927704(Query0) does not contradict the initial-only global budget; it exposes no guaranteed later bound and is not a global final-DDP measurement. The checkpoint was never modified by this probe. Full catalogue extraction independently exited0 with bitwise hard-prefix and exact SID identity.

Stage3 MCP accepted the legal same-SID run without collision waiver, pid809893/dispatch ddp, fixed150epochs/seed42/beam20/no_eval/skip_testfalse. At19:32:18Z it remained in initialization without train_start; do not count dispatch as completion. Full three-level training-graph curvature and the five fixed structure-validation banks are computing on CPU only. Postprocessing sources and the new validation-only adapter were created after the immutable training snapshot; they are not retroactively included in its63-source training identity.

Full CPU evaluation completed in102.302s; independent SID/reconstruction/validation/per-centre reduction exited0 and all four preregistered Stage2 forecasts passed. Distinct prefixes256/8559/22972; catalogue MSE0.00007783928001416676; validation direct-minus-control0.037791790922898125 with control0.009530292716133424. Geometry is mixed: all three macro MAEs improved versus parent255, but all micro MAEs and balanced sign accuracies worsened; each cell retains28391948triples/24326centres and exactly77zero targets. Mean Euclidean gates0.95622/0.97680/0.98551 do not imply uniform use of the three retained spaces. These are training/validation measurements, not a Recall result or single-factor causal attribution. COMMENT_21 and STAGE2_RECONCILIATION.json bind every number.

Stage3 actual Trainer layout matched ddp4/global16384/infer1024; four real CUDA workers858572/858573/858574/858575 each used about28370MiB. At19:48:11Z it reached58/150epochs with finite loss2.010012083053589 and no error markers; the single exact run directory is `Nov-25-2026_06-36-15`. The initial launcher delay was observed Lustre synchronous IO reading a torch FSDP pyc, not a reason to restart. No final test exists yet.

Pure algebra of the retained final local-gradient record gives original-dot-combined normalized values-2.3951106672(sharedKey) and-6.9843923213(Query0). `POST_GRADIENT_DIRECTION_SUMMARY.json` states the exact equation and single-rank/optimizer/causality limits; no extra GPU probe, new formal run or mid-run coefficient modification was performed.

At19:56:17Z Stage3 reached100/150epochs, loss1.916323070526123, with the same four workers and no errors. A pre-use external-validation reducer regression exposed acceptance of4/5Parquet rows; the exact assertion failed before the fix, while the full5-row/all-miss case passed. Both inference and reduction now require the immutable validation file's metadata population; the unchanged two-test file passes2/2. The old consumer source is preserved byte-exact in pre_population_fix/, and red/green receipts include output, exit codes and hashes. This is an evaluator repair, not a training-source or protocol change. EVALUATION_CONSUMERS.json freezes the final validation/persisted-test consumers before any R01 test_final exists; their actual final-model end-to-end execution remains NOT_RUN.

R01 completed the sole150-epoch Stage3 run with57439 test rows: R@5=0.034105746966346905, R@10=0.05273420498267727, NDCG@5=0.022554059459244956, NDCG@10=0.02852344807984429. Independent reduction of persisted predictions reproduced all four metrics; no second test inference or checkpoint selection was performed. The strict0.065 target is missed by0.01226579501732273. R01 is COMPLETED_NOT_ADOPTED, not completion of Issue256.

The fixed full validation population57439 independently yielded R@10=0.057643761207541916 using the same Stage3 checkpoint d8540c837dedffd8291615bb6d7578308f74f6469e3182887685a08571082a96. Of54128 validation misses,41397 had no predicted L1 prefix matching the target. All gold targets were absent from the seen-history filter. This localizes the observable error, not its cause; no recommendation-test item error list enters the next mechanism decision.

The minimum final-weight four-rank diagnostic averaged actual VJP vectors before taking norms.13/17 coupled groups exceeded the initial-only weighted-curvature/original norm budget; maximum9.0859271222(Query0). Only global sharedScore had negative original-dot-combined, normalized -0.6988326877109153. Local Key/Query0 ascent did not survive rank averaging. These observations establish later norm drift, not an R01 contract violation, optimizer-step ascent, or the cause of the Recall gap.

The unchanged uploader exited0 and SHA-verified117 remote objects under r2:genrec/102/. The archive pairs the exact Stage2 source, SID201d2cc537dfdcc27060302489324d818356fd6327b88de89584943108a5758f, checkpoint680439b57bfdbec622a5a109b0d589ab21566f71236fc007eb19d59b6108fc0c and full Stage3 recall/log evidence. R2_RECEIPT.json records the complete owned ledger entry; no partial archive is reported as complete.

After every R01 worker and diagnostic had exited, git mv archived the entire live tree to stage2_RQ-VAE/_archive/TIGER_module_budget_prefix_rq. LIVE_ARCHIVE.json proves the old live path absent and the immutable executed run unchanged. The only forward tooling source constant now points to the archive; historical execution identities remain untouched. No new mechanism was designed or copied before this archive prerequisite. Issue256 remains OPEN; the next preregistered change must address train/validation evidence and receive fresh admission rather than rerun R01 for its score.

R02 preregistration a991a69b9afbbff9d7103597c186a723798a8e5b is pushed with equal main SHAs. After R01 archival, a15-file exact baseline-only source copy was made and ARM_NAME alone changed, before freezing CONTRACT.md. The sole intervention refreshes the unchanged global17-module norm budget at existing50-epoch native block anchors1,51,...,2951. No original loss, curvature proxy/scale, assignment, representation, data, optimizer, recipe or evaluation change is registered. Root cause of low Recall remains unknown; final global norm drift is the measured intervenable lead, with explicit local-direction counterexamples and alternatives.

Before implementation, actual four-rank/global4096 real-loss red diagnostics completed all60epochs. All4 workers1746128/1746129/1746130/1746131 passed independent initial17-module budget checks, but only initial calibration occurred; the exact assertion `actual60epoch trainer must refresh its budget at epoch51` failed on all4 ranks. Exit1 is intentional semantic red, not a crash/import failure. PERIODIC_RED_RECEIPT.json preserves all records and source generations.

The implementation adds only native-block cadence and current-first-step invocation of the same budget solver. Full event-ledger validation now binds60 formal anchors to the final checkpoint coefficient. An actual isolated-file controller regression separately exposed a doubled r02/r02 inventory path before its repair; omission failed the specified source-changed assertion. Candidate, inherited-framework and controller suites plus four-rank real module/prefix/update checks, epoch51 current-state/mutation proof and inclusive31–60 performance are running on a fixed source generation; no PASS or formal dispatch is claimed before their results.

R02 complete verification generation2 passed every collected unit exactly once under the standard launcher:139 candidate/64 inherited/15 controller cases, four nonempty shards and no skips. The real four-rank validation then failed the independent fresh-replica initial-budget maximality assertion on all4 ranks before backward; epoch51, inclusive performance and machine entry remain NOT_RUN. CHECK_2 records8PASS/1FAIL/8NOT_RUN and preserves all source-bound failures.

The old shared-path oracle assumed a globally optimal assignment objective, but the inherited floor is greedy. A hand-computable2x2 counterexample invalidates that premise. The repaired verifier checks each actual greedy native replay and conditionally compares continuous derivatives with the same hard branches; it does not claim native global optimality or unconditional forward equivalence. Its2-case regression and12 controller checks pass; actual four-rank continuous-path proof remains pending behind the earlier budget check. This is a verifier repair, not a tolerance or mechanism change.

The subsequent actual one-epoch four-rank point discriminator exited0 in348.69s. Calibration and native first-step parameters/scales/encoded/gates/costs/codes are bit-identical; fresh identical-weight replicas differ in GNN gate outputs by up to3.644824028015137e-5 while hard codes in this trace remain identical. Replica maximum global budget ratio1.0000317630586744 exceeds the frozen2e-6 tolerance. Independent NumPy reduction of all4 raw traces matches the producer records. Static mean cache construction versus compiled GNN execution remain unresolved alternatives; no production defect is asserted without a discriminating component measurement. No R02 formal run has started.

The full-cache component discriminator found different fresh CUDA incoming/outgoing immutable means on all4 cards despite identical normalized features and six composed affine tensors. The same consumer repeated exactly. A permanent production-cache regression then failed the designated incoming-equality assertion on all4 cards before implementation. CUDA cache construction now reuses the already verified complete-row deterministic kernel, without rebuilding assets or changing the mean definition; the CPU branch is unchanged. Targeted green and complete142/64/15 unit coverage passed, and all4 actual independent raw-replica17-module initial budgets passed with lambda1.0698641972339828e-6. STATIC_CACHE_RED_RECEIPT.json and STATIC_CACHE_GREEN_RECEIPT.json bind the old/new source and real outputs.

The later real shared-versus-independent geometry-gate check still failed4/4 after a complete actual epoch. CHECK_3 records8PASS/1FAIL/8NOT_RUN and unchanged issue text; epoch51, inclusive performance, machine entry and the sole formal R02 remain unexecuted. A different minimum diagnostic kept actual post-epoch parameters fixed and compared full GNN/attention stages under high versus highest arithmetic. High produced token differences up to0.00327283144 and gate differences0.000141650438; highest satisfied every observed unchanged forward bound, with max gate difference1.4901161193847656e-7. All4 parameter hashes stayed fixed. This is a demonstrated numerical-policy effect, not a Recall attribution.

The next production-path gate batch-invariance test is running red before any precision implementation. BATCH_PRECISION_REPAIR.md records the intended one-line full-FP32 runtime correction, unchanged mathematical contract/recipe/tolerances, and mandatory fresh inclusive-performance validation. R02's eventual result must disclose the numerical/cache engineering repairs as well as its periodic-budget main mechanism; it cannot establish single-factor causal isolation versus R01.

The production companion-batch gate regression caught the original high arithmetic at its specified assertion before the one-line highest repair; independent CPU NumPy reduction reproduced all112 saved end-to-end stage statistics. The first full-precision generation passed that new regression but exposed an existing identical-instance hard-output test failure. A separate actual installed-quantizer test proved dead integer-id rotating-history retention; weak owner keys repair that lifecycle defect without changing live rotation or feasible formal assignments. Stage2/genrec MCP children were reloaded and actual status/fail-closed probes completed. All intermediate source, outputs and CHECK_4 are preserved.

The next complete generation passed143/64/17 cases, including lifecycle/install/identical-instance/gate checks, but all4 real validators reached a later wrong native replay mode. A true2x2 VQLayer counterexample proves auxiliary balanced=False cannot be replayed using restored model.train() True. The verifier now explicitly consumes the captured per-population mode; no production allocation or tolerance changes. CHECK_5 preserves the failed generation.

Final current-source verification exited0 after593.64s:145 candidate/64 inherited/17 controller+framework cases with no skips, actual18-module gradients/optimizer updates, all three individual prefix routes, native replay and conditional shared-GNN VJPs. Real60epoch/world4 diagnostics verify anchors1/51 at steps0/300 and17 independent current-state budgets, with all4 stale-lambda mutations caught and restored. Inclusive31–60 mean0.7827437869894008s includes evaluation50/refresh51; the0.8s ceiling passes, desired0.4–0.5s is missed. Machine entry/behavior_graph pass, and dispatch verification exits0 binding140 files and12 actual rank receipts. No R02 formal run has started; source/evidence must be pushed before its sole dispatch.

Admitted source/evidence16259df8c01a55a8689f195bcd2c301fabe7a3bb was pushed with equal main SHAs. Its first formal entry call failed closed before reservation and launcher dispatch: an inherited R01 execution_entry.py collided with the framework-generated adapter. The formal workspace does not exist; no GPU worker or training budget was consumed. A source-complete freeze_run/load_run regression reproduced the same collision in0.16s after an excluded initial fixture-setup error. The obsolete live adapter is preserved outside the tree; no training computation changed. CHECK_7 records8PASS/1FAIL/8NOT_RUN and explicitly supersedes the old source-bound credentials. The new actual workspace regression and all full-source gates are running again as bg_515 before a first real formal worker can start.
