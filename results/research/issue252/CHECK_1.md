# CHECK_1 — open, no formal launch

Issue252 body re-read in CHECK_1_ISSUE.json; SHA3e9b8ca4018cd9d0a43648efc42cd6a12ef27bc892fdb981590030bea37849eb unchanged. This is an intermediate check, not final-source admission. PASS4 / FAIL2 / NOT_RUN8.

C7 failed actual inclusive mean2.2026225s; continue exact optimisation. C8 failed one missing-asset exception fixture in the full mathematical run; corrected fixture and new exactGNN/floor branches still require final-source whole-suite and actual module/DDP/entry proof. Other untouched clauses are NOT_RUN, never inferred PASS.

## C1: PASS

> 从原始训练交互独立重建/对账 canonical 图（323633有向边）与冻结 visible 图（261132有向边），逐类解释62501边的差异，核查用户划分、结构 held-out、推荐 valid/test 隔离及权重规则。禁止把隐藏行为用于训练。

Threshold: canonical323633/visible261132;62501 classified;unexplained0

- `assets/v7_graph_contract/audit.json:0918afe35cddf209441fff4fe80a558a2d562e14027326d124ec2f51000acb1b`
- `tools/issue252/graph_audit.py:22a0f91b2ba8282ad84d1c3c6a6ab32f0c52baacc5f00f5ea3b12248d80ddd67`

## C2: PASS

> 此 issue 明确授权修复该冲突：若 canonical 包含专门结构验证/测试所保留的训练用户行为，则以合法可见图为训练图，并修订规范/准入为可机器复验的具名图契约，严格证明来源和隔离；若冻结 visible 的差异源自错误构图，则按正确可见协议构建版本化新资产，重算受影响的 ego、特征、全量曲率和评估 bank。不可仅屏蔽检查或静默换图。保留旧资产及 provenance，正确资产不重复计算，新增数据持久保存至 assets/ 并更新 manifest。

Threshold: named legal graph/source/isolation exact;no duplicate correct assets

- `assets/v7_graph_contract/contract.json:d61a4ca54f6e7ea29331652aceae1b0d9495e215c44fb8415ae507ed52e38687`
- `results/research/issue252/REGISTERED_GRAPH_V9.json:2f6302719536618c3a9c270c6b3a0639f8f4375fc40f3c42222a77b4f269bec6`
- `tools/research/stage2/graph_contract.py:f2a6c9cd8d59f8412856a2d222ce4a5af5dce7e3fb80311825899a87d7972481`

## C3: NOT_RUN

> 实际 verification_check 及 stage2 正式准入必须 entry PASS + behavior_graph PASS；回归覆盖错图、混图、泄漏、旧缓存和源版本变化。

Threshold: entry PASS and behavior_graph PASS;adversarial regressions executed

No actual acceptance evidence yet; continue execution.

## C4: PASS

> 复核以前约0.4s/epoch的真实日志及计时边界/数据/硬件/global batch，区分单卡入口、四卡正式、校准、评估和导出；对当前相同配方进行真实四卡逐阶段 profiling：数据读取、ego 构建、GNN、Attention、五几何代价、Sinkhorn、全参考距离、三元组曲率、反传、DDP通信。报告每项耗时、峰值显存、调用频率与重复计算量。

Threshold: >=30 historical epochs and all actual four-rank phase costs identified

- `results/research/issue252/PROFILING.md:bed3da2c53d2d8776fffc8915f82e5b70d80dc3907eb04b190c4b9c27c33ddba`
- `results/research/issue252/BASELINE_PROFILE.json:575a96366375d95b9799d439c1f4ac218e12ebc05ff8abdaa658da7f8aece0ed`
- `results/research/issue252/HISTORICAL_TIMING.json:b1faf9bae689a4fee945f2e5d5c934785f4e46f044ba000ac4101babb4bfd541`

## C5: NOT_RUN

> 优先消除重复整图 ego 展开、重复磁盘解压、逐商品 Python 循环、重复全目录 forward 和重复参考距离；采用持久化静态图索引/稀疏张量、批处理、共享计算、精确代数化简、分块、融合算子与预取。不能把可学习 GNN 输出长期缓存为常量或冻结 GNN；门控与 GNN 必须有真实更新。

Threshold: no repeated avoidable expansion/decompression/catalogue/reference work;learned branches update

No actual acceptance evidence yet; continue execution.

## C6: PASS

> 全量监督资产继续保存，不重新抽样构建；训练计算可采用有明确数学依据的分块/随机小批量或周期性结构更新以达到可行预算，但必须在修改前固定监督估计器、覆盖策略、归一化、参考集合、梯度语义及与全量目标的差异，验证无偏性（若声称无偏）或明确偏差及其影响。禁止把 batch 内参考均值冒充全参考精确结果、未经声明改目标、长期跳过结构 Loss 或将其仅用于评估。完整全量曲率评估仍必须保留。若采用周期性更新，必须将其成本计入摊销 epoch 时间，并记录每 epoch 实际目标覆盖及更新次数；不能用不含结构计算的 epoch 计时冒充总速度。不得以追求速度为由取消端到端机制。

Threshold: estimator frozen before edits;probabilities/coverage/bias/gradient semantics executable

- `results/research/issue252/CONTRACT.md:4ca0c421b397b20e13c0ab970a7e69cdf9752b7a947fb4d8f5ee2986f843bb92`
- `results/research/issue252/reference.py:055a86a7fec662bc67878530e3c63ca873c63f773f89d2ace66761a1b16e704a`
- `stage2_RQ-VAE/TIGER_gnn_attention_sectional_rq/model/estimator.py:bc7633a3aed6f3060a0180d99cb50bf6035070ef9f096447eb5289389ce4c024`
- `results/research/issue252/ESTIMATOR_MUTATIONS.json:9571474356235b34951d2d7014f4c366a31fb49f01040a7508c5165a2d645a0b`

## C7: FAIL

> 在原冻结四卡配方（global batch4096、seed42、3000 epochs）上做短性能验证，排除初始化后报告连续至少30 epoch 的总墙钟/epoch（包括数据加载、周期性结构计算摊销、前后向、优化器、通信），以及中位数/P95、完整验证窗口耗时和预计3000 epoch成本。单独报告初始化及导出开销。目标平均 <=0.4 秒附近，验收上限0.5秒/epoch；若未达到，继续定位优化，不直接开始正式3000 epoch，不通过改计时方式、删数据、缩减 batch/epoch 或隐藏重计算达标。若证明目标与硬件/完整机制存在不可兼容的成本下界，报告实测及可行方案，明确未达标，不伪造结果。

Threshold: >=30 consecutive inclusive epochs;mean<=0.5s;report median/P95/init/export

- `results/research/issue252/benchmark/issue252_benchmark_4185524_577609457/rank_0.json:a24a85aab801e6c92a6202a97608d084e9c5eb60e3167c8177ea4f5e01cbfca5`

## C8: FAIL

> 每项优化先红后绿，独立对照数值、梯度和硬SID；近似算法必须单列，不称精确等价。复核 GNN/Q/K/V/score/Encoder 梯度、权重干预影响真实码字、硬q与导出一致、固定度量不旁路、DDP一致、重建语义和碰撞。禁止为了速度改变原始768/32维、三层SID、容量及原Sinkhorn硬分配协议。

Threshold: red-to-green independent numeric/gradient/hard-SID/DDP proofs;frozen recipe unchanged

- `results/research/issue252/CONTRACT_TEST_RUN.log:dbf2b3bfc79dc799a190cced1803c79e5f991375f65e580d2826a8d9986fa188`
- `results/research/issue252/STATIC_MEANS_TESTS.log:086c75a1ad48b77a7f936846c30b3ab15e11530ce6cb460b7e9ff079dbdc1dae`

## C9: NOT_RUN

> 全部准入、性能与数学测试通过后，使用标准 launcher/MCP 正式执行一次四卡 Stage2 3000 epoch（固定原配方，不做多种子、不训额外基线、不调参筛选）。记录 run_id、源哈希、资产哈希、完整损失/梯度/时间/显存、门控/注意力分布及真实完成状态。

Threshold: exactly3000 formal four-card epochs;global4096/seed42;actual completed run

No actual acceptance evidence yet; continue execution.

## C10: NOT_RUN

> 从本次 checkpoint 导出真实 SID，重新编码严格对齐；记录前三码碰撞率、占用、重建、硬q与连续z曲率宏/微MAE、分符号 balanced accuracy、held-out结构保真与前缀指标。Stage3 碰撞硬门槛 <0.4，不能用其他快照、第四列计数器或软表示代替。

Threshold: same-run checkpoint/SID exact;collision<0.4 for Stage3;full diagnostics

No actual acceptance evidence yet; continue execution.

## C11: NOT_RUN

> 如发生真实运行、OOM、数据、导出、碰撞或准入故障，诊断根因、以回归测试修复，并在授权范围内恢复必要运行；不得把故障恢复冒充多种子实验或以测试结果挑选模型。保留失败证据，不覆盖历史产物。

Threshold: all actual failures diagnosed, regression-tested and source-bound;no model-selection reruns

No actual acceptance evidence yet; continue execution.

## C12: NOT_RUN

> 使用该次合法 SID 实际完成一次 Stage3 150 epochs，遵守冻结 T5 配方，不能替换 SID、提前停止或跳过最终 test。保存 R@5/R@10、NDCG、实际日志、checkpoint、SID哈希与 run 绑定，并按现有仓库规定归档 R2。完整训练完成与研究收益是两件事：如未超过历史 TIGER 或仓库科学阈值，如实报告负结果，不为追指标重复训练。

Threshold: exactly150 formal Stage3 epochs/final test;exact R2 bundle verified

No actual acceptance evidence yet; continue execution.

## C13: NOT_RUN

> TIGER 只用已冻结历史产物对比，不重新训练 TIGER；无同源几何、重建或 Recall 证据的指标标 NA，不混用其他 checkpoint。

Threshold: only frozen historical TIGER;unbound metrics NA;no retrain

No actual acceptance evidence yet; continue execution.

## C14: NOT_RUN

> 完成最终独立指标复算、全部资产持久化、代码提交与远端 SHA 核对；提交 VERDICT、性能报告、准入证据、Stage2/Stage3 完整记录及归档索引。

Threshold: independent metrics/asset/source verification;commit/push/remote SHA equality

No actual acceptance evidence yet; continue execution.

No Stage2 formal run, Stage3 run or final SID/metric is claimed. ExactGNN/floor optimisation and corrected warmed profiler are ongoing; tests and diagnostics do not replace required training.
