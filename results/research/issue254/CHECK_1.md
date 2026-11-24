# Issue254 final acceptance round1

Reopened issue body SHA `ce03103cf5d97ff95a8b62cfe321cbbec5f4a5fd9bcbff8bee7a92c92e844dac` is unchanged. No training required or performed.9 PASS /0 FAIL /0 NOT_RUN. Module-level unavailable/forbidden operations are explicitly NA, not counted as fake passes.

## C1 — PASS

1. 冻结证据：读取 #252/#253 最终 REPORT/VERDICT、不可变正式 Stage2/Stage3 run、R2 100/101、原始测试预测/目标秩及实际输入序列，核验 test parquet、用户/样本 ID、目标 item ID、候选目录、SID/checkpoint、训练配置、源码与资产 SHA。两次 Stage2 同源 checkpoint 导出 SID 为权威，不用 trainer 原表或其他诊断快照。检查两次 Stage3 测试总体（57439）、相同 split/beam/cutoffs、排序及候选解码规则，严格一对一 join，拒绝仅按行号配对或把未命中当缺失。无法取得原始预测候选序列时明确缺口；若同源 Stage3 checkpoint 与推理条件可用，允许只读确定性评估并逐样本验证复现原秩，禁止重训、修改参数或把不匹配推理表当历史真值。

Observed: 57,439 samples/users; exact catalogue/shard/target/SID reconstruction; both R2 generation bindings and source hashes match. Original missing user/history-tensor dumps explicitly qualified.

Evidence: `results/research/issue254/EVIDENCE.json` SHA256 `2ce08437b558f911c27f4d475e98706e95a49833f6137419382b334a8ff7de04`. Final production/analysis source inventory: `MANIFEST.json` SHA256 `67b068bfc10aae7daaa593e91c7408d0828159e835d09209dd1a88ad3e7e2ada`.

## C2 — PASS

2. 全量配对错误归因：逐用户生成 test ledger，含目标商品、历史商品序列索引、两臂 target rank、hit@5/10、NDCG贡献及变化、预测 top-K 商品与 SID（若存在）。分别统计四组：两者命中、仅#252命中（退化）、仅#253命中（改善）、两者未命中；对@5/10分别验证净变化=(改善数−退化数)/N，独立复算 R/NDCG 与报告一致。保存全量逐样本表，逐商品聚合出现次数、退化/改善计数、命中率、净 Recall 差贡献（明确分母）、rank变化；区分同一商品多个用户的不同上下文。列出贡献最大的退化/改善商品及代表案例，不只挑错误案例。

Observed: Full ledger and16,462 target aggregates; net@10=-164/57439, net@5=-78/57439; all four original metrics and per-item contributions reconcile.

Evidence: `results/research/issue254/PAIRED_SUMMARY.json` SHA256 `61747bd011d9edf484aeac62f017647890d53b777224e57e2d0c492f2f591d19`. Final production/analysis source inventory: `MANIFEST.json` SHA256 `67b068bfc10aae7daaa593e91c7408d0828159e835d09209dd1a88ad3e7e2ada`.

## C3 — PASS

3. 实际错误预测：对退化组分析#253 top-K 错推商品、#252 的正确目标原排序、两臂共同候选/新增候选、预测 SID 前缀与目标 SID 的共享层级、最近正确分支、合法解码/碰撞/第四计数器、目标排名从top10跌到何处（超出beam记截尾而非无穷具体秩）。区分prefix选择失败、同前缀内部排序失败、商品SID冲突/解码失败、候选覆盖缺失；只有实际生成轨迹/概率可支持时才细分，不能由最终rank猜测T5内部原因。提供目标—错推商品—最近历史商品的真实图关系例子。

Observed: 40,716 wrong-candidate rows and12 balanced cases; selected decode legal; observed prefix coverage distinguished from missing beam probabilities/trajectories.

Evidence: `results/research/issue254/CANDIDATE_SUMMARY.json` SHA256 `d1c222b0802cb2925a7c53025d7dd5625f5ee5fe39d027ffd311defec6c8153e`. Final production/analysis source inventory: `MANIFEST.json` SHA256 `67b068bfc10aae7daaa593e91c7408d0828159e835d09209dd1a88ad3e7e2ada`.

## C4 — PASS

4. 关联结构：使用冻结训练可见有向图、1/2-hop ego、全量三元组截面曲率、Graphlet/Orbit既有资产、节点度/互惠/局部密度及真实转移关系，按商品 ID 严格连接。计算每个目标（并在可得时最近历史/错推商品）的入出度、邻域重叠、结构类型分布、截面曲率正零负比例/均值/分位数/方差及数量，必要时区分链/分叉/汇合/三角等；未覆盖角色/无合法三元组保留mask，不将4节点抽样称全量。检查退化组是否集中于某些拓扑、负/正/混合曲率、长尾或热门商品，以及历史→目标的真实行为关联；测试集关系仅用于事后诊断，绝不进入训练资产或用于调参。

Observed: 24,587 catalogue nodes,28,391,948 original triples, complete1/2-hop and57,439 contexts;261 no-triple masks; sampled roles explicitly marked.

Evidence: `results/research/issue254/STRUCTURE_SUMMARY.json` SHA256 `bee2094138db206478d249ceb71de4481919127c9e4cf1d6ceae967f710c4614`. Final production/analysis source inventory: `MANIFEST.json` SHA256 `67b068bfc10aae7daaa593e91c7408d0828159e835d09209dd1a88ad3e7e2ada`.

## C5 — PASS

5. 关联机制：读取或由相同 checkpoint 只读重算两臂逐商品 gate/attention、硬q/连续z、三层SID和逐商品曲率保真。分别记录#252五专家与#253三专家原始权重，另按负/零/正曲率合并以形成可比口径；分析欧氏权重增量、注意力对1/2-hop的分配、门控熵、曲率保真误差变化（用共同固定评价度量，不混用不同尺度），重建误差、量化误差及码本利用。SID跨模型整数码字没有相同语义，不直接比较码号是否相等，而比较各自前缀共享、同前缀邻居、目标和真实行为邻居的关系、碰撞及群组变化。将与真实拓扑相同/不同的组比较，检查#253欧氏权重88.4%与#252约70.27%的总体差是否也出现在受损商品上，注意不能以平均权重解释单商品因果。

Observed: Original five/three-expert CPU gate/attention, q/z/reconstruction and common_fixed3 item errors; exact SID-selected hard q. Candidate/history/target joins and folded-sign entropy in CONNECTIONS_SUMMARY.

Evidence: `results/research/issue254/MECHANISM_SUMMARY.json` SHA256 `4157ca61f2da9c1c7ec753b1d8d1c7f6c8a4f1252f5bd65df63799ee9e4dc282`. Final production/analysis source inventory: `MANIFEST.json` SHA256 `67b068bfc10aae7daaa593e91c7408d0828159e835d09209dd1a88ad3e7e2ada`.

## C6 — PASS

6. 严格统计：以配对命中变化为主终点，报告 McNemar 精确检验、配对 bootstrap 置信区间（固定统计种子，不是模型多种子训练）；按目标频率、行为度数、历史长度、基线 rank、曲率分布分层，报告样本数、分母、效应量、不确定性、多重比较校正；必要时商品聚类 bootstrap/按用户配对，控制热门商品及长序列混杂。区分用户加权与商品宏平均。对两者都错、改善组、两者都对作参照；不能只看退化组。报告条件关联而非因果；明确两臂同时改变专家数、固定监督度量、性能实现及实际训练轨迹，不能归因于单一因素。报告未经独立检验的假设和反例。

Observed: Exact McNemar;10,000 paired/target-cluster/macro bootstrap;66 corrected subgroup tests; frequency/degree/history adjusted item-cluster associations.

Evidence: `results/research/issue254/STATISTICS_SUMMARY.json` SHA256 `c0390539aa6e7a79d1a4850cd64d659f228cc364efe9eebd5f45acddfe0eed7f`. Final production/analysis source inventory: `MANIFEST.json` SHA256 `67b068bfc10aae7daaa593e91c7408d0828159e835d09209dd1a88ad3e7e2ada`.

## C7 — PASS

7. 结论可操作：给出退化的主要贡献商品/结构类型（以净 Recall 贡献及覆盖率排序）、哪些现象支持/反驳门控偏欧氏、曲率保真不足、SID前缀关系变化、语义重建下降、候选排序等假设，分别标为已验证/观察关联/未识别。给出下一轮最有证据的一个方法修改建议及其可证伪实验，不启动实验、不创建额外训练臂。若证据不足，诚实说明不能确定原因。

Observed: Hypotheses separated into verified facts/observational associations/unidentified causes; contrary gate evidence, macro uncertainty and one falsifiable unlaunched proposal.

Evidence: `results/research/issue254/REPORT.md` SHA256 `054bb3ed1b4fbe2a5f6051fd452509dedc1ddf9ea3964484ecc53d94f2c18d44`. Final production/analysis source inventory: `MANIFEST.json` SHA256 `67b068bfc10aae7daaa593e91c7408d0828159e835d09209dd1a88ad3e7e2ada`.

## C8 — PASS

8. 持久化全部分析数据：共享研究存储保存逐用户配对表、逐商品汇总、候选错误表、图/曲率/gate/attention/SID连接表、分层报告、可复算脚本、manifest（schema、行数、路径、输入/输出SHA、来源、用途=evaluation）、日志、REPORT/VERDICT及可读案例。复用 assets/ 已有静态训练图数据，不重复构建/覆盖。事后 test 衍生资产单独放 evaluation/analysis 区，禁止被 TrainingView 读取。仅提交仓库允许的小型摘要、脚本和索引，不把大二进制或个人用户原始信息公开。保存完整结果并重新加载验证。

Observed: 16 full analysis tables/arrays persisted outside TrainingView;256 input/dependency hashes and every table reload verified; private raw data ignored; code, logs, REPORT and CASES published.

Evidence: `results/research/issue254/MANIFEST.json` SHA256 `67b068bfc10aae7daaa593e91c7408d0828159e835d09209dd1a88ad3e7e2ada`. Final production/analysis source inventory: `MANIFEST.json` SHA256 `67b068bfc10aae7daaa593e91c7408d0828159e835d09209dd1a88ad3e7e2ada`.

## C9 — PASS

9. 验收：对手算配对样例、重复ID、缺失样本、rank截尾、同目标多用户、SID跨模型码号、mask、符号定义、分母、净贡献与全集重算添加断言；报告逐项 PASS/FAIL/NA 和证据路径。独立复核最终配对 Recall 差与逐商品贡献之和一致。推送代码与证据后核对远端 SHA。

Observed: 37 tests,0 failures/errors/skips;6 designated mutations. Independent final Recall and item contribution sum verified. Source/report commit927fb3c2780dc3685f00cc9b457685db0933a44d pushed and local/remote main SHA observed equal in DELIVERY_1.log.

Evidence: `results/research/issue254/TEST_RESULTS.json` SHA256 `df387f2506506c06d5a875b388e640b38d5040b9aa0f40261e3b87b432c439c3`. Final production/analysis source inventory: `MANIFEST.json` SHA256 `67b068bfc10aae7daaa593e91c7408d0828159e835d09209dd1a88ad3e7e2ada`.

## Commands and exits

Each actual analysis CLI command is recorded in MANIFEST.json; EVIDENCE_RUN, PAIRED_RUN, STRUCTURE_RUN, MECHANISM_RUN, CANDIDATE_RUN, STATISTICS_RUN, CONNECTIONS_RUN and VERIFY_RUN record successful final executions (exit0). Failed earlier schema/mutation harness generations are separately named and never reused as PASS. Final GREEN.log records37 tests with0 failures/errors/skips.

The four displayed REPORT metric rows were compared to the independent raw-prediction reduction at absolute tolerance1e-12. Metadata publication and the closing comment bind the final delivery SHA, without a self-referential hash inside this file.
