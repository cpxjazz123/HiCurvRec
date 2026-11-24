# Issue255 CHECK_3 — corrected pre-dispatch checkpoint

Issue body SHA-256: `2fa23b1ee2faa897155f2a12999b3cf4cd7c085a5fa0a5f8a99c53f54b1c361d`. Actual reread: `CHECK_3_ISSUE.json`. Full original criteria follow verbatim.

Counts:8 PASS,0 FAIL,5 NOT_RUN. Post-training/full-evaluation criteria are not waived; this checkpoint only establishes permission for the approved corrected dispatch.

## C1 — PASS

从冻结 TIGER 基线及 #253 已归档机制核对实现，新建独立候选；保留768维原始语义输入、32维latent、三层256码字、#253 的实际分配协议（L1/L2 floor-constrained nearest，L3 Sinkhorn）、原始重建/commitment/codebook、四卡global4096、seed42、3000/150 epoch。唯一专家曲率 {-1,0,+1}，不恢复±3。GNN、门控端到端可训练，不冻结。

## C2 — PASS

方向感知 token：保留完整1-hop与2-hop诱导子图，新增中心+全部入邻居诱导子图、中心+全部出邻居诱导子图（保留选定节点间所有真实有向边），四种token分别经共享有向GNN/readout，专家注意力前不合并。明确空邻居、重复/互惠、root、mask、类型编码及同一商品全局身份；不将入/出子图称作互斥，不裁剪2-hop。仅从具名冻结训练可见图读取；不得用#254测试错误清单或测试关系生成特征。

## C3 — PASS

分层门控：复用一次GNN token计算；每层l有三种曲率专家的可训练Query或明确等价的层条件化Cross-Attention，得到w_i^(l)=softmax_k(score(H_i,l,k))，形状[N,3层,3专家]。注意力在token维归一、几何权重在专家维归一；不同层可以学习不同权重，不手工规定哪层使用哪种曲率。每层混合代价Σ_k w_i,l,k d_k²(r_i,l,e_l,j)，保留实际硬选码与同代价可微代理；不加入邻居距离项。

## C4 — PASS

累计量化前缀q_i^(l)=Σ_{j≤l}e_i,j，硬前向逐位对应实际SID前l个码字。对于同一合法三元组(m;b,c)与参考集合，逐层以固定、门控无关的三专家等权对称距离计算GraphMoRE A.1代理K_l=mean_a[d²(q_a^l,q_m^l)+d²(q_b^l,q_c^l)/4−(d²(q_a^l,q_b^l)+d²(q_a^l,q_c^l))/2]。沿用冻结图目标、#253 训练采样估计器和全量评估；三个层级共用同一抽样及参考集合，不为每层重复GNN/量化。

## C5 — PASS

用分层截面监督替换原仅最终q的结构项：L_total=L_original_RQVAE+λ Σ_l α_l Huber(K_l/target_scale,K_G/target_scale)。首版 α_l=1/3，λ通过实际初始encoder梯度校准一次冻结；逐层距离尺度须在训练前根据训练侧初始累计q确定、冻结并记录，以避免L1/L2因表示尺度不同而天然承担更大误差；若采用其他尺度规定，先给数学解释及固定测试，不按test/Recall挑选。明确这是累计前缀的几何代理，不是直接L1同码字约束，也不保证保留曲率必然提升L1共享。无独立附加行为Loss，无人为规定Graphlet对应码字。

## C6 — NOT_RUN

写清实际训练的有限采样/量化总体引入的偏差，不声称全量训练或无偏梯度；保留全28,391,948三元组与全参考的独立评价，固定统一跨模型评价尺度与度量。训练目标/评估口径区别清楚。

## C7 — PASS

读取现有 assets/manifest.json，复用具名图、全部曲率、完整ego CSR、静态特征、采样调度和 Triton 计算；入/出诱导子图确实缺失时才创建，必须全量精确、持久存放共享 assets/ 版本目录、更新 manifest（用途、shape、行数、hash、来源）并验证重载，保留旧资产。不重复计算/复制已有文件。

## C8 — PASS

复用 #253 同一步重建与辅助商品并集GNN编码、各自独立量化总体；不跨optimizer step缓存动态输出。对4-token额外计算进行真实四卡profiling；尽量共享固定邻域特征、稀疏索引及可证明等价的计算，保持完整2-hop。三层门控共享一次GNN，三层曲率共用一次硬量化前缀及抽样索引，复用距离计算公共中间量，避免重复全目录forward、CPU同步、重复索引、重复磁盘I/O。依据#253最终实测0.6032s/epoch，目标0.4–0.5秒，保持此前授权硬上限0.8秒；用至少30连续稳态四卡inclusive epoch验证，含所有结构计算、加载、通信、周期评估摊销，报告平均/median/P95、初始化与导出单列、逐模块/逐rank时间和显存。不得通过删2-hop、冻结GNN、减采样或隐瞒工作达标；超过上限先优化并复验。

## C9 — PASS

TDD红→绿，验证入/出egos手算、完整邻接/方向/互惠/空集、全部ID与覆盖、四token及层专家归一、不同层有效权重干预、三曲率距离及零极限、硬前缀与SID逐位一致、每个L_curv^(l)对相应前缀的梯度路径（不允许未来层参数影响早期前缀的直接计算），切断代价路径的负向测试、各层GNN/Q/K/V/score/encoder/码本梯度、独立数值参考、DDP等价、数据隔离与资产重载。保持真实入口和图机器准入PASS，正式运行前保存真实验证凭据。记录每层未加权/加权梯度范数、与重建梯度方向、裁剪及权重分布。不能以非零梯度、连续或软SID代替硬选码证据。

## C10 — NOT_RUN

仅在准入与性能通过后执行一次正式四卡Stage2（3000 epochs），保存真实checkpoint、SID与训练日志。重导出与实际最终checkpoint逐项一致；三码碰撞<0.4才进入Stage3。一次正式冻结Stage3（150 epochs，beam20，最终test），不重训TIGER、不新建对照训练、不做多种子、不追分调参。

## C11 — NOT_RUN

预先冻结主要机制指标：训练可见有向边的出邻居L1共享率（按边及按中心宏平均）、严格 held-out 结构组L1共享率及度匹配对照，L1/L2/L3前缀、三码碰撞、码本占用、重建误差；分层硬q及全量曲率宏/微MAE、符号平衡准确率、正零负召回、逐商品分布、各层三专家门控/attention。按同一固定三专家评价口径复算#253及本臂，不直接混用不同层或不同尺度的原生MAE。冻结TIGER SID作基线，不补训；无同源历史指标则NA。

## C12 — NOT_RUN

机制假设：L1真实边保持及独立结构评估改善，并且最终同一57,439测试总体的R@10高于#253 0.046431866850049616；报告与#252 0.0492870697609638的历史参考、冻结TIGER可溯源指标及仓库采纳阈值>0.065（阈值不是TIGER成绩）。记录R@5/10、NDCG@5/10、逐用户配对及净变化，避免仅凭碰撞低或训练曲率下降宣称推荐收益。若前缀改善但Recall未改善，否定本轮性能修复假设；若两者均下降/不变，如实负结果。由于同时修改方向token、分层门控与分层监督，结果仅支持组合机制，不宣称任何单模块因果贡献。

## C13 — NOT_RUN

保留全量结果、每商品及分层数据、模型/源码/资产/SID/checkpoint/run身份、性能/准入日志、报告/VERDICT和R2精确归档，并独立复算最终指标、哈希与重载。测试衍生分析数据只放evaluation区，不进入训练。提交并核对远端SHA。

## Executed commands

- Required Stage2 Python: `-B -m tools.issue255.launch_validation`, exit0.
- `python3 results/research/issue255/run_tests.py`, exit0; children use the project Stage2 interpreter.
- Required Stage2 Python: `-B -m tools.issue255.launch_steady_profile`, `launch_benchmark`, `admission`, `collect_evidence`, all exit0.

Exact source inventories and rank hashes are retained in the following receipts. No post-formal result exists for the corrected generation yet.

|Receipt|SHA-256|
|---|---|
|`TEST_RESULTS.json`|`eca5d9bf5b60651dcc1de9763dc5f6c96177b9f2b7e3ee2b9cd32e451e398a19`|
|`ADMISSION.json`|`d9393bcf430d03c2697ef111b2099d92b00bad9da0b4ff395597e1f233e070c1`|
|`VALIDATION.json`|`90286cb217bcb0212a300467b823903691400b9934c51834529615aaf7744b21`|
|`MODULE_CHECKS.json`|`d68b1d869ab888b17903f07ed80be055fa727a72a2ac0f00d6f8371cfe2495c2`|
|`PERFORMANCE.json`|`31059a38f2a1a183f675dd054aab7b25e8d668a837369e86e81df491402bf84d`|
|`PROFILE_SUMMARY.json`|`f61a90c66e037edd2bbe145ec462ba2a9fc88b583481e3940dba773ff118ac89`|
|`OWNER_APPROVAL.json`|`232e63be311610b5cd9053182269e571aeab4bee8da6ceed1187c60328c7367a`|
