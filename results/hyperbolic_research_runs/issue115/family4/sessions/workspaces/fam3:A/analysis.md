# 冻结协议下的实测瓶颈分析

## 1. ER 与 Stage3 排序要同时移动，必须关注什么量？

**现有测量没有证明任何单一量的变化能因果性地同时改善 Hard-L1 ER 和 Stage3 排序。** 但证据指出应检查的是 SID 分区实际承载的条件信息及其尾部结构，而不是 codebook 使用率或几何评分本身。冻结控制的 4000 个样本中，3612 个不同的 (L1,L2) 单元平均仅有 **1.0969** 个 L3 code，且 **92.08%** 单元只有一个 L3 code；与此同时，各层已使用 **243/256、255/256、256/256** 个 code。codebook 近乎用满并未带来丰富的条件后缀信息（`issue112/residual_information_baseline.json`、`issue112/q7_measured_obstruction.md`）。

ER 是不充分的下游代理：R4-08 的 ER 为 **16.402627511591962**，超过 **16.38953942232631**，但 Recall@10 为 **0.06295374223088843**、NDCG@10 为 **0.03454767796170093**，均低于冻结门槛 **0.06406796775709883**、**0.03530651701940677**（`results/stage2_RQ-VAE/R4_SUMMARY.md`）。因此，待移动并须直接测量的结构量是 SID 对样本/推荐相关差异的**有效区分能力**（包括条件分支丰富度及 L0 分区尾部）；目前没有证据表明仅移动其中某项就必然令三项指标一起过线。成功仍定义为同一机制同时严格超过三项冻结门槛。

## 2. 六个已关闭家族各自留下了什么未变？

1. **产品测地 level-0 assignment，以 codeword 半径评分**：评分被 codeword 半径支配，实测所有 **24,587** 个 item 落入同一 code，ER 为 **1.0**。它没有留下有效的 L0 分区差异；固定表示/codebook 的几何评分没有改变训练所得表示和残差尺度（`issue112/refutation_hyp_product_l1_quantiser.md`、`issue112/dag/state.json`）。
2. **Horospheric / Busemann level sets**：实测 level gap 没有分离宿主标签，方向相反；留下未变的是对宿主标签的有效区分（相关测量见 `issue112/cgv_hyp_busemann_boundary.json`、`issue112/join_e_hyp_busemann_boundary.json`）。
3. **Codeword-frame transported residual**：可良定义形式是恒等变换，assignment 改变量为 **0.0**；因此没有改变 assignment 或其承载的信息（`issue112/pt_frame_probe.json`、`issue112/pt_frame_probe_note.md`）。
4. **Role-assigned token layout**：审计为 **VESTIGIAL**，声明的几何没有进入 forward；实际前向计算未发生所声明的几何变化（`issue112/role_separation_check.json`、`issue112/layout_implementation_seam.md`）。
5. **Gyro-composition / Möbius associator**：在定义审查阶段因类型错误被拒两次；没有可执行的有效算子，故没有相应的前向 assignment 量被测为改变（`issue112` 中该家族的定义/审查记录）。
6. **Unsigned tangent-space wedge area**：14 轮讨论后仍未获一致批准，候选被搁置；现有记录没有证明它改变了真实 assignment 流或 SID 条件结构，不能把“未被数学反驳”写成效果证据（`issue115/joinr_nonconvergence.md`）。**LOT​​V** 是另一个被作者撤回的 tie-break 提案：其作用依赖近似平局，但没有先测 assignment stream 中达到预注册阈值的平局频率；因此没有证据证明它会改变常见 assignment。它不是 wedge 候选的测量结果，也不能据此复活 tie-break（`issue115` 的 LOTV 撤回/诊断记录）。

跨家族的结构性实测背景是：各 codebook 已基本用满；欧氏链中 L3 单元内 spread / 最近邻间距为 **0.7895**，改用 Möbius 位移后为 **1.4251**，但单 L3 code 单元比例仍由 **0.9208** 变为 **0.9263**。即评分几何可改变距离比，却没有让训练 codebook 的条件冗余下降（`issue112/q7_measured_obstruction.md`）。

## 3. 一个可证伪的“ER 更好、下游更差”因果解释

**假设：** ER 改善可能来自 L0 配对/分区尾部变化；若这种变化使 L0 最大 cell 变大、让更多 item 共用相同前缀，则 Hard-L1 ER 可以上升，但更深层 SID 可用于区分候选的有效后缀信息可能下降，导致 Stage3 Recall/NDCG 不升反降。这是待检验解释，不是已证实因果结论。

**测量依据：** 历史记录中 **14 个过 ER 门槛的臂没有一个改善下游**。控制 L0 最大 cell 为 **237**；两个记录到的较差尾部案例分别增至 **374**、**382**。控制使用 **245/256** 个 L0 code，Gini 为 **0.2162**，L1→L2 分支数为 **65.91**（`issue112/reading_sheet_after_the_run.md`）。R4-08 也显示 ER 过线并不保证 Stage3 过线，见上文指标。

**区分测量：** 对同一候选的 Stage2 导出 SID，同时记录 L0 最大 cell、使用 code 数、Gini、L1→L2 分支数及 (L1,L2) 条件下 L3 code 数/单码单元比例；再按冻结流程测同一候选的 ER、Recall@10、NDCG@10，并与控制并列。若 ER 提升而 L0 尾部不增、条件后缀信息不减，且 Stage3 仍恶化，则“L0 尾部变粗导致下游恶化”的解释被证伪；若 L0 尾部变粗并伴随后缀区分减少、Stage3 不升或下降，则与假设相符，但仍需同一候选的配对对照才能支持因果判断。所有测量须来自冻结数据、划分及配方；最终成功仍要求三项指标同时严格过门槛。
