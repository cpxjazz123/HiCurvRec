# 冻结协议下：共同提升所需移动的量（仅据实测）

## 1. ER 与 Stage3 排名必须共同改善，什么量必须动？

**仓库中的测量没有证明某一个量一旦移动，就能因果地同时提升 Hard-L1 ER 和 Stage3 排名。** 因此不能把任何结构统计宣称为已验证的共同充分原因。冻结的成功判据是同一候选同时严格超过 Hard-L1 ER **16.38953942232631**、Recall@10 **0.06406796775709883**、NDCG@10 **0.03530651701940677**。

若问现有证据指出的、必须检验是否发生实质变化的 SID 结构量，最明确的是**给定前缀后的条件区分信息**，尤其是第三 token 在给定 `(L1,L2)` 后的分支丰富度：4000 个项目、3612 个不同前缀单元中，每个单元平均只有 **1.0969** 个 L3 code，**92.08%** 单元只有一个 L3 code；但三层 codebook 已分别使用 **243/256、255/256、256/256** 个 code。因此“更多 code 被用到”不是证据所支持的瓶颈，必须测的是 SID 在条件上下文中实际区分了什么。该量是测得的结构性瓶颈，不是 ER 与排序共同改善的因果保证。

反例说明 ER 不能做代理：R4-08 的 ER **16.402627511591962** 已高于门槛，但 Recall@10 **0.06295374223088843**、NDCG@10 **0.03454767796170093** 均低于各自门槛。历史记录还称有 **14** 个过 ER 门槛的臂没有一个改善下游。故必须在同一候选上同时测 ER、两项 Stage3 指标及 SID 条件结构，不能从 ER 推断排序会变好。

## 2. 六个已关闭家族分别留下了什么没动？

以下按任务给定的六个家族归纳；“未动”指测量显示的机制边界，不把被拒/停放误写成实证反证。

1. **产品测地 level-0、按 codeword 半径评分**：没有得到有信息的 level-0 分区——全目录 **24,587** 个项目都分到同一码，Hard-L1 ER 为 **1.0**。改变几何评分没有改变出有效区分的 L0 SID；训练码本本身并未塌缩，是 assignment 塌缩。
2. **Horospheric / Busemann level sets**：实测 level gap 没有分开宿主标签（且方向相反）；没有建立可用的标签区分，也没有改变冻结数据、训练后表示或码本拟合结构。
3. **Codeword-frame transported residual**：其良定义形式是共同坐标变换，assignment 改变量为 **0.0**；没有改变残差携带的信息或码本结构。
4. **Role-assigned token layout**：所声明的几何没有进入 forward，审计结论为 **VESTIGIAL**；因此前向计算实际没有改变，token 角色声明没有改变模型所用的量。
5. **Gyro-composition / Möbius associator score**：在定义层面因类型错误被两次否决；没有可执行且定义成立的 score，故没有测得它改变 assignment、SID 或下游指标。
6. **Unsigned tangent-space wedge area**：14 轮未获一致认可而停放；它**未被 refute**，但也没有获批准为候选或实测证明有效。现有材料没有证据显示 wedge 项改变了 assignment 或其与行为标签的对齐；不得把未测写成“已证明无效”。

补充边界：LOT​​V 有向四面体体积（tie-break only）是作者撤回，不属于上述六项；撤回理由是没有 assignment-stream 证据证明预注册的每层 **≥5%** tie rate。它不能在未先测 tie 发生率时复活。

## 3. 一个可证伪的“ER 更好、下游更差”解释

**假说：L0 尾部变粗带来 ER 收益，却合并了下游排序需要区分的项目。** ER 可以随 L1 正负碰撞比例变化而升高，但如果改善主要来自更大的 L0 cell，后续 SID 可用的区分结构可能受损；这与共同因果尚未证实，必须用配对测量检验。

已有观测与该假说相容：历史队列中 14 个过 ER 门槛的臂均未改善 downstream；最差尾部案例的 L0 最大 cell 从控制值 **237** 增至 **374**（Case A）或 **382**（Case E）。控制 L0 使用 **245/256** 个 code、Gini **0.2162**，L1→L2 分支数为 **65.91**。R4-08 则展示了 ER 过线而两项 Stage3 未过线的背离。以上是关联性测量，不能单独确立因果。

**区分测量 / 证伪条件：** 对同一数据划分、同一候选导出的 SID，逐臂并列记录 L0 最大 cell、L0 code 使用数与 Gini、L1→L2 分支数、各 `(L1,L2)` 单元的 L3 分支数、ER 的正负 share、Hard-L1 ER、Recall@10 和 NDCG@10。若 ER 上升伴随 L0 最大 cell 增大、条件分支变差而 Stage3 不升或下降，结果符合假说；若在可比候选中尾部不增（相对控制 **237**）而 ER 与两项 Stage3 同时越过冻结阈值，或尾部增长并不对应下游变差，则“尾部合并导致背离”的解释受到反证。所有结论只以同一候选上的配对结果判断，且不能以某一结构量替代三项严格门槛。

## 证据位置

- `results/hyperbolic_research_runs/issue112/q7_measured_obstruction.md`
- `results/hyperbolic_research_runs/issue112/residual_information_baseline.json`
- `results/hyperbolic_research_runs/issue112/refutation_hyp_product_l1_quantiser.md`
- `results/hyperbolic_research_runs/issue112/reading_sheet_after_the_run.md`
- `results/hyperbolic_research_runs/issue115/fanout/sessions/workspaces/fan:A:analysis/analysis.md`（R4-08 的配对指标）
