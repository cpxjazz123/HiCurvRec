# 冻结证据中的共同瓶颈与可证伪解释

## 1. ER 与 Stage3 排序必须共同移动的量

**现有测量没有证明任何一个单一量的改变能让 Hard-L1 ER 与 Stage3 排序同时改善。** 因此不能仅凭 ER 或某个结构指标宣称已有共同因果量。

测到的关键 SID 结构量是条件分配的信息丰富度，尤其是给定 `(L1,L2)` 后第三 token 的分支丰富度：在 4000 个样本、3612 个不同 `(L1,L2)` 单元中，平均每个单元只有 **1.0969** 个 L3 code，**92.0819%** 的单元只有一个 L3 code；与此同时各层已分别使用 **243/256、255/256、256/256** 个 code。故“codebook 使用率”不是缺口，若要改变 SID 的深层区分能力，候选至少应改变这种条件信息结构，而不只是换距离度量。残差移除比例为 L0 **61.1345%**、L1 **26.0942%**、L2 **20.1593%**。来源：`issue112/residual_information_baseline.json`、`issue112/q7_measured_obstruction.md`。

但这只是已测的结构性瓶颈，**不是已证实能共同推动 ER 与 Stage3 的量**。冻结成功要求同一机制同时严格超过 ER **16.38953942232631**、Recall@10 **0.06406796775709883**、NDCG@10 **0.03530651701940677**；过 ER 门槛不能替代后两项。

## 2. 六个关闭家族各自未改变的内容

1. **Product-geodesic level-0、以 codeword radius 评分**：没有形成有效的 level-0 分区；最小半径 codeword 吞并全部 **24,587** 个项目，只有 **1** 个 L0 code 被使用，ER 为 **1.0**。它未改变出有信息量的 SID，且没有 Stage3 结果（ER 门槛失败）。来源：`issue112/refutation_hyp_product_l1_quantiser.md`。
2. **Horospheric / Busemann level sets**：实测 level gap 未把宿主标签分开；没有改变数据、冻结表示和训练后 codebook 的结构。
3. **Codeword-frame transported residual**：未改变实际 assignment；其良定义形式是共同坐标变换这一恒等结果，assignment 改变量为 **0**。
4. **Role-assigned token layout**：声明的几何没有进入 forward（VESTIGIAL），所以实际前向计算及其几何机制未改变。
5. **Gyro-composition / Möbius associator**：在定义层因类型错误被拒两次；没有一个可执行的 score 机制改变 assignment。
6. **Unsigned tangent-space wedge area**：停放而非 refuted；14 轮未取得一致批准，故没有获准实现或测得的机制效应。它未改变冻结实验结果，也不能作为当前候选提出。

相关测量还显示，仅换几何并未改变第三层冗余：within-cell spread / L3 最近邻间距从欧氏 **0.7895** 移至 Möbius **1.4251**，但单 L3 code 单元占比从 **0.9208** 到 **0.9263**，并未下降。来源：`issue112/q7_measured_obstruction.md`。

## 3. 一个可证伪的“ER 更好、下游更差”解释

**因果假说（不是既成结论）：** 某些机制通过增大 L0 分区尾部来改善 Hard-L1 ER，但更大的 L0 单元把不同项目合并到同一前缀，损害后续 SID 对候选的区分，因此 Stage3 排序不升反降。

**现有证据与判别测量：** 历史记录称有 **14** 个过 ER 门槛的臂，没有一个改善下游；尾部案例的 L0 最大单元由控制的 **237** 增至 **374**（Case A）或 **382**（Case E）。对同一候选的导出 SID，预先记录 L0 最大单元、L0 使用 code 数、Gini，以及深层分支/条件 L3 丰富度；再对完全相同的候选运行冻结 ER 与 Stage3，报告三项未舍入结果。若 ER 提升时 L0 最大单元增大，并伴随 Recall@10 或 NDCG@10 未提升，结果符合该解释；若 ER 提升而 L0 尾部不增、且两项 Stage3 指标均提升，则该“尾部变粗导致下游恶化”的解释被证伪。控制参照：L0 最大单元 **237**、使用 **245/256** 个 code、Gini **0.2162**、L1→L2 分支 **65.91**。来源：`issue112/reading_sheet_after_the_run.md` 及 `issue112/q7_measured_obstruction.md`。

此解释不能把相关性冒充因果；判别关键是同一候选、同一导出 SID 上同时测分区统计与冻结 ER/Stage3 结果。
