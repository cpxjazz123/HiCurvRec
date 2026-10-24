# 冻结协议下：哪些量尚未被移动

## 1. ER 与 Stage3 排序必须一起看的量

**现有测量不支持“存在一个已经证实、单独移动即可同时提高 ER 与 Stage3 排序的量”。** 可由记录确定的是：不能把 Hard-L1 ER 当作排序效果的代理；必须让同一臂的 ER、Recall@10、NDCG@10 同时过冻结阈值。R4-08 的 ER 为 **16.402627511591962**，超过 **16.38953942232631**，但 Recall@10 为 **0.06295374223088843**、NDCG@10 为 **0.03454767796170093**，仍低于各自阈值 **0.06406796775709883** 和 **0.03530651701940677**（`results/stage2_RQ-VAE/R4_SUMMARY.md`）。

若问下一步必须测量并尝试移动的结构量，记录指向 **SID 分区所保留的条件区分信息**，而不是 codebook 使用率或单纯的距离/度量变化。Issue112 的 4000 个样本中有 **3612** 个不同的 (L1,L2) 单元；每单元平均仅 **1.0969** 个 L3 code，**92.08%** 的单元只有一个 L3 code。与此同时三层分别已使用 **243/256、255/256、256/256** 个 code，故“用满 codebook”不是缺口。测量指出，若要让第三个 SID token 真正增加信息，需改变它所编码的因子/条件分配；但当前证据**没有证明**这种变化会同时提升 ER 与 Stage3 排序。来源：`results/hyperbolic_research_runs/issue112/q7_measured_obstruction.md`。

另一条直接由历史队列支持的诊断量是 **L0 分区尾部**：14 个过 ER 门槛的臂没有一个改善下游；若尾部变粗，不能把 ER 上升误当作共同改善。控制的 L0 最大 cell 为 **237**，使用 **245/256** 个 code，Gini 为 **0.2162**；失败队列中的最大 cell 曾增至 **374**（Case A）和 **382**（Case E）。因此，需同时报告 L0 尾部/占用与后缀条件区分，并在同一冻结评估臂上测三项终点；仅报 ER 不足以判断共同移动。

## 2. 六个已关闭家族各自没有改变什么

以下“未改变”限定为记录中的实际测量或审计结论，不外推到未测机制。

1. **产品测地 level-0 assignment，以 codeword 半径评分**：没有得到有区分度的 L0 分区；24587 个 item 全部落入同一个 code（最大 cell **24587**），Hard-L1 ER 为 **1.0**。它没有改变训练表示或 codebook 的拟合结构，实际把全体样本折叠到一个 code。见 `issue112/refutation_hyp_product_l1_quantiser.md` 与 `issue112/dag/state.json`。
2. **Horospheric / Busemann level sets**：实测 level gap 没有把宿主标签分开（方向还相反）；因此未改变宿主标签可分性，也没有证据显示其改变训练后的表示、残差或 codebook 结构。见 `issue112/cgv_hyp_busemann_boundary.json`、`issue112/join_e_hyp_busemann_boundary.json`。
3. **Codeword-frame transported residual**：其良定义版本是共同坐标变换，assignment 改变量为 **0.0**；因此没有改变 assignment 或 SID 分区。见 `issue112/pt_frame_probe.json`、`issue112/pt_frame_probe_note.md`。
4. **Role-assigned token layout**：审计结论为 **VESTIGIAL**；声明的几何没有进入 forward，所以实际前向计算没有改变，预期的角色几何也没有影响 SID。见 `issue112/role_separation_check.json`、`issue112/layout_implementation_seam.md`。
5. **Gyro-composition / Möbius associator score**：两次在定义审查阶段因类型错误被拒；没有可执行的有效算子或相应实验测量。因此它没有被证明改变任何前向 assignment、SID 或训练量，不能把未运行说成测得无效。见 issue115 的候选审查记录。
6. **Unsigned tangent-space wedge area**：14 轮审议后暂停，并非实证 refutation；round 14 的未决要求是有限精度定义尚不充分。没有候选运行结果可证明它改变或改善 ER / 排序。故此家族目前未提供改变 SID 结构或任务指标的测量证据；不能称为已被实验否定。见 `issue115/join_r14/` 与 `issue115/joinr_nonconvergence.md`。

跨家族的 Issue112 结构测量也说明：仅换查询度量可以明显移动 assignment，却未改变冗余结构。欧氏与 Möbius residual chain 的“单 L3 code 的 (L1,L2) 单元比例”分别为 **0.9208** 和 **0.9263**；几何改变了单元内 residual spread / L3 最近邻间距比（**0.7895** 到 **1.4251**），冗余比例却没有下降。见 `issue112/q7_measured_obstruction.md`。这不是第 3、5、6 项的直接实验结论，而是对度量类杠杆的总体实测限制。

## 3. 一个可证伪的“ER 更好、下游更差”解释

**因果假设：** 某些优化可通过让 L0 分区尾部变粗来提高 Hard-L1 ER，但更大的最大 cell 会使一部分样本共享同一首 token、降低后缀 SID 对候选的有效区分，因而 Stage3 Recall@10 / NDCG@10 不升反降。这是待检验解释，不是已确立因果事实。历史上 14 个过 ER 门槛的臂均未改善下游，且最差尾部案例的最大 cell 从控制值 **237** 增至 **374/382**，与该解释相符，但相关性本身不能确证因果。

**区分测量：** 对控制臂和同一候选臂，用同一份 Stage2 SID 导出，成对记录 L0 最大 cell、L0 code 使用数、Gini、L1→L2 分支数，以及 (L1,L2) 单元的 L3 code 数分布（含单 code 单元比例）；随后按冻结的 Stage3 流程测同一臂的 ER、Recall@10、NDCG@10。若 ER 上升、L0 最大 cell 不增（相对控制 **237**）、占用/分支及 L3 条件丰富度不退化，且两项 Stage3 指标仍下降，则“L0 尾部变粗导致下游恶化”被证伪；若 ER 上升伴随尾部扩大、后缀条件丰富度下降而排序变差，则与假设一致。最终成功仍只按同一臂严格同时超过 ER **16.38953942232631**、Recall@10 **0.06406796775709883**、NDCG@10 **0.03530651701940677** 判定。
