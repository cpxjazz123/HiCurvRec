# 冻结协议下的实测瓶颈分析

## 1. 哪个量能同时推动 ER 与 Stage3 排序？

**现有测量没有证明存在一个可移动的单一量，能让冻结 Hard-L1 ER 与 Stage3 排序一起变好。** 尤其不能把 ER 当作下游排序的代理：R4-08 的 ER 为 **16.402627511591962**，高于冻结门槛 **16.38953942232631**，但 Stage3 Recall@10 **0.06295374223088843**、NDCG@10 **0.03454767796170093**，分别低于目标 **0.06406796775709883**、**0.03530651701940677**（`results/stage2_RQ-VAE/R4_SUMMARY.md`）。

唯一明确测出的结构性瓶颈是第三层的**条件信息/分支丰富度**：4000 个样本中，3612 个不同的 (L1,L2) 单元平均仅有 **1.0969** 个 L3 code，**92.08%** 的单元只有一个 L3 code；各层 codebook 却已使用 **243/256、255/256、256/256**（`results/hyperbolic_research_runs/issue112/q7_measured_obstruction.md`）。因此“codebook 是否用满”不是杠杆；可测的结构量是各层 SID 实际编码的信息及其条件分配，而非几何距离本身。现有记录并未测出该量的变化会同时提升 ER 与 Stage3 排序，故不将其宣称为已验证的共同因果量。

## 2. 失败臂实际改了什么、没改什么

- **几何/距离分配类（issue112）**：改的是固定表示和 codebook 上的 assignment 距离/评分，或对表示作共同坐标变换；没有改变数据、编码器训练出的残差尺度及 codebook 的拟合结构。产品测地 L1 的 level-0 评分被 codeword 半径支配，所有 item 分到同一码，ER **1.0**；Möbius 位移把单元内残差 spread / L3 最近邻间距从欧氏的 **0.7895** 改到 **1.4251**，但单 L3 code 的单元比例仍从 **0.9208** 到 **0.9263**，没有消除冗余。参见 `q7_measured_obstruction.md`、`refutation_hyp_product_l1_quantiser.md`。
- **Busemann/horospheric 家族**：改变 level-set/边界评分；实测 level gap 没有分开宿主标签且方向相反，测到的是近径向等距算子（`cgv_hyp_busemann_boundary.json`、`join_e_hyp_busemann_boundary.json`）。没有改变训练后的表示、codebook 或数据。
- **transported residual / frame 家族**：改变残差坐标框架；可良定义版本只是共同坐标变换，assignment 改变量 **0.0**（`pt_frame_probe.json`、`pt_frame_probe_note.md`）。没有改变残差的信息或 codebook 学到的结构。
- **role-assigned layout**：声明了 token 的角色/布局，但审计发现声明的几何未进入 forward（VESTIGIAL；`role_separation_check.json`、`layout_implementation_seam.md`）。所以实际前向计算没有改变。
- **RQ-VAE 辅助目标家族（R8 十机制）**：在保持原数据、训练日程、PCGrad/loss assembly、AdamW 和 scheduler 不变的冷启动训练中，加入各自的行为/图/几何辅助项；未改变冻结协议与 Hard-L1 评估。十臂 ER 全部未过门槛，范围 **1.0006946–15.9130435**，新鲜冷控制为 **16.3587375**；Stage3 均未运行。部分机制重组 L1 codebook 并导致塌缩，另一些只重加权并损失 ER（`results/stage2_RQ-VAE/r8_ten_mechanisms/SYNTHESIS.md`）。

## 3. 一个可证伪的因果解释及区分测量

**解释（假设）：** ER 奖励的是 L1 上正负配对碰撞比例的变化，而下游推荐依赖更深层 SID 对候选的区分；如果提升 ER 的训练让 L0 分区尾部变粗，它会提高离线 ER，却减少有效的后缀区分，因此 Stage3 变差。这不是已证实结论：它可被同一候选的分区及下游测量证伪。

**已知测量依据：** 历史队列中 **14 个过 ER 门槛的臂没有一个改善下游**；最差的尾部案例把 L0 最大 cell 从控制的 **237** 增到 **374**（Case A）或 **382**（Case E）。控制 L0 使用 **245/256** 个 code、Gini **0.2162**；R4-08 虽过 ER 门槛，Recall@10/NDCG@10 仍未达阈值。来源：`reading_sheet_after_the_run.md` 与 `R4_SUMMARY.md`。

**区分测量：** 对同一候选、同一 Stage2 导出的 SID，记录 L0 最大 cell、使用 code 数、Gini，以及 L1→L2 分支数；再按冻结流程测该同一候选的 ER、Recall@10、NDCG@10，并与控制值并列。若 ER 提升且 L0 尾部不增、同时两项 Stage3 排序改善，该“尾部变粗导致下游恶化”的解释被证伪；若 ER 提升伴随尾部增大而 Stage3 不升/下降，则与该解释相符。控制参照为最大 cell **237**、使用 code **245**、Gini **0.2162**、L1→L2 分支 **65.91**。严格成功仍要求同一候选同时超过三项冻结阈值。
