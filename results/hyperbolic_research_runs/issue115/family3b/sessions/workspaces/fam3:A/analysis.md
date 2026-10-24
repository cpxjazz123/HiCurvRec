# 冻结协议下的新机制必须移动什么

## 1. 必须移动的量：SID 的有效条件信息，而非仅仅重排最近码字

现有测量没有证明某一个标量能够因果地同时提升 Hard-L1 ER 和 Stage3 排序；但它指出了必要的结构目标：**三个 SID token 在条件于前缀后实际携带的信息，尤其是 L3 在给定 (L1,L2) 后的有效区分能力**。仅改变几何度量或码字分配、却不改变这种结构，不足以支持共同提升的主张。

冻结 host 的 4000 项测量显示：各层 codebook 使用数分别为 **243/256、255/256、256/256**；然而在 **3612** 个不同的 (L1,L2) 单元中，平均每格只有 **1.0969** 个 L3 code，**92.08%** 的格子只有一个 L3 code（`results/hyperbolic_research_runs/issue112/q7_measured_obstruction.md`；`residual_information_baseline.json`）。所以“codebook 用满”不是瓶颈证据；第三 token 在给定前缀后的冗余才是直接测出的结构障碍。

这也不是说只要提升 L3 分支数就必然提升排名。实际任务仍要求同一机制同时严格超过 ER **16.38953942232631**、Recall@10 **0.06406796775709883**、NDCG@10 **0.03530651701940677**。已有反例：R4-08 ER 为 **16.402627511591962**，但 Recall@10 **0.06295374223088843**、NDCG@10 **0.03454767796170093**，后二者均未达门槛（`results/stage2_RQ-VAE/R4_SUMMARY.md`）。因此应把条件信息/分支结构作为必须测量的中间量，而不能宣称它已被证明是 ER 与排序共同改善的充分因果量。

## 2. 六个已关闭家族各自没有改变什么

| 已关闭家族 | 测量或裁定 | 留下未改变的关键量 |
|---|---|---|
| 1. 乘积测地 L0、按码字半径评分 | 全部项目落到同一码；实跑 distinct codes **1**、最大 cell **24587**、ER **1.0**（`issue112/refutation_hyp_product_l1_quantiser.md`） | 没有形成有信息量的 L0 划分；失败结果没有 Stage3，故不构成排序改善证据。其问题是半径支配分配，不是有效条件 SID 信息被提高。 |
| 2. Horospheric / Busemann level sets | 记录判定 level gap 未能分离 host 标签，方向错误；测得近径向等距算子 | host label 的可区分性没有改善；只引入 level/径向关系，未改变有用的条件分支结构。 |
| 3. Codeword-frame transported residual | 判为定义不适定；适定版本是恒等变换 | 残差表示/分配不变；恒等形式实测 assignments changed **0.0**。 |
| 4. Role-assigned token layout | 实测为 **VESTIGIAL**：声明的几何没有进入 forward | forward 中实际计算及其产生的 SID 不变；角色声明本身没有改变有效信息结构。 |
| 5. Gyro-composition / Möbius associator score | 两次在定义层面因类型错误拒绝 | 没有可执行的评分或分配变化；因此没有移动任何测量结果。 |
| 6. Unsigned tangent-space wedge area | 14 轮未获一致认可后搁置；不是被证伪的候选 | 没有完成可接受的测量证明其改变真实 assignment-stream 的有效条件信息。它不是被数据 refute，而是没有资格作为候选继续推进。 |

LOT​​V tie-break（作者撤回）另属 6b 而非上述六个家族：缺少预注册的每层至少 **5%** tie-rate 证据；其可能触发的稀有情形频率未先测量，故不能作为有效机制主张。

## 3. 一个可证伪的“ER 更好、下游更差”因果解释

**解释（假说，不当作已证实因果）：** 某些改动通过扩大 L0 尾部/最大 cell，提高了 Hard-L1 ER 所奖励的行为信号，但合并了下游推荐所需区分的项目；于是 ER 上升而 Stage3 排序下降。它预测：相对控制，ER 上升的 arm 会同时出现 L0 最大 cell 增大（以及更差的前缀条件分支结构），而在控制该尾部变化或比较尾部未增长的 arm 时，ER 上升不再系统性伴随 Recall@10/NDCG@10 下降。

**判别测量：** 对同一冻结数据和评估流，逐 arm 配对记录 Hard-L1 ER、Stage3 Recall@10/NDCG@10、L0 largest-cell（控制为 **237**）、L0 distinct-code 数（控制为 **245**）、L0 Gini（控制 **0.2162**）及 L1→L2 branches（控制 **65.91**），并检查 L3 在每个 (L1,L2) cell 的条件分支数。现有 cohort 摘要报告 **14** 个过 ER gate 的 arm 均未改善 downstream，尾部恶化的案例由 **237** 增至 **374**（Case A）或 **382**（Case E）（`results/hyperbolic_research_runs/issue112/reading_sheet_after_the_run.md`）。若在可比 arm 中 ER/下游背离不随尾部增长出现，或尾部增长的 arm 并不更常下游变差，该解释即被反驳。单凭现有汇总相关性不能判定因果；这个配对测量正是区分“尾部合并导致损害”与“ER 指标本身和下游目标不一致/其他因素”的必要证据。
