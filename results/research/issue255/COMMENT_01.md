冻结 #255 的 13 条验收原文与正文 SHA256 `2fa23b1ee2faa897155f2a12999b3cf4cd7c085a5fa0a5f8a99c53f54b1c361d`（`results/research/issue255/ACCEPTANCE.json`）。

**需要训练**：第10条明确要求一次正式四卡 Stage2（3000 epoch）与一次冻结 Stage3（150 epoch），碰撞门槛与 R@10 只能由新权重产生。因此本轮按 §4 走训练路径；不建第二个臂、不做多种子、不追分调参、不重训 TIGER。

新臂 `stage2_RQ-VAE/TIGER_prefix_curvature_gate_rq`，是已归档 `TIGER_three_expert_attention_rq`（其本身为冻结 baseline + #253 三专家机制）的独立副本，只改 `ARM_NAME` 一个常量。保留 768/32、三层256码、#253 实际分配协议（L1/L2 floor-constrained nearest、L3 Sinkhorn）、原重建/commitment/codebook、四卡 global4096、seed42、专家曲率仅 {-1,0,+1}、GNN 与门控端到端可训练。新增仅限框架内部：方向感知 token、逐层三曲率门控、累计前缀截面监督。不新增行为邻域约束、pair ranking、InfoNCE、边预测或额外行为 Loss。

按 §1.0a：唯一剩余的顶层机制目录 `TIGER_three_expert_attention_rq` 已 `git mv` 到 `_archive/`（`baseline/`、`_archive/` 未动）；当前顶层只有 baseline、_archive 与新候选，且 intake 时无训练进程、四张 L40S 空闲。

数学契约 `CONTRACT.md` 已先行冻结（四类 token 的节点/边集合与不互斥声明、w_i^(l) 归一、累计前缀 q_i^(l) 逐位对应 SID 前 l 码、逐层 A.1 代理 K_l、α_l=1/3 与 λ 的一次性梯度校准规则、尺度冻结规则、容差与断言清单）。接下来按 TDD 红→绿实现，并按第8条只在确需时新建方向 ego 资产。
