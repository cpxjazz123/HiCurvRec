# iter2 文献检索：双曲 assignment / VQ-RQ 稳定性与 codebook collapse

- 检索日期：2026-09-21
- 检索范围：2023 年及以后；主题限定为可学习双曲 manifold assignment collapse、hyperbolic VQ、残差量化稳定性、dead/codebook collapse。
- 任务边界：本文档只记录检索命中、论文机制和可检验的候选思路；不包含候选之间的优劣裁决，不引入 Stage3 信号或协同信号。
- 失败背景：iter1 使用双曲 MLR/决策边界 assignment；训练约 20k 步后 L0/L1/L2 usage 为 1/1/1，30k raw SID 仅有 1/24587 个唯一组合，梯度通路检查通过。以下“可能修复”均针对该现象提出机制层面的文献启发，不等同于本仓库实验结果。

## 一、检索 query 与每个 query 的 top-3 命中

### Query 1
`hyperbolic vector quantization manifold codebook assignment collapse 2023 2024 2025`

1. **HyperVQ: MLR-based Vector Quantization in Hyperbolic Space** — Goswami, Mukuta, Harada；**2024**（arXiv:2403.13015；后续 TMLR 版本为 2025）。论文把 VQ 表述为双曲 multinomial logistic regression，并报告针对“小部分 codebook vectors 被有效使用”的 collapse 与 cluster separability 问题。URL：[arXiv](https://arxiv.org/abs/2403.13015)
2. **HiHPQ: Hierarchical Hyperbolic Product Quantization for Unsupervised Image Retrieval** — Qiu, Liu, Chen, King；**2024**。论文研究层级双曲 product quantization，并使用 hyperbolic codebook-attention 与量化对比学习构造层级离散表征。URL：[arXiv](https://arxiv.org/abs/2401.07212)
3. **HVQ-VAE: Variational Auto-encoder with hyperbolic vector quantization** — Chen, Fang, Harandi, Le, Cai, Phung；**2025**。论文将 VQ-VAE latent space 与 codebook 放在 Poincaré ball 中，并采用双曲近邻聚类与 Riemannian optimization；检索摘要报告了 codebook utilization 和收敛方面的实验比较。URL：[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1077314225001158)

### Query 2
`vector quantization codebook collapse dead code online clustering regularization 2023 2024`

1. **Online Clustered Codebook** — Zheng, Vedaldi；**2023**（ICCV 2023）。CVQ-VAE 对 inactive/dead codevectors 使用 encoder features 作为 anchors 进行在线更新，同时对 active codevectors 保持标准优化。URL：[ICCV OpenAccess](https://openaccess.thecvf.com/content/ICCV2023/html/Zheng_Online_Clustered_Codebook_ICCV_2023_paper.html)；[arXiv](https://arxiv.org/abs/2307.15139)
2. **Regularized Vector Quantization for Tokenized Image Synthesis** — Zhang et al.；**2023**（CVPR 2023）。方法使用 prior-distribution regularization、stochastic-mask regularization 和 probabilistic contrastive loss，目标是降低 codebook collapse 并提高 token 使用。URL：[CVPR OpenAccess](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Regularized_Vector_Quantization_for_Tokenized_Image_Synthesis_CVPR_2023_paper.html)
3. **Addressing Representation Collapse in Vector Quantized Models with One Linear Layer** — Zhu et al.；**2024**（arXiv:2411.02038；后续 ICCV 2025）。SimVQ 以潜在基和可学习线性变换重新参数化 codebook，使更新作用于整个线性空间，而不只作用于被选中的少数 codevectors。URL：[arXiv](https://arxiv.org/abs/2411.02038)

### Query 3
`residual vector quantization implicit neural codebooks codebook collapse 2024 2025`

1. **Residual Quantization with Implicit Neural Codebooks** — Huijben, Douze, Muckley, van Sloun, Verbeek；**2024**（ICML 2024）。QINCo 为每个残差阶段构造依赖此前向量近似的专用 neural codebook，针对固定共享 codebook 未反映阶段条件残差分布的问题。URL：[arXiv](https://arxiv.org/abs/2401.14732)；[ICML/ML Anthology](https://mlanthology.org/icml/2024/huijben2024icml-residual/)
2. **ERVQ: Enhanced Residual Vector Quantization with Intra-and-Inter-Codebook Optimization for Neural Audio Codecs** — Zheng et al.；**2024**。论文直接讨论 RVQ codebook collapse，组合 online clustering、code-balancing loss 与 inter-codebook diversity 以平衡条目使用并增加不同量化阶段之间的多样性。URL：[arXiv](https://arxiv.org/abs/2410.12359)
3. **Hyperbolic Residual Quantization: Discrete Representations for Data with Latent Hierarchies** — Piękos, Kayal, Karatzoglou；**2025**（arXiv:2505.12404）。HRQ 将多级 residual quantization 的距离、残差更新和重构转到双曲空间，使用 hyperbolic distance、Möbius subtraction/addition 与多级 codebooks。URL：[arXiv](https://arxiv.org/abs/2505.12404)

### Query 4
`representation collapse vector quantization fixed implicit codebook utilization 2023 2024`

1. **Representation Collapsing Problems in Vector Quantization** — Zhao et al.；**2024**。论文区分 token/codebook collapse 与 continuous embedding collapse，并讨论 restricted initialization、encoder capacity 等触发因素；可作为 iter1 1/1/1 现象的诊断性参考。URL：[arXiv](https://arxiv.org/abs/2411.16550)
2. **Finite Scalar Quantization: VQ-VAE Made Simple** — Mentzer et al.；**2023**（ICLR 2024）。FSQ 用少量 scalar quantizers 的笛卡尔积形成 implicit codebook，避免传统 learned vector codebook 的 dead-entry dynamics；其几何形式不是双曲空间。URL：[arXiv](https://arxiv.org/abs/2309.15505)；[ICLR](https://openreview.net/pdf?id=8ishA3LxN8)
3. **Scaling the Codebook Size of VQGAN to 100,000 with a Utilization Rate of 99%** — Zhu et al.；**2024**（VQGAN-LC，NeurIPS 2024）。方法以预训练视觉特征初始化大 codebook、冻结基础 codebook 并训练 projector，报告大规模 codebook utilization；其训练空间为欧氏空间。URL：[arXiv](https://arxiv.org/abs/2406.11837)；[NeurIPS](https://papers.neurips.cc/paper_files/paper/2024/file/1716d022edeac750e57a2986a7135e13-Paper-Conference.pdf)

## 二、候选机制（P1/P2/P3）及其可能的 collapse 修复原因

### P1：固定近邻 assignment + 双曲残差更新（HyperVQ / HVQ-VAE / HRQ 线索）

避免让每个 codevector 同时承担可学习决策边界和被选中后的更新职责：保留单一双曲几何 novelty，以 Poincaré-ball codebook、hyperbolic distance 的离散近邻选择和 Möbius-native residual update 为核心。该拆分可能减少 iter1 中 MLR 决策边界竞争造成的自强化赢家通吃，并使残差阶段仍在同一双曲几何内传播；HyperVQ、HVQ-VAE 与 HRQ 分别提供双曲 VQ、双曲 codebook 优化和双曲 residual update 的文献依据。

### P2：双曲 dead-code online clustering（Online Clustered Codebook 线索）

保持 assignment 规则不变，只对在一个窗口内没有被选中的 codevector 做显式再激活：从当前 encoder/残差特征取 anchor，并在 Poincaré ball 中用合法的流形映射或流形均值更新该 codevector。CVQ-VAE 的核心依据是让 inactive entries 重新获得代表性样本，而不是继续等待其获得梯度；在 iter1 的 1/1/1 状态下，这可能直接打破所有层只剩一个活跃条目的自强化循环，同时不需要 Stage3 或协同信号。

### P3：条件化的双曲残差 codebook（QINCo / HRQ 线索）

将第 [0mℓ[0m 层 codebook 或其候选 logits 条件化于前面层已经形成的双曲部分近似，使每个残差阶段面对与自身残差分布匹配的局部 codebook，而不是让所有阶段竞争固定且共享的几何尺度。QINCo 提供“codebook 依赖此前向量近似”的残差量化依据，HRQ 提供 hyperbolic/Möbius residual 版本的几何依据；这种条件化可能减少 L0 的单一选择向后续层级传播而造成的级联 collapse，但 QINCo 本身主要报告量化/搜索性能，因此该解释仍需单独验证。

## 三、与 iter1 失败根因的对应关系

| 观察到的现象 | 文献中对应的机制问题 | P 候选所针对的环节 |
|---|---|---|
| L0/L1/L2 usage = 1/1/1 | 少数 codevector 被持续选择，其他 entries 无有效更新 | P1 的 assignment/update 解耦；P2 的 inactive-entry reactivation |
| 30k raw SID 仅 1/24587 唯一组合 | 层级选择在早期形成自强化路径，后续 residual code 失去分辨率 | P1 的双曲近邻残差更新；P3 的阶段条件化 |
| 梯度通路通过但 usage 仍塌缩 | “有梯度”不等于所有 codevector 获得足够且均衡的训练信号 | P2 的 dead-code anchor；P3 的条件化 codebook |
| 可学习 MLR/决策边界引入额外自由度 | assignment 参数与 codebook 参数共同适应，可能放大赢家通吃 | P1 讨论的固定近邻 assignment 与几何 residual 分离 |

## 四、边界与记录说明

1. HyperVQ、HVQ-VAE、HRQ 的几何任务、数据和训练设置与本仓库 Amazon recommendation RQ-VAE 不同；这里只引用其机制描述，不把论文结果外推为本项目结果。
2. Online Clustered Codebook、Reg-VQ、SimVQ、FSQ 和 VQGAN-LC 主要是欧氏 VQ 或通用 codebook 研究；若转写为双曲版本，需要明确写出流形映射、距离和更新的定义，不能直接把欧氏更新公式当作双曲更新。
3. QINCo 与 ERVQ 关注残差量化或音频 codec；它们对本项目的启发是阶段条件化、inactive-entry 更新和层间多样性机制，而不是 Stage3 评价信号。
4. 本日志不包含候选排序、推荐、最优判断、GO/NO-GO 或训练结果判断；P1/P2/P3 仅为彼此可分离记录的候选机制名称。
