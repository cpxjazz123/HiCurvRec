# iter3 文献搜索与候选生成记录

- **检索日期**：2026-09-21
- **任务边界**：只做文献搜索和机制候选生成；不修改训练代码，不评判候选优劣，不涉及 Stage3。
- **本轮故障上下文**：iter2 的 P2 双曲 dead-code online clustering 通过梯度检查并完整运行 100k，但训练期没有触发 dead-code 条件；raw 三-token unique 为 23243/24587，Stage3 150 epoch 的 `test_R@10=0.053186859102700254`，比正式 baseline `0.05356987412733508` 低约 0.7147%。因此，本轮候选均避免把“出现未使用码字”作为唯一触发条件，关注离散边界、残差路径、中间 token 分配和 codebook 更新密度。

## 已读取的项目参考

已读取当前机制池：

`/home/wlia0047/ar57/wenyu/GeneRec/.curvature-rqvae-iter-skill/references/mechanism_pool.md`

该池已列出 cyclic `c(t)`、layer-wise curvature、Lorentz、adaptive margin、Sinkhorn `c`-dependent epsilon、Riemannian Adam、temperature curriculum、negative curvature、product manifold 和 hierarchical curvature 等方向；本记录中的候选不再把这些已有项目作为新颖机制本身。

在项目树及 `/home/wlia0047/ar57/wenyu` 下检索 `lit_search_iter2.md`、`direction_decision`、`iter2 audit` 等文件，未发现可读取的 iter2 文献搜索/方向决策审计文件。因此，iter2 的故障事实以任务说明和现有训练代码为准，未对不存在的审计内容作推断。

当前量化实现中，`modules/quantize.py` 使用 Poincaré 距离、Sinkhorn assignment、argmax ID 和训练态 STE；`modules/rqvae.py` 使用逐层 residual、M2 intrinsic residual、M3 跨曲率 transport，再汇总三层 embedding。故本轮重点区分：

1. **几何表示的因子化/码本参数化**，不是再次改变曲率日程或 Sinkhorn epsilon；
2. **Semantic-ID 残差路径与中间 token 的稳定性**，不是再次添加碰撞扩展；
3. **始终有效的 codebook/STE 更新机制**，不是仅在 dead-code 计数达到阈值时更新。

---

## P1：双曲乘积流形上的因子化码本与几何一致量化

> 这是一个独立候选方向：把 latent/codebook 的几何因子化和量化决策作为核心，不改变现有 cyclic curvature 公式，也不使用 layer-wise learned curvature 作为主要机制。

### 三篇文献命中

1. **HiHPQ: Hierarchical Hyperbolic Product Quantization for Unsupervised Image Retrieval**（2024，AAAI 2024）  
   [arXiv:2401.07212](https://arxiv.org/abs/2401.07212) · [AAAI 论文](https://ojs.aaai.org/index.php/AAAI/article/download/28261/28514)

   论文将表示放入双曲乘积结构，使用 hyperbolic product quantizer、hyperbolic codebook attention 和层级语义监督。其可借鉴点是将一个大码本的离散选择拆成多个几何因子，并让不同因子共同表达层级关系，而不是单纯依赖一次全空间最近邻。

2. **HyperVQ: MLR-based Vector Quantization in Hyperbolic Space**（2024；页面含 2025 修订版）  
   [arXiv:2403.13015](https://arxiv.org/abs/2403.13015)

   论文把双曲量化表述为 hyperbolic multinomial logistic regression，码向量对应双曲决策超平面的几何参数，而不是普通欧氏聚类中心。摘要强调双曲空间的体积增长和决策间隔可以改善离散表示的可分性与解耦；页面没有给出足以直接复现的完整 codebook 更新公式，因此这里只提取机制思想，不把未公开细节当作事实。

3. **HVQ-VAE: Variational Auto-Encoder with Hyperbolic Vector Quantization**（2025）  
   [ScienceDirect 论文页面](https://www.sciencedirect.com/science/article/pii/S1077314225001158)

   该工作将 VQ-VAE codebook 直接约束在 Poincaré ball，并使用 Riemannian optimization 保持码向量位于双曲流形。与当前实现只在距离计算和 residual/transport 中使用双曲运算不同，该方向把“码本参数自身的流形约束”作为量化器的一部分。

### 机制摘要

候选机制可以抽象为：将 embedding/codebook 划分为若干几何因子（例如乘积流形因子或切空间子块），每个因子执行 manifold-aware quantization，再以稳定的组合规则生成 Semantic ID；码本参数的更新始终经过对应流形的投影/重参数化。这里的关键变化是**码本结构和决策边界**，而不是再次调整 `c(t)`、每层曲率、Sinkhorn epsilon 或 optimizer。

### 为何可能针对 iter2 失败

- iter2 的 dead-code 条件没有触发，说明“重新激活长期零使用码字”不能解释或覆盖这次 regression；P1 不依赖零使用统计，而是让活跃码字、边界附近样本和残差因子共享几何结构。
- raw unique 接近 item 数并不保证 Semantic ID 对输入微扰稳定；不同因子边界若在全空间纠缠，微小的 encoder/residual 改变可能导致一个或多个 token 翻转。乘积流形的因子化决策可把这种翻转限制在局部因子，降低整条三-token 路径被改变的概率。
- 当前实现已有 Poincaré distance，但这不等同于“码本在流形上结构化”。P1 的可检验差异是码本参数的几何约束及因子化，而不是重复现有距离函数。
- 层级 residual 中，前层量化误差会传播到后层；几何因子化可能使不同尺度/方向的残差信息由不同子空间承载，从而减少中间 token 改变对后续 token 的级联影响。

### 可观察的文献对应现象

- 码本因子之间的使用是否比单一码本更平滑；
- 相邻 encoder 输入或 checkpoint 之间的 SID 汉明变化率；
- 单层 token 翻转是否会引起后续层的级联翻转；
- 重构误差相近时，Semantic-ID 路径是否更稳定。

这些是候选机制的诊断维度，不构成采用标准或结果判断。

---

## P2：面向残差 Semantic-ID 路径的中间 token 稳定化与 anti-hourglass 分配

> 这是一个独立候选方向：直接处理 RQ-SID 的中间层 token 分配、残差路径和语义邻域一致性，不以 dead-code 计数或简单 collision 扩展作为主要机制。

### 三篇文献命中

1. **Breaking the Hourglass Phenomenon of Residual Quantization: Enhancing the Upper Bound of Generative Retrieval**（2024，EMNLP 2024 Industry Track）  
   [ACL Anthology: 2024.emnlp-industry.50](https://aclanthology.org/2024.emnlp-industry.50/) · [arXiv:2407.21488](https://arxiv.org/abs/2407.21488)

   论文研究 RQ-SID 的 hourglass 现象：中间层 code token 的使用过度集中，造成有效 ID 路径稀疏和码本利用受限；论文将数据稀疏和长尾分布视为重要原因，并提出针对性的缓解方案。其问题定义与“三 token unique 很高但下游性能下降”直接相邻，因为全局 unique 不会揭示中间 token 的路径集中和条件分布失衡。

2. **CoST: Contrastive Quantization based Semantic Tokenization for Generative Recommendation**（2024，RecSys 2024）  
   [arXiv:2404.14774](https://arxiv.org/abs/2404.14774) · [DOI](https://doi.org/10.1145/3640457.3688178)

   CoST 用 contrastive quantization 学习 semantic item tokens，同时纳入 item neighborhood relationships 和语义信息。其出发点是 reconstruction-only 的 RQ-VAE 不一定保留推荐所需的邻域关系；对比式目标可使相似 item 的离散路径保持可用的相对结构，而不是只追求输入重构。

3. **LETTER: Learnable Item Tokenization for Generative Recommendation**（2024，CIKM 2024）  
   [arXiv:2405.07314](https://arxiv.org/abs/2405.07314)

   LETTER 以 RQ-VAE 作为层级语义正则，同时融合 collaborative signal、contrastive alignment 和 diversity loss。其机制重点是把 item token 的语义、协同关系与 code assignment diversity 放在同一 tokenization 目标中，减少单一语义重构造成的分配偏置。

### 机制摘要

候选机制可以抽象为一个**路径级而非码字级**的正则化：对每个 item 的三层 residual quantization path，约束相邻/语义相近 item 的前缀 token、后缀 token 和 residual 关系；对中间层的条件占用进行平滑或长尾补偿；同时用局部邻域对比目标避免仅凭 reconstruction loss 学出对下游推荐不稳定的 token 边界。其关键对象是 `p(L1 | L0)`、`p(L2 | L0,L1)`、相邻样本的路径一致性和残差层级，而非“某个 code 是否从未被使用”。

### 为何可能针对 iter2 失败

- iter2 的 dead-code 机制没有触发，而 hourglass/路径集中可以在**所有码字都有一定使用**时发生；中间 token 仍可能过度承担高频前缀，故需要直接观察条件分布和路径稳定性。
- 当前 `sid_quality.py` 的 full Gini、per-layer Gini、L0-L1 unique pairs 与 `H(L1|L0)` 是描述性指标，但它们不能测量同一 item 在邻近 latent/训练 checkpoint 下的 SID 稳定性，也不能测量相似 item 的前缀是否一致。P2 的文献启发正好把这些“条件路径”现象作为机制对象。
- 当前已有 collision extension，解决的是 token 冲突后的可区分性；P2 关注的是冲突发生以前的分配偏置、长尾路径和语义邻域保持，因而不是重复添加 disambiguation token。
- Sinkhorn 已在当前量化器中使用，且先前已经探索过 `c`-dependent epsilon；P2 的独立性在于用 residual-path/semantic-neighborhood 信号改变训练目标或路径一致性，而不是再调 assignment temperature/epsilon。
- 下游 regression 说明“重构良好且 SID unique 较高”并不充分；对比式邻域保持和路径级 diversity 提供了与推荐序列目标更接近的约束来源。

### 可观察的文献对应现象

- `H(L1|L0)` 与 `H(L2|L0,L1)` 的条件变化，而不仅是边际 Gini；
- 同一 item 在 encoder 输入微扰、不同 batch 顺序和 checkpoint 间的 token flip rate；
- item-neighbor 对的 SID 前缀一致率与 residual 距离；
- 中间 code 的条件长尾程度和每个前缀下的有效后缀数。

这些指标用于检验候选所针对的 failure mode，不是采用门槛。

---

## P3：不依赖 dead-code 触发的稠密 STE/codebook 更新与重参数化

> 这是一个独立候选方向：改变 codebook/STE 的更新拓扑，使有效更新不再取决于“某个码字长期未被选择”；不改变 Stage3，不以在线 dead-code reinitialization 作为唯一动作。

### 三篇文献命中

1. **Straightening Out the Straight-Through Estimator: Overcoming Optimization Challenges in Vector Quantized Networks**（2023，ICML 2023）  
   [PMLR: Huh et al.](https://proceedings.mlr.press/v202/huh23a.html) · [arXiv:2305.08842](https://arxiv.org/abs/2305.08842)

   论文分析 VQ 中 embedding/code-vector 分布错配、STE 梯度误差、codebook gradient sparsity 和 commitment loss 非对称性，并讨论 affine re-parameterization、alternating optimization 与改进 commitment 对齐。其核心启示是：即使没有完全 dead 的 code，只有少量被选 code 得到梯度也会使决策边界和 encoder 分布逐渐错位。

2. **Online Clustered Codebook**（2023）  
   [arXiv:2307.15139](https://arxiv.org/abs/2307.15139)

   该工作提出 CVQ-VAE，通过编码特征中的锚点为长期未活跃 code 提供在线更新，同时让活跃 code 继续接受原始量化损失。这里可借鉴的不是原样复制 dead-code 条件，而是把 codebook 更新看作在线聚类问题，并使更新参考当前 feature 分布，而非只依赖被 argmax 选中的单个 code。

3. **Addressing Representation Collapse in Vector Quantized Models with One Linear Layer（SimVQ）**（2024）  
   [arXiv:2411.02038](https://arxiv.org/abs/2411.02038)

   SimVQ 用可学习线性变换重参数化一组基础 code vectors，使一次更新能在整个 codebook 的线性空间中产生联动，而不是只更新被选中的少量独立码向量。论文将 representation collapse 与 vanilla VQ 的局部 code 更新联系起来。

### 机制摘要

候选机制可以抽象为以下互相独立、需单独实现和验证的更新思路：

- 用 affine/linear re-parameterization 生成 manifold codebook，使 codebook 更新在全体基础向量上具有稠密影响；
- 将 hard argmax 的训练信号与 soft assignment、近邻候选或 feature-anchor 更新结合，使边界附近 code 也获得有效学习信号；
- 将 STE 的 encoder-side gradient 与 codebook-side update 解耦或交替，减少 `ids = argmax(assignments)` 导致的梯度稀疏和 distribution mismatch。

这些思路可以在双曲坐标的切空间或合法流形参数化下研究；候选定义本身不指定实现细节，也不把“未使用超过固定步数”当作必要条件。

### 为何可能针对 iter2 失败

- iter2 已证明 dead-code online clustering 的梯度通路正确，但运行期间没有 dead code；这只说明触发式分支未被执行，不说明 codebook 更新足够密集。P3 直接绕过该触发条件，让普通活跃/近活跃 code 的更新也改变相邻决策边界。
- 当前 `modules/quantize.py` 在 Sinkhorn assignment 上 `detach()` 后取 `argmax`，embedding 使用训练态 STE；因此可能出现“总 unique 很高、每个 code 也被使用，但 codebook 梯度集中在少数路径”的状态。P3 针对的是这种 gradient sparsity，而不是 code 使用计数。
- SimVQ 和 affine STE 文献提供了从参数化层面减少 code-vector 独立更新的路线；这与当前已有的 Poincaré distance、M2/M3 residual/transport 和 cyclic `c(t)` 正交。
- 若 regression 来源是 SID 边界对微小 latent 漂移过于敏感，稠密/平滑的 codebook 更新可降低相邻边界的随机漂移；若来源是 encoder-codebook mismatch，则 alternating/affine 对齐可直接作用于该错配。

### 可观察的文献对应现象

- 每一步各 code 的梯度范数分布、有效梯度 code 数和 top-1 梯度集中度；
- top-1 与 top-k assignment 的边界 margin，以及 checkpoint 间 margin 漂移；
- codebook 更新前后所有向量的位移是否只集中在已选 code；
- 相同 raw unique 下，SID flip rate、邻近样本路径一致率与重构损失的变化。

这些诊断用于区分“dead-code 没发生”和“codebook 更新仍然稀疏”这两个不同现象。

---

## 方向去重与范围说明

- **不重复 MLR assignment**：P3 的 MLR/STE 文献关联的是梯度稀疏、参数化与错配，不提出再次采用已有 MLR assignment 路线。
- **不重复 per-layer curvature / Sinkhorn `c`-dependent epsilon / Riemannian Adam**：P1 的改变对象是 product-manifold codebook 结构；P2 的改变对象是 Semantic-ID 路径目标；P3 的改变对象是 codebook 参数化和 STE 更新拓扑。
- **不重复 collision extension**：P2 处理的是冲突前的条件路径分配、hourglass 和邻域稳定性，不增加 token 长度作为主要机制。
- **不重复 dead-code online clustering**：P3 明确要求在没有 dead code 时仍产生有效更新；Online Clustered Codebook 只作为该候选的文献来源，用于说明在线聚类/anchor update 的背景。
- **不评价候选排序**：P1、P2、P3 是三个彼此可分离的候选包；本文件不写推荐、最优、GO/NO-GO 或采用结论。

## 参考文献链接汇总

- [TIGER: Recommender Systems with Generative Retrieval（NeurIPS 2023）](https://arxiv.org/abs/2305.05065)
- [Better Generalization with Semantic IDs（2023）](https://arxiv.org/abs/2306.08121)
- [Finite Scalar Quantization: VQ-VAE Made Simple（2023 arXiv / ICLR 2024）](https://arxiv.org/abs/2309.15505)
- [HiHPQ（2024）](https://arxiv.org/abs/2401.07212)
- [HyperVQ（2024）](https://arxiv.org/abs/2403.13015)
- [CoST（2024）](https://arxiv.org/abs/2404.14774)
- [LETTER（2024）](https://arxiv.org/abs/2405.07314)
- [Breaking the Hourglass Phenomenon（2024）](https://arxiv.org/abs/2407.21488)
- [SimVQ（2024）](https://arxiv.org/abs/2411.02038)
- [HVQ-VAE（2025）](https://www.sciencedirect.com/science/article/pii/S1077314225001158)
- [Straightening Out the Straight-Through Estimator（2023）](https://proceedings.mlr.press/v202/huh23a.html)
- [Online Clustered Codebook（2023）](https://arxiv.org/abs/2307.15139)
