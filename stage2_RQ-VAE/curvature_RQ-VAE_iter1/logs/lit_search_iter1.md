# iter1 文献搜索记录

## 上一轮失败根因

当前正式基线使用全局 cyclic curvature `c(t)`（`c_min=0.3`、`c_max=1.0`、`period=50000`）。Stage3 使用刚生成的四 token SID 完整训练 150 epoch 后，最终 `test_R@10=0.05356987412733508`，低于硬目标 `0.065`。本轮失败不能归因于 Stage2 未完成或 SID 导出错误：SID 形状为 `(24587, 4)`、全表唯一，Stage3 日志确认 `n_digit=4`。因此需要区分几何分配、码本优化和推荐语义之间的失配。

## Query 1：hyperbolic/manifold residual VQ、codebook collapse 与 diversity remedy

### Top-3 命中

1. **HyperVQ: MLR-based Vector Quantization in Hyperbolic Space**（2024）
   - URL：https://arxiv.org/abs/2403.13015
   - 相关事实：将双曲量化建模为多类逻辑回归/双曲决策边界，而不是单纯最近双曲距离；论文以 codebook collapse 为动机之一，并报告更好的判别性能与更解耦的潜表示。
   - 与当前失败的关系：直接针对距离分配造成的码字使用不足，但没有生成式推荐或 RQ-VAE-SID 的下游实证。

2. **HiHPQ: Hierarchical Hyperbolic Product Quantization for Unsupervised Image Retrieval**（AAAI 2024）
   - URL：https://arxiv.org/abs/2401.07212
   - AAAI 页面：https://ojs.aaai.org/index.php/AAAI/article/view/28261
   - 相关事实：使用双曲乘积量化、双曲码本注意力和量化对比学习，并通过分层语义监督保留多层语义关系。
   - 与当前失败的关系：说明分层语义与双曲量化可以联合训练，但任务是图像检索，未证明 Amazon 生成式推荐有效。

3. **ERVQ: Enhanced Residual Vector Quantization with Intra-and-Inter-Codebook Optimization for Neural Audio Codecs**（2024）
   - URL：https://arxiv.org/abs/2410.12359
   - 相关事实：在码本内部使用在线聚类和 code-balancing loss，在连续 residual codebook 之间做 inter-codebook 优化，报告更高码本利用率和层间多样性。
   - 与当前失败的关系：与多层 RQ-VAE 的层内使用率、层间残差冗余问题对应，但没有推荐下游证据。

### 候选 P1：双曲 MLR/决策边界式量化

从 HyperVQ 抽取的候选是：用双曲空间中的可学习多类决策边界参与 code assignment，而不是只使用最近双曲距离。该候选可能改变码本覆盖和分配边界。

与现有 cyclic `c(t)` 的潜在冲突：cyclic 曲率会令量化边界处于非平稳几何环境；如果 MLR 参数在固定曲率下学习，训练边界与最终 SID 推理边界可能失配。若把 `c(t)` 直接并入 logits，还需重新定义曲率对超平面、温度和梯度的影响。

## Query 2：Riemannian optimizer / curvature-aware quantization 与生成式推荐下游

### Top-3 命中

1. **Learnable Item Tokenization for Generative Recommendation（LETTER）**（2024）
   - URL：https://arxiv.org/abs/2405.07314
   - 官方实现：https://github.com/honghuibao2000/letter
   - 相关事实：使用 RQ-VAE 生成 semantic ID，并加入语义正则、协同信号对比学习和 diversity loss，直接在生成式推荐中验证。
   - 与当前失败的关系：说明单纯改善量化空间或码本统计不一定改善推荐；code-assignment bias、协同关系和 diversity 目标可能更接近 Stage3 需求。

2. **CoST: Contrastive Quantization based Semantic Tokenization for Generative Recommendation**（RecSys 2024）
   - URL：https://arxiv.org/abs/2404.14774
   - ACM DOI：https://doi.org/10.1145/3640457.3688178
   - 相关事实：指出 reconstruction-only tokenization 不能充分保留 item-neighborhood 关系，通过 contrastive quantization 将项目关系和语义信息引入离散 token 学习。
   - 与当前失败的关系：直接对应“SID Gini/碰撞改善但 Stage3 `R@10` 下降”的现象；但该候选会引入推荐协同目标，不能直接视为纯曲率证据。

3. **Robust Hyperbolic Learning with Curvature-Aware Optimization**（2024）
   - URL：https://arxiv.org/abs/2405.13979
   - 相关事实：推导 Lorentz 模型下的 Riemannian AdamW，研究可调曲率、双曲缩放和曲率感知优化对训练稳定性的影响；未直接研究 RQ-VAE 或生成式推荐。
   - 与当前失败的关系：提供 Riemannian 优化和曲率缩放的实现依据，但不能推断其必然提升推荐 `R@10`。

### 候选 P2：残差码本的层内均衡与层间去冗余

从 ERVQ、LETTER 和 CoST 的共同方向抽取：控制每层码字使用分布，减少连续 residual codebook 的重复编码，并可将 item-neighborhood 信号纳入离散 token 学习。该候选关注码本优化与推荐语义，而不只是改变流形距离。

与现有 cyclic `c(t)` 的潜在冲突：强制均衡可能破坏有用的非均匀粗粒度结构；层间去冗余可能抑制 RQ-VAE 后续层对前层误差的必要重复表达；额外 diversity/contrastive 梯度可能与 cyclic 曲率造成梯度竞争，指标改善也不保证推荐生成改善。

## Query 3：adaptive curvature / curvature-dependent quantization difficulty 与 cyclic `c(t)`

### Top-3 命中

1. **Curvature-Adaptive Meta-Learning for Fast Adaptation to Manifold Data**（2023）
   - URL：https://pubmed.ncbi.nlm.nih.gov/35380955/
   - PDF：https://wu-yuwei-bit.github.io/paper/TPAMI2023_Gao_Curvature-Adaptive-Meta-Learning.pdf
   - 相关事实：在乘积流形上学习任务相关曲率及曲率更新机制，在 few-shot classification、regression 和 reinforcement learning 中验证。
   - 与当前失败的关系：支持不同数据区域或层级可能需要不同曲率，但没有残差量化或推荐生成验证。

2. **Adaptive Discrete Communication Bottlenecks with Dynamic Vector Quantization for Heterogeneous Representational Coarseness**（AAAI 2023）
   - URL：https://ojs.aaai.org/index.php/AAAI/article/view/26061
   - 相关事实：根据输入复杂度动态调整离散瓶颈的粒度，让不同复杂度样本使用不同离散表示。
   - 与当前失败的关系：支持量化难度不应对所有 item、层级和 residual 统一，但任务是视觉推理和强化学习。

3. **Robust Hyperbolic Learning with Curvature-Aware Optimization**（2024）
   - URL：https://arxiv.org/abs/2405.13979
   - 相关事实：研究可学习曲率、曲率感知优化和 Lorentz 几何中的 Riemannian AdamW，报告曲率缩放对稳定性和优化效率的影响。
   - 与当前失败的关系：可作为自适应曲率优化参考，但没有 RQ-VAE 或生成式推荐证据。

### 候选 P3：按层级、residual 难度或 item 复杂度自适应曲率/量化粒度

候选形式包括由 residual norm、局部密度或 assignment entropy 决定曲率；由层级 `l` 使用不同 `c_l`；或按输入复杂度动态调整量化温度/粒度。它把全局时间驱动的 cyclic schedule 扩展为数据或层级条件控制。

与现有 cyclic `c(t)` 的潜在冲突：两个曲率控制器会造成不可辨识的非平稳闭环；曲率改变距离、assignment、residual 和后续层输入，可能形成 difficulty→curvature→assignment→difficulty 的反馈；训练与最终 SID 推理若条件不一致，会产生分布偏移。

## 补充对照证据

- **HQ-VAE: Hierarchical Discrete Representation Learning with Variational Bayes**（2023/2024）：https://arxiv.org/abs/2401.00365。针对 hierarchical VQ-VAE、VQ-VAE-2 和 RQ-VAE 的 codebook/layer collapse，报告利用率改善，但不提供双曲推荐证据。
- **Regularized Vector Quantization for Tokenized Image Synthesis**（CVPR 2023）：https://arxiv.org/abs/2303.06424。使用 uniform-prior regularization、stochastic mask 和 probabilistic contrastive loss 促进码本使用，但不改变曲率。
- **Finite Scalar Quantization: VQ-VAE Made Simple**（ICLR 2024）：https://arxiv.org/abs/2309.15505。通过有限标量量化绕过学习型码本 collapse，适合做机制对照，但不能直接与当前双曲 RQ-VAE 叠加。
- **Recommender Systems with Generative Retrieval（TIGER）**（NeurIPS 2023）：https://proceedings.neurips.cc/paper/2023/hash/20dcab0f14046a5c6b02b61da9f13229-Abstract.html。建立 RQ-VAE semantic ID 到 Transformer 生成推荐的链路，说明最终裁决仍应是 Stage3 下游指标。

## 文献层面的根因分解

1. 几何分配层：曲率改变最近码字边界，可能造成新的 collapse、碰撞或层间冗余。
2. 码本优化层：即使几何合理，码字更新、层间重复和不均衡使用仍可能造成失效。
3. 推荐语义层：码本统计改善不等于 item-neighborhood 和用户行为关系改善；LETTER/CoST 直接指出 reconstruction-only tokenization 的缺口。

以上为搜索记录，不包含候选优先级判断。