# iter5 文献检索与候选机制报告

- 数据集：Amazon-2023 Instruments
- 检索日期：2026-09-21
- 上轮失败：iter4 `iter4_lorentzian_centroid_recenter` 的 Stage3 `test_R@10=0.054022528247358065`，低于 `0.065` 硬目标。
- 本报告角色：Agent A（literature-hunter）。仅围绕失败根因检索并生成候选；P1/P2/P3 编号不表示选择、优劣或推荐。
- 机制池参考：`/home/wlia0047/ar57/wenyu/GeneRec/.curvature-rqvae-iter-skill/references/mechanism_pool.md`。

## 1. 上轮失败根因的精准界定

iter4 的改动是 Lorentzian centroid/re-centering：在 Poincare codebook 与现有 assignment 结构中引入 Lorentz 原型中心化，再映回 Poincare。Stage3 仍降至 `0.0540225`，因此“仅稳定 codebook 原型中心”没有解决下游离散 token 的有效语义问题。

从 iter4 审计可归纳出三个需要检索的具体根因，而不是把失败归因于一般性的“双曲模型无效”：

1. **原型中心稳定不等于 residual-SID 语义稳定。** centroid 只改变 codeword 的几何位置，未直接约束逐层 residual 量化所形成的 token 组合；Poincare 距离、Sinkhorn assignment、M2/M3 residual/transport 与 hard `argmax` 之间仍可能存在决策边界和层间语义失配。
2. **Stage2 描述性多样性与 Stage3 生成检索性能脱钩。** iter4 的 Stage2 日志中 `full_gini=0.12050945102632496`、`l01_unique_pairs=11818`、`h_l1_given_l0=5.1386859079049625`，但 Stage3 仍失败；故不能把利用率、碰撞率或条件熵单独视为语义有效性的证据。
3. **RQ 层中间 token 的结构化语义可能失真。** residual quantization 的中间码本可能集中或形成不利的层级分布；只做 centroid recentering 不处理 codebook attention、层次关系、离散边界或推荐相关 item-neighborhood 信息。

## 2. 检索策略与查询边界

查询均围绕“Lorentz centroid/re-centering 失败后，如何改善双曲 residual quantization 的离散决策、层次码本结构或生成式推荐 token 语义”，没有使用泛化的 `hyperbolic recommender` 查询作为唯一依据。

### Query Q1：双曲 residual quantization 与层次离散表示

`2023 2024 hyperbolic residual quantization hierarchical discrete representations codebook`

Top-3 命中：

1. **Hyperbolic Residual Quantization: Discrete Representations for Data with Latent Hierarchies**，2025（超出 2023+ 时间窗口但作为直接机制线索单独标记）。提出在双曲流形中以双曲运算和距离进行 residual quantization；摘要报告 WordNet 超词树的监督层次建模与层次发现，不报告推荐实验。[arXiv](https://arxiv.org/abs/2505.12404)
2. **HiHPQ: Hierarchical Hyperbolic Product Quantization for Unsupervised Image Retrieval**，AAAI 2024。提出双曲 product quantizer、双曲 codebook attention，并在双曲 product manifold 上学习量化表示。[arXiv](https://arxiv.org/abs/2401.07212)；[AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/28261)
3. **HyperVQ: MLR-based Vector Quantization in Hyperbolic Space**，2024。将 VQ 建模为双曲 multinomial logistic regression，使用双曲决策超平面表示 codeword。[arXiv](https://arxiv.org/abs/2403.13015)

### Query Q2：RQ-SID 中间码本集中与生成检索

`2024 residual quantization intermediate codebook concentration hourglass generative retrieval semantic IDs`

Top-3 命中：

1. **Breaking the Hourglass Phenomenon of Residual Quantization: Enhancing the Upper Bound of Generative Retrieval**，2024。指出 RQ-SID 中间层 token 过度集中，造成码本利用不足并限制生成式检索；数据稀疏和长尾分布与该现象相关。[arXiv](https://arxiv.org/abs/2407.21488)
2. **Recommender Systems with Generative Retrieval**，2023。奠定 RQ-VAE/Semantic-ID 生成式检索管线，涉及 residual quantization、codebook collapse 与离散 item 标识。[arXiv](https://arxiv.org/abs/2305.05065)
3. **CoST: Contrastive Quantization based Semantic Tokenization for Generative Recommendation**，2024。指出仅依赖 reconstruction 的 RQ-VAE tokenization 难以捕捉 item-neighborhood 关系，并以 contrastive quantization 同时利用 item relation 与 semantic information。[arXiv](https://arxiv.org/abs/2404.14774)

### Query Q3：双曲量化决策边界与 codebook 可分性

`2024 hyperbolic vector quantization decision boundary codebook collapse separability MLR`

Top-3 命中：

1. **HyperVQ: MLR-based Vector Quantization in Hyperbolic Space**，2024。摘要描述双曲 MLR 量化、双曲决策超平面、降低 codebook collapse 与增强 cluster separability。[arXiv](https://arxiv.org/abs/2403.13015)
2. **HiHPQ: Hierarchical Hyperbolic Product Quantization for Unsupervised Image Retrieval**，2024。以双曲码本 attention 加速量化，并保留多层级语义关系。[arXiv](https://arxiv.org/abs/2401.07212)
3. **Resizing Codebook of Vector Quantization Without Retraining**，Multimedia Systems 2023。使用 hyperbolic embeddings 改进 VQ codebook 结构并支持无需重训练的 codebook resizing；并非 RQ-VAE 推荐方法。[论文 PDF](https://chywang.github.io/papers/ms2023.pdf)

### Query Q4：Poincare/Lorentz 数值稳定性与原型几何替换

`2023 hyperbolic Poincare Lorentz numerical stability optimization prototype centroid representation`

Top-3 命中：

1. **The Numerical Stability of Hyperbolic Representation Learning**，ICML 2023。指出双曲训练可能出现浮点表示导致的 NaN；在优化角度 Lorentz 模型优于 Poincare ball，但两种模型在表示容量与数值性质上存在差异。[PMLR](https://proceedings.mlr.press/v202/mishne23a.html)
2. **Hyperbolic Representation Learning: Revisiting and Advancing**，ICML 2023。提出 HIE，利用到原点的双曲距离提供层次信息；摘要未给出 Lorentz centroid 算法，因此不把“centroid”归因给该文。[PMLR](https://proceedings.mlr.press/v202/yang23u.html)
3. **Fully Hyperbolic Convolutional Neural Networks for Computer Vision**，ICLR 2024。检索资料涉及全双曲层和 Lorentz 几何运算；本次 PDF 抓取未能可靠解析正文，故仅作为相关标题/方法线索，不据此声称具体 centroid 实证。[论文 PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/d0e83f2f6efb967577a7ec4239edefd5-Paper-Conference.pdf)

## 3. 候选机制 P1：层次双曲 residual quantization（HRQ 式层间残差与距离）

### 机制摘要

将当前逐层 residual 的欧氏式组合/局部 Poincare 处理改成明确的双曲 residual quantization：每层在同一曲率标定下计算双曲 residual，使用双曲距离选择下一层 codeword，并将残差保留在对应双曲流形坐标中；层间 token 组合保持原有固定 SID 长度和 codebook 容量。候选只取“流形内 residual + distance”的机制，不引入额外推荐 loss、扩大 codebook 或修改 Stage3。

### 为何可能针对 iter4 失败根因

iter4 只重定位 codeword 中心，却没有改变 M2/M3 后 residual 的表示方式；若 residual 在 Poincare/Lorentz 转换和原点参考之间产生几何失配，稳定的 centroid 仍可能生成无效的层间 SID。HRQ 式每层流形 residual 直接作用于“原型中心稳定但 token 组合失真”的链路，并把层次离散结构与双曲层次数据的几何距离绑定。2025 HRQ 的 WordNet 层次离散实验提供机制相关证据，但没有推荐实验，不能据此宣称 Amazon Instruments 下游收益。

### 曲率相关关键词

`hyperbolic`, `manifold`, `curvature`, `geodesic distance`, `Riemannian`, `hyperbolic residual`, `hierarchical discrete representation`。

### 风险

1. 当前 M2/M3 已有 residual 与跨曲率 transport；再改 residual 几何可能重复改变现有 pipeline 的参考点，造成层间 SID 大幅漂移。
2. 2025 直接命中文献超出用户要求的 2023+ 范围虽仍满足“2023+”，但缺少 2023–2024 同任务验证。
3. 双曲 residual 的数值尺度与 `c(t)` 联动，可能放大 `artanh`/boundary saturation 或使后层有效输入分布改变。
4. 即便层次结构更清晰，也未必提高 T5 对 SID 的可生成性；推荐方向只能由后续实验判断。

## 4. 候选机制 P2：HiHPQ 式固定容量层次双曲 product-codebook attention

### 机制摘要

在不改变 `[256,256,256,1]` codebook 容量、SID 长度和 Stage3 接口的前提下，为每个 RQ 层加入双曲 codebook attention，并在固定维度内组织 product-manifold 的层次响应；训练期用双曲相似度形成 codeword 权重，导出时保持确定性 hard ID。候选不引入 CoST 的 contrastive loss，也不把 product quantization 解释为扩容。

### 为何可能针对 iter4 失败根因

centroid recentering 只移动单个 codeword 的位置，无法表达“粗粒度 token 与细粒度 residual token 的条件关系”。HiHPQ 的层次 product quantization/codebook attention 直接针对中间 RQ 层语义集中与组合失配：让 codeword 响应在层次结构中有显式权重，可能缓解 iter4 中 Stage2 diversity 指标不低但 Stage3 仍不能利用 token 组合的现象。该候选的文献实验来自无监督图像检索，不是推荐数据集。

### 曲率相关关键词

`hyperbolic product manifold`, `hyperbolic codebook attention`, `hierarchical semantic similarity`, `manifold`, `geometric quantization`, `curvature`。

### 风险

1. attention、Sinkhorn 和 hard `argmax` 可能产生多个不一致 assignment 定义，训练期 soft 权重与导出期 SID 存在 gap。
2. product 分块如果改变有效 codeword 组合数，会破坏固定词表/词频协议；必须保持容量不变，否则迭代不可比较。
3. M2 的整体 intrinsic residual 与 product 分块不天然兼容，可能形成跨块坐标和 residual 参考点失配。
4. HiHPQ 的层次语义/量化对比学习信号来自图像任务；去掉额外 contrastive 监督后，attention 本身是否能改善推荐 token 语义没有直接证据。

## 5. 候选机制 P3：HyperVQ 式双曲 MLR 码字决策边界

### 机制摘要

把当前“Poincare distance → detached Sinkhorn → hard `argmax`”的主码字决策改为双曲 MLR 决策：将 codeword 表示为双曲决策结构参数，以双曲 logits 形成 codeword 后验，并以确定性最大后验输出原有 SID。候选不增加 codebook 数量或 token 长度，且不叠加新的推荐邻域损失。

### 为何可能针对 iter4 失败根因

iter4 只修正 prototype center，仍可能保留“训练期 assignment 权重与最终 hard SID 边界脱钩”的问题。HyperVQ 的双曲决策超平面把 codeword 可分性与选择边界直接纳入量化机制，可能比单纯 recentering 更直接地处理 Stage2 几何指标与 Stage3 token 可用性脱钩。其 2024 摘要报告 codebook collapse 缓解与 cluster separability 改善，但未提供本项目 Amazon Instruments 结果。

### 曲率相关关键词

`hyperbolic MLR`, `hyperbolic decision hyperplane`, `manifold`, `curvature`, `distance metric`, `codebook separability`。

### 风险

1. Sinkhorn 的 batch-level balance 与 MLR 的类别边界是两个潜在教师；同时保留两者可能导致 assignment oscillation 或 soft/hard 不一致。
2. 该路线属于 assignment/decision-boundary 改动，可能再次遭遇离散导出与 T5 消费之间的 train/export gap。
3. M2/M3 residual 和动态曲率改变输入尺度后，MLR logits 的曲率标定与温度需要保持一致，否则会产生边界漂移或数值饱和。
4. 机制与 iter3 已失败的 dense STE/codebook decision 路线在接口层面相近，不能把 HyperVQ 的跨任务结果视为本项目下游保证。

## 6. 额外相关证据（用于失败根因定位，不作为第四候选）

- **CoST（2024）**的摘要明确指出 reconstruction-only RQ-VAE tokenization 难以捕捉 item-neighborhood 关系；这支持“iter4 的几何中心改善可能仍未改善推荐语义”的根因假设，但本报告不把 contrastive/neighborhood loss 生成成候选，以遵守机制池边界和单机制约束。[arXiv](https://arxiv.org/abs/2404.14774)
- **Hourglass（2024）**指出 residual quantization 中间 token 过度集中会限制生成式检索；该证据支持检查中间层结构，而非把 Gini/利用率改善直接等同于 Stage3 改善。[arXiv](https://arxiv.org/abs/2407.21488)
- **Mishne et al.（ICML 2023）**支持区分“数值/优化稳定性”与“下游离散语义有效性”：Lorentz 优化性质较好并不意味着 centroid 改动一定改善推荐。[PMLR](https://proceedings.mlr.press/v202/mishne23a.html)

## 7. 统一证据边界

1. 本轮检索至少得到三个候选 P1/P2/P3；编号仅用于独立描述，不构成推荐或排序。
2. 2023+ 文献来自双曲表示、双曲量化、图像检索、生成式检索和生成式推荐等不同任务；没有任何结果证明某候选一定提高 Amazon-2023 Instruments 的 Stage3 `test_R@10`。
3. 所有候选若实施，均需保持固定 SID 长度、codebook 容量、item 顺序和 Stage3 输入协议；本报告不修改 Python 代码，也不运行训练或 gradient check。
