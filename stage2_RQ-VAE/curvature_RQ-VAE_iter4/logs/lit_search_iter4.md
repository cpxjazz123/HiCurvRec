# GeneRec Amazon-2023 Instruments：iter4 文献检索与候选机制

- **检索日期**：2026-09-21
- **对象**：Amazon-2023 Instruments；当前 `Poincare + Sinkhorn + M2/M3` RQ-VAE 栈
- **本文件性质**：文献候选池，不是实验结论；P1/P2/P3 只是候选编号，不构成“应选哪个”或最优性判断。
- **明确排除**：dead-code clustering、dense STE/codebook reparameterization、cyclic `c(t)`、per-layer/per-item curvature、Sinkhorn epsilon sweep、邻域 loss。以下候选不把这些作为机制主体。

## 1. 当前失败现象与可操作的根因假设

### 1.1 已知实验事实

用户给出的 iter3 结果为：双曲切空间 dense STE/codebook reparameterization 已通过梯度检查并完整训练，但 Stage3（150 epoch）`test_R@10=0.05247305837497171`，低于 `0.065`；iter2 dead-code clustering 为 `0.053186859102700254`。本地结果文件还显示，iter3 的另一份 200 epoch 运行在 `/fs04/ar57/wenyu/GeneRec/results/stage3_T5Train/iter3/logs/Amazon_2023_Instruments/Sep-18-2026_04-47-38/test_final.json` 中为 `test_recall@10=0.05520639286895663`；因此这里把 150 epoch 数值作为用户指定的失败记录，不把不同训练时长的数值混作同一实验。

本地 iter3 Stage2 质量文件
`/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter3/out/rqvae/instruments/quality_final.json`
记录：`full_gini=0.050989196559802774`、`n_unique_full=23288/24587`、`l01_unique_pairs=15570`、`h_l1_given_l0=5.645681923346339`。这说明“只把码本利用率/碰撞改善”不能直接等价为 Stage3 可用的 SID 语义改善。

### 1.2 代码层面可核对的失配点

当前量化实现（参考
`/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE/modules/quantize.py`）的关键路径是：

1. Poincaré 距离计算后，距离先执行 `distances.detach()`；
2. detached 距离进入 Sinkhorn；
3. `ids = argmax(assignments)` 产生离散 SID；
4. 训练梯度主要通过选中的 codeword 的 commitment/codebook loss 与 STE 输出回传；
5. M2 residual 与 M3 跨曲率传输继续改变后续层输入。

因此，iter3 即使“梯度非零、训练能收敛”，仍可能存在以下链路：**训练时可微的切空间/codebook 路径 ≠ 最终 `argmax` SID 的决策边界**；Stage2 描述性利用率改善 ≠ Stage3 T5 可学习的离散语义；M2/M3 产生的残差几何与单一 hard assignment 的边界不一致。候选机制均针对“决策边界/原型几何/数值稳定性/层次语义”的其中一条链路，而不是再增加一个利用率 gate。

## 2. 候选 P1：HyperVQ 式双曲 MLR 决策量化

### 机制

把“由 Poincaré 距离生成 detached Sinkhorn assignment 再 `argmax`”改为**双曲 multinomial logistic regression（MLR）式码字决策**：每个 codeword 作为双曲决策结构的参数，使用双曲几何 logits 形成类别/码字后验；训练时学习的是码字决策边界与潜变量的联合关系，而不只是对已经选中的 codeword 做 commitment。最终仍输出一个离散 SID，保持 Stage3 的词表接口和 SID 长度不变。

候选的最小边界是：不引入 dense STE/codebook reparameterization，不引入额外 codebook，不扫 Sinkhorn epsilon；可以保留现有固定 `sk_eps` 作为对照路径或只把 MLR 作为 assignment head。文献启发来自 MLR 决策量化，不等于照搬论文中的全部损失。

### 为什么精准对应 iter3 失败链路

- **根因 A：assignment 与梯度路径脱钩。** 当前距离在 Sinkhorn 前 `detach()`，而 `argmax` 的离散决策没有对边界提供梯度；dense STE 只能保证数值上的前向/反向通路，并不能保证训练目标优化的是最终 SID 的边界。MLR 将“哪个 codeword 获胜”显式写成双曲分类决策，直接针对该脱钩点。
- **根因 B：利用率与可辨识性不是同一目标。** Sinkhorn 能平衡 assignment 数量，但平衡并不保证 codeword 对消费序列的语义可分；HyperVQ 文献主张以双曲决策结构改善 cluster separability 与 disentanglement，正好对应 iter3 `full_gini` 改善而 `test_R@10` 仍低的现象。
- **根因 C：M2/M3 后续残差会改变局部决策几何。** MLR 的双曲边界可以把 residual 所处的几何位置直接纳入 codeword 选择，而不是只依赖一次距离排序。

### 2023+ 论文与检索证据

- **Nabarun Goswami, Yusuke Mukuta, Tatsuya Harada, “HyperVQ: MLR-based Vector Quantization in Hyperbolic Space”, 2024, arXiv:2403.13015。** [论文链接](https://arxiv.org/abs/2403.13015)
  - 页面摘要/检索证据明确给出：将 VQ formulation 为 hyperbolic MLR；利用双曲空间指数体积增长缓解 codebook collapse、改善 cluster separability；目标包含 compact/structured/disentangled latent representation。
  - 注意：该证据支持“决策量化机制”，不证明它在 GeneRec 或 Amazon Instruments 上会提高 `test_R@10`。
- **检索 query**：`2024 Riemannian vector quantization hyperbolic codebook paper`
  - Top-3：HyperVQ（arXiv:2403.13015）；HiHPQ（arXiv:2401.07212）；`LLMs are Good Action Recognizers`（CVPR 2024，Poincaré-ball codebook）。
- **检索 query**：`2024 hyperbolic vector quantization VQ-VAE manifold codebook`
  - Top-3：HyperVQ；HiHPQ；`Hyperbolic Residual Quantization`（检索结果为 2025，不作为本候选的 2023+ 证据年份）。

### 与现有 Poincare+Sinkhorn+M2/M3 栈的交互风险

1. **与 Sinkhorn 的目标冲突**：Sinkhorn 追求 batch-level balanced assignment，MLR 追求决策边界；两者同时作为主 assignment 可能形成两个不一致的教师。必须明确只保留一个主决策输出，否则会出现 assignment oscillation。
2. **与 M2 residual 的边界漂移**：M2 会把 residual 重新映射；MLR 边界若在原点切空间拟合，跨 M2 后可能不再保持等距，导致层间 ID 语义漂移。
3. **与 M3 transport 的坐标依赖**：M3 传输的是不同曲率下的 residual，MLR 参数若未同步按同一曲率变换，logit 的尺度会随 `c` 变化；这不是再做 cyclic `c(t)`，而是现有动态曲率下的兼容性问题。
4. **Stage3 风险**：训练中 soft posterior 与导出时 hard SID 仍有 train/export gap；必须保持导出路径为确定性单一 `argmax`，否则 T5 词表统计会改变。文献只提供机制证据，不提供该数据集上的 downstream 保证。

## 3. 候选 P2：HiHPQ 式层次双曲码本注意力（固定容量、无邻域 loss）

### 机制

采用 **hyperbolic codebook attention**：对每个 residual 与 codeword 的双曲相似度形成受温度控制的码字权重，并按 RQ 层/固定产品分块表达层次语义；最终在每层仍做确定性离散化，输出原有 `[256,256,256,1]` 规模的 SID，不增加 token 数、hidden dim 或 codebook 总容量。

本候选只取 HiHPQ 的“hyperbolic product quantization + codebook attention + hierarchical semantic quantization”思想，**不采用其 quantized contrastive/neighborhood loss**，以满足本轮排除项。产品分块应理解为现有固定维度内的几何分解，不是扩容。

### 为什么精准对应 iter3 失败链路

- **根因 A：RQ 层的 residual 语义不一定等于全局距离最小。** 现有 `argmax` 是逐层距离选择，M2/M3 后各层输入的语义粒度可能不同；层次码本注意力让不同层的 codeword 响应有显式层次权重，减少“每层都在优化局部几何、但组合 SID 不可被 Stage3 复原”的情况。
- **根因 B：Sinkhorn 平衡只约束计数，不表达层次相似。** iter3 的 `l01_unique_pairs` 与条件熵高于 baseline，但 downstream 仍低，表明 pair diversity 本身未形成有效层次语义。层次 attention 的目标是让同一 coarse component 下的 fine codeword 响应保持结构化，而不是单纯增加 unique pair。
- **根因 C：hard assignment 对边界附近样本不稳。** attention 在训练时可表达多个近邻 codeword 的相对权重，减少单次 Poincaré 距离微小扰动导致的 ID 跳变；导出时仍固定 hard ID，因此不把 dense STE/reparameterization 作为机制。

### 2023+ 论文与检索证据

- **Zexuan Qiu, Jiahong Liu, Yankai Chen, Irwin King, “HiHPQ: Hierarchical Hyperbolic Product Quantization for Unsupervised Image Retrieval”, AAAI 2024, pp. 4614–4622。** [AAAI 论文页](https://ojs.aaai.org/index.php/AAAI/article/view/28261)；[arXiv:2401.07212](https://arxiv.org/abs/2401.07212)
  - 摘要证据：以双曲几何表达 hierarchical semantic similarity；提出 hyperbolic product quantizer、hyperbolic codebook attention，并在 hyperbolic product manifold 上进行量化表示学习。
  - 本文是图像检索，不是推荐系统；这里仅抽取“层次码本注意力”的可迁移机制，不宣称任务等价。
- **检索 query**：`2024 hyperbolic product quantization codebook attention`
  - Top-3：HiHPQ（AAAI 2024）；HiHPQ official implementation；`HyperVQ: MLR-based Vector Quantization in Hyperbolic Space`（2024）。
- **检索 query**：`hyperbolic prototype learning Frechet mean codebook paper 2024`
  - Top-3：HiHPQ；`Fully Hyperbolic Convolutional Neural Networks`（ICLR 2024，Lorentzian centroid）；`Hyperbolic vs Euclidean Embeddings in Few-Shot Learning`（WACV 2024）。

### 与现有 Poincare+Sinkhorn+M2/M3 栈的交互风险

1. **产品分块与 M2 的残差加法不天然兼容**：M2 当前在整体 embed space 做 intrinsic residual；分块 attention 若不定义块间组合，会把一个 residual 拆成不可比的局部坐标，可能破坏 M2 的几何抵消。
2. **attention 与 Sinkhorn 的双重归一化**：attention softmax、Sinkhorn row/column normalization 若同时存在，权重可能过度平坦或过度尖锐；必须保留一个明确的 assignment 温度语义，不能把 epsilon sweep 当作解决方法。
3. **固定容量约束**：若产品分解实际增加 codeword 组合数，会改变 Stage3 的 token 频率和词表分布，不能据此与当前结果直接比较；候选定义要求总容量和 SID 长度固定。
4. **层次语义的外部监督风险**：HiHPQ 使用的图像层次语义/对比学习信号不能直接移植到 Instruments；本候选不引入邻域 loss，因此层次 attention 的语义来源只能来自已有 residual/codebook 结构，效果需要单独验证。
5. **导出确定性**：attention 训练权重必须在导出时折叠成唯一 ID；若把 soft codeword 加权和直接导出，Stage3 的离散词表接口会失效。

## 4. 候选 P3：Lorentz 稳定原型中心（closed-form Lorentzian centroid/re-centering），不整体替换 Poincare 接口

### 机制

保留现有 Poincaré SID 接口、M2/M3 和 codebook 大小，仅在 codebook 原型的几何中心化/更新步骤引入 **Lorentzian centroid**：将 Poincaré 点映射到 Lorentz 模型，在同一 assignment 权重下求闭式加权 Lorentzian centroid，再映回 Poincaré 作为 codeword 的稳定中心。候选不是“把整个模型改成 Lorentz RQ-VAE”，也不是新增 Riemannian Adam；重点是减少原型沿 Poincaré 边界漂移和重复 exp/log map 引入的数值误差。

闭式中心的典型形式是对 Lorentz 点的加权和做 Lorentz 归一化；它可作为 codebook re-centering/更新规则，而不是额外 loss。若实现不能保持现有离散 SID 接口，则该候选的定义不成立。

### 为什么精准对应 iter3 失败链路

- **根因 A：梯度通过不代表原型几何稳定。** iter3 dense STE 已通过梯度检查，但梯度检查只证明 backward 通路存在；它不排除 codeword 在 Poincaré 球边界附近的半径漂移、exp/log 的局部尺度变化和 hard ID 的不稳定。
- **根因 B：Poincaré 数值/优化性质存在已知张力。** 现有代码在 `sqrt(c)`、`atanh`、Möbius add、transport 中反复计算；在 bf16/高维 codebook 下，原型中心的微小数值误差可能被 `argmax` 放大为 SID 改变。Lorentzian centroid 提供稳定的全局中心化坐标，目标是减少“Stage2 几何指标正常但 Stage3 token 语义漂移”。
- **根因 C：避免再次修改曲率调度。** 该候选不改 `c(t)`、不改 per-layer/per-item curvature；它只处理原型更新的模型坐标/中心化稳定性，因而与已经尝试的 curvature sweep 分离。

### 2023+ 论文与检索证据

- **Gal Mishne, Zhengchao Wan, Yusu Wang, Sheng Yang, “The Numerical Stability of Hyperbolic Representation Learning”, ICML 2023, PMLR 202:24925–24949。** [PMLR 论文页](https://proceedings.mlr.press/v202/mishne23a.html)；[arXiv:2211.00181](https://arxiv.org/abs/2211.00181)
  - 摘要证据：分析 Poincaré ball 与 Lorentz model 的数值不稳定；64-bit 下 Poincaré ball 表示容量较大，但 Lorentz model 在 optimization 角度具有优势。
  - 该文支持“优先检查坐标模型的数值/优化稳定性”，不直接证明 centroid 更新适用于本 RQ-VAE。
- **Bdeir, Schwethelm, Landwehr, “Fully Hyperbolic Convolutional Neural Networks for Computer Vision”, ICLR 2024。** [OpenReview 论文入口](https://openreview.net/forum?id=ekz1hN5QNh)；[论文 PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/d0e83f2f6efb967577a7ec4239edefd5-Paper-Conference.pdf)
  - 检索证据显示其使用 closed-form weighted Lorentzian centroid 进行双曲重心/平均池化相关运算；该证据用于支持“闭式 Lorentz 原型中心”这一几何操作。
  - 页面正文抓取受 OpenReview 验证限制，故这里将其标为检索证据而非逐字摘要引用。
- **检索 query**：`2024 Lorentzian centroid hyperbolic neural network closed form centroid`
  - Top-3：`Fully Hyperbolic Convolutional Neural Networks for Computer Vision`（ICLR 2024）；`Lorentzian Residual Neural Networks`（arXiv 2024）；`Lorentzian Graph Convolutional Networks`（基础 centroid 公式，2021）。
- **补充检索 query**：`2023 hyperbolic recommender systems Lorentz Riemannian optimization paper`
  - Top-3：`The Numerical Stability of Hyperbolic Representation Learning`（ICML 2023）；`Lorentz Equivariant Model for Knowledge-Enhanced Hyperbolic Collaborative Filtering`（2023）；`Hyperbolic Neural Collaborative Recommender`（2023）。

### 与现有 Poincare+Sinkhorn+M2/M3 栈的交互风险

1. **模型间距不一致**：Lorentz centroid 是 Lorentz 模型的闭式加权中心，而现有 assignment 使用 Poincaré 距离；映射/回映射的数值误差若超过 hard-ID 间距，可能增加 SID 跳变。必须用同一个 `c` 和严格可逆的坐标转换核对距离，而不是混用未标定的 centroid。
2. **与 M2 intrinsic residual 的语义冲突**：M2 当前以原点 exp/log map 和 Möbius difference 定义 residual；只更新 codebook 中心却不相应处理 residual 参考点，可能使 residual 不再是同一组 geodesic 的量。
3. **与 M3 transport 的参考点冲突**：M3 假设通过原点在不同曲率之间传输；若 centroid 更新引入非原点中心，需要确认 transport 仍以原点为参考，否则会形成两套平行移动约定。
4. **更新时机风险**：每 batch 直接重心更新会使 codebook 非梯度状态更新与 optimizer.step 竞争，导致 reproducibility 和 DDP 一致性问题；候选只说明几何机制，不预设更新频率。
5. **边界饱和风险**：Lorentz 稳定性优势不等于 Poincaré 回映射没有边界饱和；若 centroid 回映射靠近球边界，`atanh`/Möbius 路径仍可能把微小误差放大。

## 5. 候选间的共同验证边界（不是额外候选，也不是推荐）

- 三个候选都必须保持现有 Stage3 输入协议：同一 `codebook_size=[256,256,256,1]`、同一 SID 长度、同一 item 顺序和相同数据切分；否则 downstream 差异不能归因于几何机制。
- 先验证 assignment 的有限性、确定性导出、`loss.backward()` 非零梯度与 DDP 一致性，再进行完整 Stage3；这只是项目既有验证约束，不是候选机制。
- 记录至少：Stage2 的每层使用率/条件熵、SID 字节哈希、Stage3 valid/test `R@10`，并分开报告“几何指标改善”和“推荐指标改变”；不能用某一项 SID 描述性指标替代 Stage3 结果。
- 文献证据来自跨任务方法（图像检索、视觉、自监督或通用双曲表示学习）；没有任何链接在本文件中被解释为 GeneRec Amazon-2023 Instruments 的效果保证。
