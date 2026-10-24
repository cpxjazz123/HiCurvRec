# Role L — hyperbolic code-selection prior art and candidate distinctions

## Scope and evidence labels

本筛查只讨论：非平凡双曲运算是否实际参与离散 SID code 的选择。冻结的 32 维 latent、三个 256-entry codebook、三个整数 SID token、数据/切分及 Stage 2/3 recipe 均不改变。**FULL_TEXT** 表示检查了公开全文；**ABSTRACT_ONLY** 表示只检查摘要/元数据；**UNKNOWN** 表示相关方法细节无法确认。相邻领域的双曲表示学习不等同于离散码选择。

## 可核验 prior art 与精确重叠

| Identifier | 标题 | 年份 | 检查范围 | 与离散 code 选择的精确重叠 |
|---|---|---:|---|---|
| [arXiv:2403.13015](https://arxiv.org/abs/2403.13015) | *HyperVQ: MLR-based Vector Quantization in Hyperbolic Space* | 2024（后续版本 2025） | **FULL_TEXT** | 将 VQ 表述为双曲多项逻辑回归，以 codebook 向量表示双曲决策超平面；双曲决策规则直接选择离散 code。因此占据了“基于双曲 MLR/决策超平面的 VQ 分配”这一具体机制。它不是本冻结的三层 SID 推荐评测。 |
| [arXiv:2505.12404](https://arxiv.org/abs/2505.12404) | *Hyperbolic Residual Quantization: Discrete Representations for Data with Latent Hierarchies* | 2025 | **FULL_TEXT** | 直接研究多级 residual quantization：在双曲空间中进行残差计算、距离度量并逐层选择离散 token。占据双曲 residual quantization 的一般机制范畴；不能把同类逐级残差更新/距离分配改名后当作新家族。其论文任务并非本冻结推荐协议。 |
| [10.1609/aaai.v38i5.28261](https://doi.org/10.1609/aaai.v38i5.28261) | *HiHPQ: Hierarchical Hyperbolic Product Quantization for Unsupervised Image Retrieval* | 2024 | **ABSTRACT_ONLY** | 摘要称其使用双曲 product quantizer 和 “hyperbolic codebook attention mechanism”。这足以确认双曲 product quantization/attention 与 code 选择相邻；未检查全文，故不推断其精确 assignment 方程，也不把更细的机制边界当作已核验事实。 |

以上不是“hyperbolic quantization 尚无人做”的结论：相关方向已有直接工作。标为“未占据”的下列区别仅指上述已检查方法描述，不构成全面新颖性/专利性意见。上述 arXiv 记录含可读全文；HiHPQ 仅核对 DOI 元数据和摘要，未核读全文。

## 仍可区分的机制想法（不是性能承诺）

下列均避开已关闭家族，不是已批准实现。有限精度声明在这里给出可执行的数值域/边界约定；实施前仍须将公式写成测试并由冻结代码验证。所有想法保持原有 codebook 和三枚整数 token。每个想法的“活跃率”需先在冻结 assignment stream 上计算；当前工作区及 issue115 下未找到 `frozen_artifacts.json`，因此不能诚实填报测得数值，不能据此声称其配置常见。缺少该工件时不得推进候选。

### 1. 双曲 Voronoi 单元体积校正的主 assignment 项

对每个现有 code 的双曲 Voronoi cell 预先计算固定体积 `V_k`，在原候选分配分数上加 `λ log(V_k + ε)`，使 cell measure 真正进入 argmin/argmax，而非仅作罕见平局的 tie-break。它区别于 HyperVQ 的 MLR 边界参数化，也区别于普通双曲距离分配；但属于双曲 codebook 邻近方向，需谨慎主张区别。

**数值契约草案：**在曲率 `κ=1` 的 Poincaré 单位球中，仅允许 `||x||₂ ≤ 1−10⁻⁶`；越界点径向投影至该闭球。`V_k` 以固定、公开 seed 的球内 Sobol 样本估计，样本数固定为 `2²⁰`，每点只归属距离最小的 cell；距离以 float64 计算。令 `V_k=max(估计体积,10⁻¹²)`，`λ` 在查看验证结果前固定。有限/非有限值不做静默回退：任一分数非有限即终止并报错。分数绝对差 `≤10⁻¹²` 视作相等，先选冻结基线 winner，再按 code 整数升序。边界点按投影后计算。

**预注册活跃率：**在 artifacts 的每层基线 code usage 上计算 `R_l = 被体积项改变 winner 的 item-level assignment 数 / 该层所有有效 assignment 数`；另报 `V_k` 分布及各 code 计数。若无法从冻结 artifacts 重放每个候选比较，该想法不进入实验。

### 2. 双曲角余量（signed boundary margin）分配

在现有候选 code 的双曲决策边界上，使用带符号的点到边界余量作为主要选择分数，而非 HyperVQ 的 MLR 概率/似然。须明确限定边界族，避免实质退化为其超平面分类器重述。

**数值契约草案：**输入及原型均为 float64，曲率 `κ=1`，定义域 `||x||₂,||c||₂≤1−10⁻⁶`，越界径向投影；所有范数平方小于 `10⁻²⁴` 视为零向量。余量公式的分母以 `max(分母,10⁻¹²)` 截断，符号在精确零时定义为 `0`。任意非有限中间量直接报错。比较 tolerance 为绝对 `10⁻¹²`；差在 tolerance 内时采用冻结基线 winner，再按 code id 升序。边界余量为零时保留零，不人为加符号。

**预注册活跃率：**从 frozen ER pair shapes / latent shape 及分配流重放，报告每层“余量排名与基线 winner 不同”的 assignment 占比，以及 top-1/top-2 余量差 `>10⁻¹²` 的占比。若该量不可由冻结工件计算，停止该候选。

### 3. 跨层双曲测地线一致性作为联合主分数

一次评估完整的现有三 code 组合：对 item 到 L0 code、L0 到 L1 code、L1 到 L2 code 的三段测地距离构造有方向的路径一致性项，并将其作为联合选择目标的一部分；仍输出三个既有整数。它不是“在固定分数的近似平局时才介入”的机制。它与已有 hyperbolic residual quantization 相邻，只有在联合路径项被证明不是逐层残差/距离 assignment 的代数重述时才保留。

**数值契约草案：**Poincaré ball `κ=1`、float64、径向域界 `1−10⁻⁶`；log/exp 映射的范数小于 `10⁻¹²` 按零向量处理，`atanh` 输入夹至 `[-1+10⁻¹²,1−10⁻¹²]`，平方根负的舍入值夹至零。路径项为有限项的有序和；任一中间量非有限则报错终止。联合分数差绝对值 `≤10⁻¹²` 时保留冻结三元组，再按整数三元组字典序；边界投影在每次映射后执行。

**预注册活跃率：**用 artifacts 的 prefix-cell 结构先报告有多个后续 code 的 `(L0,L1)` cell 比例；再报告这些 cell 中联合项改变完整 SID 三元组的比例（分母同时报全部有效 item 与可分叉 cell 内 item）。若多 code prefix-cell 罕见，或联合分数从不改变 winner，则撤回。

### 4. 双曲局部扰动稳定性主选择

对候选 code 的 assignment 分数评估其在固定幅度双曲切向扰动下的最坏情形，选择最稳定的 code。扰动必须影响通常的 argmin，而不是只在 tie 时生效；它不属于已关闭的半径、horospheric level、运输残差或有符号/无符号面积家族。

**数值契约草案：**`κ=1`，输入投影至 `||x||₂≤1−10⁻⁶`，float64；切向方向固定取相邻 frozen pair latent 差在 `x` 处的黎曼投影并归一化，`δ=10⁻⁴`。若方向范数 `<10⁻¹²`，定义扰动为零且标记该样本 inactive（不静默换方向）。用 `exp_x(δv)` 后投影；计算原点及扰动点两处 score，稳定分数取二者较大值。非有限值报错；score 差 `≤10⁻¹²` 判平，先取基线 winner，再按 code id 升序。

**预注册活跃率：**冻结流上报告方向有效率、扰动后 winner 改变率、以及该稳定性分数改变基线 winner 的比例，逐层给出分子/分母。必须先获得这些统计，再判断是否值得实验；不能以几何上可能发生代替频率证据。

## 实施门槛与结果边界

1. 当前找不到所需 `frozen_artifacts.json`，因此本报告没有伪造任何 occurrence rate。获得工件并逐层复算前，以上候选均不应进入训练或 Stage 2/3。
2. 工件可用后先披露分子、分母、比率和数值稳定性/非恒等性检查；由 artifacts 可计算的频率不达预注册门槛或机制实际不改变选择时，撤回候选。
3. 冻结阈值不变。只有同一机制同时严格超过 Hard-L1 ER `16.38953942232631`、Stage3 Recall@10 `0.06406796775709883`、NDCG@10 `0.03530651701940677` 才算成功。本文不声称任一候选可达到该结果。
