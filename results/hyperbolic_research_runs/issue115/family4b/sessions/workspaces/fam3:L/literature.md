# 新机制族：离散 SID 选码位置上的非平凡双曲操作

## 范围与证据口径

检索目标限定为：双曲运算实际参与离散 code/SID 的候选选择或离散表示构造，而不是仅用于连续编码器、最终排序打分，或训练后解释。以下文献只报告已核实的信息；`FULL_TEXT` 表示本轮直接检查了论文全文，`ABSTRACT_ONLY` 表示只检查了摘要，`UNKNOWN` 表示没有检查全文或摘要。未把元数据命中等同于全文核实。

冻结约束仍全部不变：32 维 latent、恰好 3 个 256-entry codebook、每项恰好 3 个整数 SID token、原数据/切分/Stage2/Stage3 recipe；同一机制必须严格超过给定的 Hard-L1 ER、Recall@10、NDCG@10 三个阈值。本文件只提出待审查机制，不声称已满足性能门槛，也不改变协议。

## 已核实的直接相关先例

| 标识；题名；年份 | 检查范围 | 精确重叠及边界 |
|---|---|---|
| arXiv:2505.12404；Piotr Piękos, Subhradeep Kayal, Alexandros Karatzoglou, “Hyperbolic Residual Quantization: Discrete Representations for Data with Latent Hierarchies”；2025 | **ABSTRACT_ONLY**（直接读 arXiv 摘要；未读论文全文） | 明确提出 HRQ：数据嵌入双曲流形，以双曲运算和距离逐级量化 residual，并用于产生多 token 离散表示。故“在多级离散码选择/构造时使用双曲几何”这一宽泛主张已被占据；不能把“把 Euclidean RQ 换成双曲 residual/距离”包装成新机制。本摘要没有确立其是否使用本任务的三层、每层 256 码、固定冻结配方，亦没有建立其操作等同于下面任何具体候选。来源：[arXiv 摘要](https://arxiv.org/abs/2505.12404)。 |
| DOI: [10.1016/j.cviu.2025.104392](https://doi.org/10.1016/j.cviu.2025.104392)；Shangyu Chen et al., “HVQ-VAE: Variational auto-encoder with hyperbolic vector quantization”；2025 | **UNKNOWN**（核实 DOI、题名、年份及期刊元数据；本轮未能取得摘要或全文） | 题名与索引关键词表明其覆盖 hyperbolic vector quantization / 离散 latent codebook，是直接相关的优先查全文文献。由于未检查摘要/正文，不推断它具体如何计算最近码、是否在 Poincaré 球、是否属于 HRQ，也不据此断言下列精细机制已被占据。来源：[Crossref DOI 落地页](https://doi.org/10.1016/j.cviu.2025.104392)；OpenAlex 元数据记录为 2025 年 *Computer Vision and Image Understanding* 258:104392。 |

### 先例覆盖判断

HRQ 已占据“对 residual 做双曲几何量化”这一机制大类；HVQ-VAE 的存在使泛称“hyperbolic VQ”也不能作为新颖性主张。下列方向只有在全文核查确认没有相同运算、并且差异化定义经审查后，才可称为未占据候选；本轮证据不足以作绝对的新颖性结论。明确排除已关闭的 product-geodesic level-0 radius、horospheric/Busemann level gap、transported residual、role-layout、gyro/Möbius associator，以及 unsigned wedge / LOTV tie-break。

## 仍可审查的机制构想（非结论、非已验证新颖性）

四个构想都让双曲量进入实际离散选码，而非仅进入连续前向或打分；每个都须先从冻结分配流量化其触发情形，再决定是否值得实现。这里的数值条款是构想的预注册草案，必须经数学/实现审查后冻结，不能在观察结果后调参。

### A. 前缀条件的双曲 Fréchet 原型选码（只作用于仍有歧义的前缀）

- **操作**：在选择 level-2 token 时，给定已选前缀 `(c0,c1)`，对该前缀下每个 level-2 候选簇内的已有向量按 Poincaré 距离定义 Fréchet 原型；分配时选择使当前向量到该候选原型的距离最小的 token。它不是以 codeword 半径给 level-0 一次性塌缩，也不是整体 geodesic-product assignment；实际候选必须用前缀条件局部统计量改变离散 token。
- **先测发生率**：冻结流中有 16,147 个 `(L0,L1)` 前缀格，4,624 格（28.64%）有至少两个观测到的 code；11,523 格（71.36%）只有一个。该粗粒度发生率上限约束了任何只在多 code 前缀内起作用的机制，不能把它描述为全体样本都会改变。真正能改变分配的比例需再从冻结 latent/code stream 计算“候选距离排序发生变化率”，当前摘要没有该数值，故候选暂不放行。
- **有限精度草案**：仅允许有限 Poincaré 球半径 `||x||₂ ≤ 1−2^-12`，输入和原型坐标先以 FP64 算，最后选码使用 FP64 距离；`atanh` 输入夹到 `[-1+2^-24,1−2^-24]`；平方范数非负项夹到 `[0,1−2^-24]`；Fréchet 迭代固定 32 步、步长 `1/32`，每步投影到半径 `1−2^-12`；比较容差 `τ=2^-40`，距最小值不超过 τ 时取最小整数 code ID；空簇/非有限值视为硬错误并停止，不回退到 Euclidean 距离。边界等号按夹取后的值计算。

### B. 双曲对数映射残差的角向选码（不是 codeword-frame transport）

- **操作**：在每层候选码的选择点，先将当前点和候选原型映射到同一个固定原点的切空间，计算 `Log_o(x)−Log_o(q)`，再以该切空间残差的双曲诱导度量二次型选最近 token。原点固定且不随 codeword 搬运；不计算 transported residual，也不使用 Mobius/gyro 复合。相较 HRQ，必要的新差异是保持固定锚点并以切空间各向异性度量改变 code 的 argmin，而不是双曲 residual 递推；仍需核对 HRQ/HVQ-VAE 全文。
- **先测发生率**：每项在三个 level 都必经一次 code 选择；摘要显示 L0/L1/L2 分别使用 245/256/256 个 distinct codes，因此不是“只在稀有 tie 才触发”。但多少项的候选排序会改变，冻结摘要未给出，必须用实际 frozen latents 和 codebook 无训练地测量后再提案。
- **有限精度草案**：曲率固定 `−1`；所有点限于球半径 `1−2^-12`；`Log_o(x)=atanh(r)x/r` 在 `r<2^-20` 时定义为 `x`，否则按该式算；`atanh` 输入夹至 `1−2^-24`；向量、内积及距离在 FP64 计算；二次型负值仅在 `[-2^-40,0)` 时夹为 0，低于 `−2^-40` 即报错；选码比较容差 `2^-40`，并列取最小 code ID；超界向量直接投影到半径 `1−2^-12`，非有限输入报错。

### C. 由双曲体积失真驱动的局部 codebook 选择

- **操作**：对一个输入与一组有限候选码，构造由固定原点、输入点、候选码及其余切空间方向边界构成的局部双曲 simplex；选取使该 simplex 的双曲体积失真最小的码。目标是利用局部体积而非点到码的半径/距离，且体积必须直接参与 argmin，不能退化为 tie-break。它不是已停放的 signed Lorentzian tetrahedral tie-break 或 unsigned tangent wedge：不在目标近似相等时追加量，而是全候选的主排序准则。不过四点/高维 simplex 的规范化本身仍是数学风险，需预先定义并由评审确认非退化。
- **先测发生率**：每个样本每层都评估候选，操作触发率为 100%；候选排序实际变化率未知。禁止依赖罕见 tie，故须先在 frozen assignment stream 上报告“体积目标相对冻结目标改变 argmin 的比例”和逐层比例；若成本/定义只保留 tie 情况则立即撤回。
- **有限精度草案**：切空间向量 FP64；所有输入限定 `||x||≤1−2^-12`；Gram 行列式用带主元 Cholesky 计算，若最小主元 `≤2^-40` 则该候选体积定义为 `+∞`（所有候选均 `+∞` 时硬错误）；体积值为 `sqrt(max(det(G),0))`，其中仅 `det∈[-2^-40,0)` 可夹零、更小负值报错；每个 level 的总体积按固定的 3-simplex 组合方案、禁止根据数据改顶点；并列容差 `2^-40·max(1,|best|)`，并列取最小整数 code ID。方案及组合顺序必须在读取评测标签前冻结。

### D. 双曲各向异性温度的概率选码（只作为确定性 MAP）

- **操作**：对每个 level 的候选码，以 `−d_H(x,q)^2/T_q` 作为离散 logits，`T_q` 由该 code 局部切空间协方差特征值的截断行列式给出；实际 SID 取该分布的确定性 MAP，不采样。此举让双曲局部体积/各向异性改变 code argmax；它不是排序头上的双曲打分，也不是把 role 几何塞进未使用的前向。
- **先测发生率**：每项、每层均执行 MAP；冻结使用率显示 L0 245/256、L1 256/256、L2 256/256 个 code 有被使用，但这不代表候选排序必然变化。需报告无训练的逐层 MAP 改码率。level-3 有 781 个 distinct codes、最大 cell 占 96.2175%，但协议恰为三 token；不以 level-3 扩展 SID。
- **有限精度草案**：局部协方差样本必须至少 2 个且满定义，否则该 code 的温度估计为硬错误（不设默认温度）；FP64 对称化 `(Σ+Σᵀ)/2`，特征值夹至 `[2^-20,2^20]`；`T_q=clip((det Σ)^(1/d),2^-10,2^10)`，行列式通过 log-eigenvalue 求和；距离有限且输入遵守球半径 `1−2^-12`；softmax 仅用于诊断，MAP 直接比较 logits，容差 `2^-40·max(1,|best|)`，并列最小 code ID；任意 NaN/Inf 报错，不回退为统一温度或欧氏距离。

## 冻结摘要对候选筛选的约束

- 数据量 24,587；冻结 latent 为 `(24587,768)`、`float32`，平均行范数 0.559254。注意：这是 768 维输入 latent 摘要，不可误读为允许改变协议的 32 维模型 latent。
- 三个有效层分别使用 245、256、256 个 distinct codes（每层 codebook 容量均为 256）；L0/L1/L2 cell 中位数分别为 94、89、95；L0/L1/L2 的 99 分位分别为 203、254、132。
- `(L0,L1)` 有多码格占 4,624/16,147 = **28.64%**，单码格占 71.36%。任何只在多码格触发的候选，初始可触及范围最多是上述前缀格的成员；按格数比例不能假装成样本比例，也不能替代逐样本计数。
- ER frozen pair shapes：anchors `(52855,)`、positives `(52855,)`、negatives `(52855,10)`。候选的选码作用率与性能比较不得改变这些输入、原切分或 Stage2/Stage3 recipe。

## 决策

不推荐将泛化的“hyperbolic residual quantization”作为新家族，它已被 HRQ 摘要直接覆盖；不推荐任何 tie-break-only 机制。优先级上，先取全文核查 HRQ 与 HVQ-VAE，再从 A/B 中选一个进行零训练触发率审计；只有测得足够常见的实际 argmin 改变率、数值定义通过边界审查后，才进入实现。C/D 定义风险或统计要求较高，暂列探索项。以上均不构成对三项严格超阈值结果的承诺。
