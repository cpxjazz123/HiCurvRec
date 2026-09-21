# iter4 方向评判

## 评判范围与证据边界

本评判只读取并评判 Agent A 的 `logs/lit_search_iter4.md`、机制池
`/home/wlia0047/ar57/wenyu/GeneRec/.curvature-rqvae-iter-skill/references/mechanism_pool.md`，以及 iter3 的失败审计记录。iter3 的 `gate_decision_iter3.md` 已从提交 `127a75c` 读取；其中唯一硬裁决为 Stage3 150 epoch `test_R@10=0.05247305837497171`，且 dense hyperbolic STE/codebook reparameterization 的 Stage2 梯度检查通过但下游仍失败。iter3 审计还记录了 `full_gini=0.050989196559802774`、`l01_unique_pairs=15570` 和 `h_l1_given_l0=5.645681923346339`，因此不能把利用率或碰撞改善当作推荐语义改善的充分条件。

评分为 1--5 分；(a)--(c) 分数越高越好，(d) 为 Stage1 端纯曲率 ceiling 风险，分数越高表示风险越低。四维总分为 (a)--(d) 之和。曲率合规是单独的硬资格检查，不并入四维总分：机制必须命中 dynamic/per-layer/per-item/per-codebook curvature、manifold replacement、Riemannian optimizer、geometric transform、hyperbolic loss 中至少一类，并且 Agent A 证据中出现 geometric/hyperbolic/manifold/Riemannian/curvature/distance metric 关键词。

## 评分表

| 候选 | (a) 明确修复 iter3 根因 | (b) 2023+ 文献证据 | (c) 与 Poincare+Sinkhorn+M2/M3 及 cyclic baseline 兼容性 | (d) Stage1 纯曲率 ceiling 风险（5=低风险） | 四维总分 | 曲率合规 | 判定 |
|---|---:|---:|---:|---:|---:|---|---|
| P1：HyperVQ 式双曲 MLR 决策量化 | 4 | 4 | 2 | 1 | 11/20 | 通过：hyperbolic / geometric / distance metric | **排除：重复失败的 MLR/assignment 路线** |
| P2：HiHPQ 式层次双曲码本注意力 | 4 | 4 | 2 | 2 | 12/20 | **不通过硬资格**：核心是 attention/层次分块，未以候选定义中的曲率机制作为主体 | **排除** |
| P3：Lorentz 稳定原型中心（closed-form Lorentzian centroid/re-centering） | 4 | 3 | 3 | 2 | 12/20 | 通过：Lorentz/manifold/geometric transform/hyperbolic/distance metric | **唯一推荐** |

## 候选逐项评判与排除理由

### P1：排除

P1 明确把码字选择改成双曲 MLR 决策边界，虽然能针对“训练时梯度存在但 hard `argmax` 决策边界未被直接优化”的失配，但它本质上仍是 MLR/assignment 主路径。项目已知的 P1 MLR/assignment 类路线已有失败记录，不能以 HyperVQ 名称重新包装后重复推荐；此外它还会与 Sinkhorn 的 balanced assignment 形成双教师，并增加 M2 residual 与 M3 跨曲率传输的边界漂移风险。因此即使文献和曲率关键词合规，也必须排除。

### P2：排除

P2 能以训练期软权重缓解边界附近的 hard assignment 抖动，且保留固定 SID 容量；但 Agent A 明确其核心是层次 codebook attention/product 分块，而不是动态或异质曲率、流形替换、Riemannian 优化、几何变换或双曲损失本身。它不满足本轮曲率相关硬资格，不能作为推荐机制。另有 attention 与 Sinkhorn 双重归一化、产品分块与 M2 intrinsic residual 不天然一致等兼容性风险。

### P3：唯一推荐

P3 直接针对 iter3 中“梯度通路通过但原型数值漂移仍可能放大为 hard SID 改变”的根因：在同一 assignment 权重下使用 Lorentzian centroid 对原型中心化，再映回现有 Poincare codebook，从而不改变 SID 接口。其 2023+ 证据（ICML 2023 数值稳定性分析、ICLR 2024 Lorentzian centroid 相关操作）支持稳定性与几何操作，但不是本数据集或 RQ-VAE 的直接下游实证，所以证据评分保守为 3；该机制仍有 Poincare/Lorentz 转换、M2 原点 residual 和 M3 原点 transport 参考点不一致的风险，故兼容性与 ceiling 风险不打高分。

## 唯一推荐理由

推荐 P3，因为它保留现有 Poincare+Sinkhorn+M2/M3/cyclic 接口，仅稳定 codebook 原型的几何中心，避免重做 assignment。
它不重复已失败的 MLR/assignment 路线，也保留明确的 Lorentz/manifold/geometric transform 曲率依据。
推荐不代表能保证超过 `0.065`；必须先验证转换有限性、确定性 SID 与既有梯度/一致性约束。

## 实现边界

1. 只允许实现一个机制：Lorentzian centroid/re-centering；不得同时引入 MLR、attention、额外 loss、Sinkhorn epsilon sweep、学习/调度曲率或新的 optimizer。
2. 保持 `codebook_size=[256,256,256,1]`、SID 长度、item 顺序、数据切分、Stage3 输入协议以及现有 Poincare SID 导出接口不变；不得把 Lorentz centroid 变成整体 Lorentz RQ-VAE 替换。
3. centroid 必须使用与当前 codeword 相同的曲率标定，并通过明确的 Poincare↔Lorentz geometric transform 回映；不得混用未标定的 `c`、距离或坐标归一化。
4. 不改变 M2/M3 residual 与原点 transport 的定义；若原型中心化会改变参考点，必须直接报错而不是静默采用另一套 parallel transport 约定。
5. 不改变 Stage1、Stage3、输入 embedding 或 SID 消费逻辑；只可在 iter4 克隆目录的 Stage2 机制实现中落地。
6. 在任何训练前，仅按既有常驻检查验证有限性、确定性、DDP 一致性及相关参数的非零梯度；本方向评判本身不运行训练或检查脚本。

RECOMMENDATION: Lorentz 稳定原型中心（closed-form Lorentzian centroid/re-centering）
