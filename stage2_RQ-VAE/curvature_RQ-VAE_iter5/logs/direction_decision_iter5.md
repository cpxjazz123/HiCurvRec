# iter5 方向裁决报告（Agent B）

- **数据集**：Amazon-2023 Instruments
- **迭代**：iter5
- **当前迭代失败回顾**：iter4 `iter4_lorentzian_centroid_recenter` 的 Stage3 `test_R@10=0.0540225`，低于 `0.065` 硬目标，NO-GO。
- **Agent A 报告**：`logs/lit_search_iter5.md`，提出 P1 / P2 / P3 三个机制；编号仅用于独立描述，不构成推荐。
- **机制池基线**：`/home/wlia0047/ar57/wenyu/GeneRec/.curvature-rqvae-iter-skill/references/mechanism_pool.md`。当前唯一 promoted baseline 为 v318 cyclic-c（stage1 端最优 `test_R@10=0.0602` 为 iter11 `sk_eps=0.5`，baseline 0.0534，硬目标 0.065）。
- **裁决日期**：2026-09-21
- **裁决角色**：Agent B（direction-judge），仅基于 Agent A 报告与既有证据做方向合理性评判；不进行 web 检索、不修改 Python 代码。

## 1. 曲率合规硬资格检查

| 候选 | 命中曲率机制类型 | 标题/摘要关键词 | 结论 |
|---|---|---|---|
| P1 层次双曲 residual quantization（HRQ） | geometric transform（residual 几何重写）+ hyperbolic loss（hyperbolic residual） | hyperbolic, manifold, geodesic distance, Riemannian, curvature | PASS |
| P2 层次双曲 product-codebook attention（HiHPQ） | geometric transform（hyperbolic codebook attention）+ dynamic curvature（per-layer product 曲率） | hyperbolic product manifold, hyperbolic codebook attention, manifold, curvature | PASS |
| P3 双曲 MLR 决策边界（HyperVQ） | geometric transform（hyperbolic decision hyperplane）+ hyperbolic loss（hyperbolic MLR logits） | hyperbolic MLR, hyperbolic decision hyperplane, manifold, curvature, distance metric | PASS |

P1 / P2 / P3 均满足"dynamic/per-layer/per-item/per-codebook curvature、manifold replacement、Riemannian optimizer、geometric transform、hyperbolic loss 至少之一"，且标题/摘要均出现 geometric/hyperbolic/manifold/Riemannian/curvature/distance metric 至少一词。三者均进入评分。

## 2. 评分维度

- (a) 对 iter4 Lorentz centroid recenter 失败根因的明确修复路径
- (b) 2023+ 文献证据强度
- (c) 与 Poincare+Sinkhorn+M2/M3+cyclic baseline 的兼容性
- (d) Stage1 端纯曲率 ceiling 风险（5 = 低风险）

满分 5 分；总分 = (a)+(b)+(c)+(d)。

## 3. 评分表

| 候选 | (a) 修复路径 | (b) 文献证据 | (c) 兼容性 | (d) ceiling 风险 (5=低) | 总分 |
|---|---|---|---|---|---|
| P1 HRQ 层次双曲 residual | 3.5 | 2.0 | 2.0 | 2.0 | **9.5** |
| **P2 HiHPQ 层次双曲 product-codebook attention** | **4.0** | **3.0** | **2.5** | **2.0** | **11.5** |
| P3 HyperVQ 双曲 MLR 决策边界 | 4.0 | 3.5 | 1.5 | 2.0 | **11.0** |

### 3.1 评分依据

**(a) 修复路径针对性**

- P1：直接命中"原型中心稳定 ≠ residual-SID 语义稳定"与"RQ 中间 token 结构化语义失真"两条根因；但本质上仍属"逐层 residual 几何改写"，与 iter4 Lorentz centroid recentering 同属 geometric-transform 一脉，方向上有与 iter4 同步退化的风险。
- P2：直接命中"RQ 中间层 token 结构化语义失真"（Hourglass 现象）+ 间接补强"Stage2 多样性与 Stage3 性能脱钩"（codebook attention 让 codeword 在层次结构中显式响应，导出期 hard ID 不变）。它填补了 centroid-only 路线未触及的"codeword 间响应"维度。
- P3：直接命中"训练期 assignment 与最终 hard SID 边界脱钩"，但与 iter3 已失败的 dense STE / codebook decision 路线接口相近，需承担更高回归风险。

**(b) 2023+ 文献证据强度**

- P1：唯一直接命中文献是 2025 年 arXiv:2505.12404（Hyperbolic Residual Quantization），满足"2023+"时间窗口但缺少 2023–2024 同任务验证，且实验为 WordNet 监督层次建模而非推荐；证据最薄。
- P2：AAAI 2024 HiHPQ（arXiv:2401.07212）同行评议，方法学为 unsupervised image retrieval；虽与推荐任务不重合，但 product manifold + codebook attention 的机制描述完整、可作为直接机制线索。
- P3：2024 arXiv:2403.13015（HyperVQ）直接论证"hyperbolic decision hyperplane + codebook collapse 缓解 + cluster separability 改善"，机制描述与本轮失败根因契合度最高，但实验为 VQ 而非 RQ-VAE 推荐。

**(c) 与 Poincare+Sinkhorn+M2/M3+cyclic baseline 的兼容性**

- 基线栈必须保留：Poincare distance（决策距离）+ Sinkhorn assignment（软分配）+ M2/M3 intrinsic residual（层间残差）+ cyclic c(t) 调度 + hard argmax SID + 固定 `[256,256,256,1]` codebook 容量 + 固定 SID 长度 + Stage3 输入协议不变。
- P1：residual 几何重写直接改动 M2/M3 后处理的参考点，可能与跨曲率 transport、cyclic c(t) 形成双重几何标定漂移；iter4 的失败已证明"只稳定几何中心"不足以驱动下游，本路线在 residual 端再做同类改写风险叠加。
- P2：在现有 codeword 表示之上叠加 attention 权重（soft re-weighting），不替换 Poincare 距离选择，不替换 Sinkhorn 平衡，不替换 hard argmax 导出；与 M2 intrinsic residual 不天然冲突（attention 作用于 codeword logits，residual 作用于 selected codeword 之后的差向量）；主要兼容风险在于"soft attention vs hard ID"导出 gap 与 product 分块与 M2 整体 intrinsic residual 的接口差异。
- P3：直接把 "Poincare distance → Sinkhorn → hard argmax" 三段替换为"hyperbolic MLR logits → MAP"，与现有 Sinkhorn batch-level balance 形成两个潜在教师；iter3 已失败 dense STE / codebook decision 路线（v325/v326/v341 等），本路线接口相似，回归风险最高。

**(d) Stage1 端纯曲率 ceiling 风险**

- 现状：stage1 端纯曲率路线历史 ceiling 锁在 iter11 `test_R@10=0.0602`（sk_eps=0.5 主导，非曲率机制本身），TIGER baseline 0.0534，iter4 跌至 0.0540，已知 92+ 次连续 R37/R36* 失败，5 个 mechanism 全部 oracle ≈ baseline ceiling；MEMORY 多条 R36h / R36p / R36n FAIL 锁定（v317/v318/v319/v320/v321/v322/v323/v325/v326/v327/v329/v330/v331/v332/v333/v335/v336/v341/v342/v343/v344/v345/v346/v359/v362/v365/v367 等）。
- 三者均为 stage1 端纯曲率类改动，无一具备"已观察到的下游 > 0.065"实证证据，故 ceiling 风险都偏高，统一打 2.0（5=低风险）；三者均不允许在无 gradient check + descriptive metrics 全 PASS + iter11 baseline 无 PROVOKED REGRESSION 的前提下启动 GPU 训练。

## 4. 唯一推荐机制

**P2：HiHPQ 式固定容量层次双曲 product-codebook attention**

理由（三行）：

1. AAAI 2024 同行评议直接命中 iter4 未触及的"中间 RQ 层 token 结构化语义失真"根因（Hourglass 现象），在 codeword 间补上显式层次响应，与 centroid-only 路线形成正交补强。
2. 不替换 Poincare distance / Sinkhorn / hard argmax / M2 intrinsic residual / cyclic c(t) 任一环节，仅在 codeword logits 上叠加 attention 重加权并保 determinstic hard ID 导出，与 baseline 栈兼容性最高。
3. 规避 P1 "geometric residual 改写 = iter4 同型退化"与 P3 "iter3 dense STE / codebook decision 路线回归"两个已观察失败模式，机制池中尚无 codebook attention 记录，novelty 与回归风险综合最低。

## 5. 实施边界（不得违反）

1. **机制池 anti-pattern 红线**：仅引入 P2 单一 novelty，禁止与其它机制叠加（不混 sinkhorn epsilon sweep、不混 Riemannian Adam、不混 adaptive margin、不混 cyclic T 变体、不混 layer-wise learned c、不混 V-shape curriculum、不混 contrastive / neighborhood loss / EMA codebook）。
2. **容量红线**：保持 `[256,256,256,1]` codebook 容量、`hidden=[512,256,128]`、`embed=32` 不变；不扩容、不修改 Stage1 输入 `item_emb.npy`。
3. **接口红线**：保持 Poincare 距离决策 + Sinkhorn assignment + hard argmax SID 三段链路原样；attention 仅作为 codeword logits 上的 re-weighting 因子，导出 SID 必须为确定性 hard ID（不允许 soft ID 进入 Stage3）。
4. **Stage3 红线**：Stage3 trainer / SID 文件协议 / 输入长度 / 词表大小均不变；不允许把 attention 输出长度传入 Stage3。
5. **曲率红线**：attention 内部距离必须与 cyclic c(t) 联动标定（attention 权重计算时使用当前 step 的 c(t)）；不允许冻结 c。
6. **训练前红线**：实施完毕必须先常驻运行 `scripts/grad_check.py`，逐项验证 `total_loss.requires_grad`、`total_loss.grad_fn`、encoder/codebook/Sinkhorn/attention 路径上至少 1 个参数非零梯度、cyclic c(t) 联动 step `(0,25000)` 的 loss 有变化、attention 路径未出现 `_last_*.detach()` 截断导致的 silent no-op；任一 FAIL 立即 R50 修复后重验，不得启动 GPU 训练。
7. **评估红线**：实施完成后 Stage2 全部描述性指标（3-token SID Gini、每层 mean Gini、collision rate、l01_unique_pairs、H(L1|L0)）只记录不 gate；裁决仍以 Stage3 完整运行后 `test_R@10 > 0.065` 为唯一 PROMOTE 判据（与 `CLAUDE.md` §2 一致）。
8. **回归红线**：若 Stage3 test_R@10 跌至 iter11 0.0602 以下或 iter4 0.0540 附近，必须按 R37 / R38 规则回滚，不得保留 promoted baseline。
9. **文档红线**：本报告为唯一裁决文件，不得在 Stage2/Stage3 之外另写 README / summary / findings .md。

## 6. 文档归档

本裁决文件与 Agent A 报告、关键 Stage2/Stage3 日志在 commit 历史中保留；iter5 训练目录按 NO-GO 流程归档清理的规则不适用于本裁决文件本身。