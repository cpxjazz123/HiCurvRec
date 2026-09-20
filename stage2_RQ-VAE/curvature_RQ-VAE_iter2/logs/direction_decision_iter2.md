# iter2 方向评判：P1/P2/P3

- 评判角色：Agent B direction-judge
- 评判日期：2026-09-21
- 唯一依据：`lit_search_iter2.md`、iter1 失败事实、当前机制池与实现边界
- 明确限制：本评判未进行外部搜索；不修改训练代码或 Stage3，不启动训练。

## 1. 评判前提与失败根因

iter1 的双曲 MLR/决策边界 assignment 虽然通过梯度通路检查，但训练约 20k 步后出现 L0/L1/L2 usage=1/1/1，30k raw SID 只有 1/24587 个唯一组合，Stage3 `test_R@10=0.0158603`。因此本轮首先要解决的是“少数 codevector 持续获选、未获选 codevector 得不到有效更新，并形成层间自强化塌缩”，而不是单纯增加一个新的可学习 assignment 参数。

评分含义：每项 1--5 分，分数越高越好；(d) 是“回归风险越低分越高”。

## 2. 完整评分表

| 候选 | (a) 直接修复 collapse 根因 | (b) 2023+ 文献实证匹配度 | (c) 与 cyclic c(t) 兼容性 | (d) Stage1 ceiling/regression 风险 | (e) 曲率/双曲/流形合规性 | 总分 |
|---|---:|---:|---:|---:|---:|---:|
| P1 固定近邻 assignment + 双曲残差更新 | 4/5 | 4/5 | 4/5 | 2/5 | 5/5 | 19/25 |
| P2 双曲 dead-code online clustering | 5/5 | 3/5 | 4/5 | 4/5 | 5/5 | 21/25 |
| P3 条件化的双曲残差 codebook | 3/5 | 4/5 | 2/5 | 2/5 | 5/5 | 16/25 |

## 3. 分项理由

### P1：固定近邻 assignment + 双曲残差更新（BACKUP）

- **(a) 4/5：** 固定几何近邻 assignment 能移除 MLR 决策边界与 codevector 同时竞争造成的赢家通吃，Möbius-native residual update 也能使后续残差留在同一双曲几何中；但它没有直接保证 inactive codevector 重新获得训练信号，所以对已经形成的 1/1/1 状态仍不是最直接的再激活方案。
- **(b) 4/5：** `lit_search_iter2.md` 中的 HyperVQ（2024）、HVQ-VAE（2025）和 HRQ（2025）均提供双曲 VQ、双曲 codebook 优化或双曲 residual quantization 的近邻机制依据；但这些命中并非本 Amazon RQ-VAE 设置下的实证。
- **(c) 4/5：** 固定近邻规则原则上可使用当前 cyclic `c(t)` 的双曲距离，和曲率周期变化没有显式目标冲突；不过 assignment 规则与残差更新同时变化，仍可能对曲率周期产生较强敏感性。
- **(d) 2/5：** 这是对 Stage2 量化表征和 SID 分布的较大结构变更，可能造成 Stage3 regression；并且它仍处在与已失败 MLR assignment 相邻的 assignment 路线。**WARNING：不得原样复用 iter1 的 MLR/决策边界实现；即使改为固定近邻，也必须把它视为高回归风险 BACKUP，不能因文献匹配而跳过梯度检查。**
- **(e) 5/5：** 直接命中双曲距离、Poincaré-ball codebook 与 Möbius 残差更新，满足曲率/双曲/流形 novelty 要求。

### P2：双曲 dead-code online clustering（唯一推荐）

- **(a) 5/5：** 它直接针对“未被选中的 codevector 永远没有有效更新”这一根因：在窗口内识别 inactive entry，用当前 encoder/residual feature 作为 anchor 重新激活，从而切断 1/1/1 的自强化循环。它不依赖增加 MLR 决策自由度，也不把已有 assignment 梯度误认为 codebook 使用均衡。
- **(b) 3/5：** `lit_search_iter2.md` 给出的 Online Clustered Codebook（ICCV 2023）对 inactive/dead codevector 的在线 anchor 更新有直接实证；但该文献本身主要是欧氏 VQ，检索材料没有给出“Poincaré dead-code reactivation”在本任务上的直接实证。因此匹配度良好但不是满分，流形化改写仍需实测验证。
- **(c) 4/5：** assignment 规则保持不变，主要新增 inactive-entry 的再激活，因此比 P1/P3 少一个与 cyclic `c(t)` 竞争的主路径；只要 anchor 到 codebook 的更新严格使用当前曲率下的合法流形映射/流形均值，就能与现有周期曲率并存。当前 `c(t)` 变化仍可能使再激活阈值和更新幅度敏感，故不评 5 分。
- **(d) 4/5：** 不修改 Stage1 embedding，也不引入 Stage3 或协同信号；相对重写 assignment 或残差结构，Stage1 ceiling 和 downstream SID regression 风险最低。但任何改变 codebook 使用分布的 Stage2 机制仍可能让 Stage3 表征退化，不能把低风险误认为无风险。
- **(e) 5/5：** 候选明确要求在 Poincaré ball 中通过合法流形映射或流形均值更新 codevector；曲率 `c` 直接参与几何更新，满足双曲/流形机制合规性。实现时不能退化为欧氏 anchor 替换，否则将不再满足本轮唯一 novelty 约束。

### P3：条件化的双曲残差 codebook（BACKUP）

- **(a) 3/5：** 让后续 codebook 依赖此前向双曲近似，理论上可使每个残差阶段匹配自己的局部分布并减轻层间级联塌缩；但它没有直接处理已死 codevector，且错误的前层选择可能把 collapse 条件化地传播到后层，修复路径间接。
- **(b) 4/5：** QINCo（ICML 2024）提供前向近似条件化残差 codebook 的实证，HRQ（2025）提供双曲/Möbius residual quantization 依据；但检索材料也明确 QINCo 主要报告量化/搜索性能，未直接证明可修复本项目的 1/1/1 collapse。
- **(c) 2/5：** 条件化 codebook 使每层 codebook 同时受前层近似与 cyclic 曲率影响，形成移动的条件分布和几何尺度；这与当前 `c(t)` 叠加时最容易出现训练不稳定或层间级联，兼容性低于 P1/P2。
- **(d) 2/5：** 这是较大的 residual/SID 结构改动，既有 codebook 分布被重写，又可能放大前层错误向后层传播，Stage3 regression 风险高。**WARNING：属于高风险结构路线，在没有先证明 codebook 使用恢复前不应优先实现。**
- **(e) 5/5：** 候选同时使用双曲距离、Möbius residual 和条件化 codebook，明确命中双曲/流形机制要求。

## 4. 唯一推荐

### 推荐：P2 双曲 dead-code online clustering

推荐理由：

1. 它最直接命中 iter1 的根因——inactive codevector 没有有效更新，而不是再次增加 assignment 决策自由度。
2. 它保持现有 assignment 和 cyclic `c(t)` 主路径不变，结构扰动与 Stage1/Stage3 回归风险相对最小。
3. 它既有 2023 年 Online Clustered Codebook 的再激活实证依据，又可在实现中明确保持 Poincaré 流形映射/流形均值，从而满足本轮唯一曲率/双曲 novelty。

P1 仅列为 **BACKUP**，且不得原样复用已失败的 MLR/决策边界；P3 仅列为 **BACKUP**，因条件化残差与 cyclic 曲率叠加的稳定性和回归风险较高。除 P2 外，本轮不建议并行或叠加其他机制。

## 5. 实施边界与最终裁决

- 只允许实现 P2 一个机制；不得叠加 P1、P3、协同信号或 Stage3 改动。
- dead-code 的重新激活必须是当前曲率下的合法双曲流形操作；禁止用欧氏更新冒充双曲机制。
- 在任何训练前必须完成 1 个 ckpt + 1 个 batch 的梯度通路检查，并检查新增再激活相关路径没有 `detach()` 截断；检查失败则直接 NO-GO，不启动训练。
- 本文件只完成方向评判，未实现、未训练、未调用 Stage3；本文件没有 Stage3 结果，不能宣称已达到 `test_R@10 > 0.065`。

**GO 仅限实现与梯度通路检查**
