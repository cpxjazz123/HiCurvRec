# 非树语义残差的行为价值

日期：2026-11-17  
范围：离线行为价值验证；不训练 RQ-VAE、不训练混合曲率、不运行 Stage3。

## 1. 目的与修正

上一版用“cosine 减去同一 tree relation 的均值”作为非树特征。该特征与 tree dummies + raw cosine 线性共线，模型预测相同，不能检验独立行为价值。

本版改为**类别残差 embedding**：

1. 对每个 fine category path 计算 item embedding 均值；
2. 每个商品减去所属 fine 类均值；
3. 对残差向量 L2 归一化；
4. 商品对的 non-tree residual = 残差向量的 cosine。

细类只有一个商品时残差为 0；它表示“无法由类别叶节点解释的 embedding 变化”。

## 2. 数据与评估

- 行为正样本：`train` / `valid` / `test` 中 `(history[-1], target)`；测试集完全不参与训练。
- 负样本：从 **同一训练流行度分位** 的商品中抽取，排除该用户 history 与 target；train 3 个负/正，valid/test 5 个负/正。
- 特征：
  - hierarchy：same fine、same coarse；
  - controls：商品训练流行度、行为图度、fine/coarse 类大小、history 长度；
  - raw cosine；
  - non-tree residual。
- 模型：L2 标准化 + logistic regression；评价 AUC、AP、log loss。
- 训练：**1,298,256** 对；验证：**330,252**；测试：**330,390**（55,065 正样本）。

## 3. 结果

| split | model | AUC | AP | log loss |
|---|---|---:|---:|---:|
| valid | hierarchy + controls | 0.6686 | 0.3762 | 0.4302 |
| valid | + raw cosine | 0.6924 | 0.3959 | 0.4240 |
| valid | + non-tree residual | 0.6726 | 0.3841 | 0.4285 |
| valid | + both | 0.6917 | 0.3967 | 0.4240 |
| **test** | **hierarchy + controls** | **0.6561** | **0.3625** | **0.4329** |
| **test** | **+ raw cosine** | **0.6830** | **0.3826** | **0.4268** |
| **test** | **+ non-tree residual** | **0.6609** | **0.3703** | **0.4313** |
| **test** | **+ both** | **0.6823** | **0.3833** | **0.4268** |

按用户的 paired bootstrap（300 次，按 user 重采样）：

| candidate | ΔAUC | 95% CI | 正比例 |
|---|---:|---:|---:|
| + raw cosine | +0.0269 | [+0.0259, +0.0279] | 1.000 |
| + non-tree residual | +0.0048 | [+0.0042, +0.0054] | 1.000 |
| + both | +0.0262 | [+0.0252, +0.0272] | 1.000 |

## 4. 同一 tree relation 内的增量

测试集：

| relation | base AUC | + raw cosine | + non-tree residual | + both |
|---|---:|---:|---:|---:|
| same fine | 0.6047 | 0.6443 | **0.6590** | 0.6565 |
| same coarse, diff fine | 0.6373 | 0.6712 | 0.6469 | 0.6709 |
| cross coarse | 0.5995 | 0.6365 | 0.6051 | 0.6353 |

解释：

- 类别残差在**同 fine**内最有价值（AUC +0.054），说明同一细类里仍有行为相关的语义差异；
- 同 coarse 不同 fine 的增益小（+0.010）；
- cross coarse 的增益更小（+0.006）；
- raw cosine 在三个关系内都更强；
- `+ both` 没有超过 `+ raw cosine`，所以非树残差与原始 cosine 高度重叠，不能声称它是独立于文本相似度的强信号。

## 5. 结论

1. **非树语义残差确实有统计上可检测的行为价值**：test ΔAUC +0.0048，95% CI 不含 0；在同 fine 内增量最大。
2. **但效应很小，且不独立于 raw cosine**：raw cosine +0.0269；加入 raw cosine 后 residual 几乎不再增加（0.6823 vs 0.6830）。
3. 因此目前只能说：**类别叶节点之外还有少量行为相关语义**；不能说“非树关系是主要的、独立的预测来源”。
4. 这仍是支持“单树不充分”的辅助证据，但不足以单独支撑混合曲率机制；需要四个点交叉关系、行为边恢复和 Stage3 共同判断。

## 6. 局限

- pair-level 负采样，不是完整排名或 Stage3 Recall；
- 负样本按训练流行度分位匹配，但可能包含用户潜在喜欢但未交互的商品；
- fine category 均值是线性去层级；非线性残差可能不同；
- Amazon 类目有噪声；
- 没有测 test 与 valid 的模型选择差异；模型结构固定。

## 7. 产物

- `results/research/tiger_cluster_graph/non_tree_behavior_value.py`
- `results/research/non_tree_behavior_value/non_tree_behavior_value.json`
- `results/research/non_tree_behavior_value/non_tree_behavior_value.md`
