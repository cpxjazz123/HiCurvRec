# GraphMoRE 官方代码核实(issue #236 C21)

**仓库**:https://github.com/Vic-GoodLuck/GraphMoRE(commit `1948d3c5f19f7251cadd8d1118c06f0e294f96b3`)

## 核查结果:**无法核实论文 Appendix A.1 的绘图参考数与归一化**

| 项 | 结论 |
|---|---|
| 是否存在截面曲率计算函数 | **否** |
| 是否有参考节点 `a` 的采样代码 | **否** |
| 是否有归一化实现 | **否** |

### 依据

仓库共 89 个文件,其中 60 个是数据集二进制文件(`datasets/**`),非数据集源码只有 29 个。
逐个检索 `kappa` / `curvature` / `shortest`:

| 文件 | 行数 | 相关内容 |
|---|---:|---|
| `utils.py` | 45 | 只有 `cal_shortest_dis(edge_index)`:对全图建 `nx.Graph`,返回**每条边两端的最短路距离**。**未组合成 κ(m;b,c)** |
| `models.py` | 217 | `kappaLinear` / `kappaGCNConv`:κ 是**每层的可学习曲率参数 `k`**(κ-GCN),不是截图曲率 |
| `exp.py` | 244 | 无曲率计算 |
| `main.py` | 90 | 无 |
| `data_factory.py` | 158 | 无 |
| `backbone.py` | — | 无 |

`models.py` 的 `compute_distortion` 用 `dis_shortest` 计算**距离失真损失**(每条边的图距离 vs 嵌入距离),
这是 κ-GCN 的训练目标,与 Appendix A.1 的截面曲率公式无关。

### 因此本 issue 的工作是

**按论文 Appendix A.1 的数学定义独立实现 Eq. A.1**,而不是复现官方代码:

$$\kappa(m;b,c) = \frac{1}{|A|}\sum_{a}\left[d(a,m)^2 + \frac{d(b,c)^2}{4} - \frac{d(a,b)^2+d(a,c)^2}{2}\right]$$

| 论文未明确的量 | 本项目的选择 | 是否可核实的 |
|---|---|---|
| 参考节点 `a` 的范围 | **主分量全部 24,556 个** | 论文未给代码,#236 取全量 |
| 归一化 | **无**,报原始值(单位平方跳数) | 论文未给代码 |
| 中点定义 | `d(b,c)=2` 且中心在最短路上 | 论文原文描述 |
| 零容差 | `1e-9` | 自定义 |

**不能声称完全复现 GraphMoRE 的实现。** 本 issue 复现的是论文**公式**,不是其绘图管线。
