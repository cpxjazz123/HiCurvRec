## v3:把多角色目标改到**邻域结构**,它才真的可承载

v2 的结论是“商品自己的 orbit 直方图不可由 768 维语义复现”。v3 改的是**目标**,不是再换一个更大的网络:
item 级监督要的是“这个商品处在什么结构环境里”,不是它自己的稀有 orbit 编号。

```
neighbour_orbit    = 观测邻域的 orbit 分布均值(376)
neighbour_family   = 观测邻域的 family 分布均值(5)
neighbour_relation = 观测邻域的 relation 向量均值(128)
```

### 载体测量(80/20 按商品,常数 profile 只用被覆盖的训练行)

| 目标 | CE | 常数 CE | 相对 | 角色秩相关 | 少数角色商品数 | 判定 |
|---|---:|---:|---:|---:|---:|---|
| relation | — | — | — | — | — | 余弦 **0.8627** |
| neighbour_relation | — | — | — | — | — | 余弦 **0.8444** |
| **neighbour_orbit** | 2.4996 | 2.7235 | **−8.2%** | **0.593** | **3,581** | **可承载** |
| neighbour_family | 0.1728 | 0.2112 | −18.2% | 0.9987 | **1** | 退化,不算多角色 |
| own_orbit | 3.8695 | 3.5004 | +10.5% | 0.212 | 2,249 | 不可承载 |
| own_subtype | 1.6242 | 1.6024 | +1.4% | 0.753 | 1,362 | CE 未过线 |

关键:邻域 orbit 的改善不是“只猜常见类”——3,581 个 eval 商品的真实 argmax 不是全局多数角色,
且预测分布与目标分布的逐商品秩相关 **0.593**。但少数角色的 top-1 仍只有 ~0.03,
所以只声明“形状/主导角色可承载”,不声明逐角色识别。

### 冻结监督(`structure_targets_v3.npz`,24,587 行)

```json
{"relation": 1.0, "neighbour_relation": 0.5, "neighbour_orbit": 0.5,
 "neighbour_family": 0.0, "own_orbit": 0.0, "own_subtype": 0.0}
```

规则先写死:relation/neighbour_relation 需余弦 > 0.8;角色分量需 CE < 常数 × 0.98 **且**
eval 少数角色商品 ≥ 50(防止单一主导类冒充多角色)。

### 识别 API(确定性,随 teacher 交付)

`structure_teacher.recognise_subgraph(matrix)` → `{family, key, orbits}`;
五种 family(K3/P3/C4/P4/K1_3)各有测试。学习式数字仍引用 #221(给邻接 subtype 0.7121 / orbit 0.6101)。

### 状态

- 边存在性预测 **成立**(C−A +0.0395、C−D +0.0343,双 seed);
- 方向 **未成立并声明**(0.5145 vs 多数类 0.5016 / 度数规则 0.5088 / 度数 logistic 0.5039);
- 多角色 **成立于邻域层面**(CE −8.2%、秩相关 0.593、3,581 少数角色商品);
- own orbit/subtype 仅作保真记录,权重 0;
- 不训练 RQ-VAE/Stage2/Stage3,不宣称 SID 或 Recall。

产物:`teacher/structure/v3/`(build SHA `1fd2b39c…`,manifest 含三件套 SHA);
报告 `REPORT_STRUCTURE_TEACHER_V3.md`;测试 `tests/test_structure_teacher_v3.py`(6 项)。
