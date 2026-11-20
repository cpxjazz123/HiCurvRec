# 冻结结构 Teacher Bundle(可直接加载使用)

日期:2026-11-21
实现:`structure_teacher_bundle.py`;产物:`teacher/structure/bundle/`
源版本:`structure-teacher-v3`(`build_sha 1fd2b39c…`)

## 0. 这个 bundle 是什么

前面几轮交付的是研究产物(目标表、载体、契约)。这一层把它们打包成一个**可加载、可校验、
可直接用于 RQ-VAE 训练损失**的对象:

```python
from structure_teacher_bundle import load_structure_teacher

teacher = load_structure_teacher("results/research/directed_graphlet_teacher/teacher/structure/bundle")
teacher.verify()                                  # 每个 SHA + 载体参数校验
teacher.recognise(matrix)                         # 诱导邻接 -> {family, key, orbits}
teacher.targets_for([12, 900, 24001])             # 图导出的监督行
teacher.predict_targets(embeddings)               # 768 维载体预测(无图回退)
metrics, loss = teacher.structure_loss(student_vectors, role_logits=logits)
```

| 文件 | 内容 | SHA-256(前 12) |
|---|---|---|
| `structure_teacher_bundle.pth` | 载体(CarrierV3)+ 能力块 + 权重 + 源 SHA | 见 manifest |
| `structure_teacher_bundle_contract.json` | API、能力、边界、权重、未成立项 | 见 manifest |
| `structure_teacher_bundle_manifest.json` | 上述文件的大小与 SHA | — |

`verify()` 同时校验源文件(`v3` 的 teacher / targets / contract)的 SHA 与载体参数摘要,
任何一件被替换都会 `healthy=False`。

## 1. 能力矩阵(来自 v3 的证据,不是本层重新声明)

| 能力 | 状态 | 数字 |
|---|---|---|
| 识别(给诱导邻接) | **成立** | 确定性 family/key/orbits;五种 family 有测试;#221 学习式 subtype 0.7121 / orbit 0.6101 |
| 多角色(邻域) | **成立** | `neighbour_orbit` CE −8.2%,逐商品秩相关 0.593,3,581 个少数角色商品 |
| 预测:边存在性 | **成立** | C−A +0.0395 [0.0372,0.0418],C−D +0.0343 [0.0322,0.0363] |
| 预测:方向 | **未成立** | 0.5145 vs 多数类 0.5016 / 度数规则 0.5088 |
| 商品级可承载 | **成立** | relation 0.8627 / neighbour_relation 0.8444 / 邻域 orbit 见上 |
| 监督可交付 | **成立** | 三个损失 + 证据权重;`structure_loss` 端到端可训练 |
| 逐角色识别(少数角色) | **未成立** | 少数角色 top-1 ≈ 0.03 |

## 2. 监督接口

`structure_loss(student_vectors, item_ids=None, role_logits=None)`:

- `student_vectors` 必须是 **(24,587, d)** 的全体商品表示(与目标表同序),pair 表按全局 id 索引;
- relation 项 = `w_relation × (relation_ranking_loss + 0.1 × relation_distill_loss)`
  `+ w_neighbour_relation × relation_distill_loss(neighbour_relation)`(邻居聚合关系几何);把 `w_relation` 置 0 会同时关掉这两项;
- 批内没有任何观测 pair 时,两项关系损失记为 `None`、总损失为可微零,不产生 NaN;
- `student_vectors` 或 `role_logits` 含非有限值时直接 `ValueError`,不把 NaN 传进梯度;
- `role_logits`(可选,376 维)与 `student_vectors` 接受同样的两种形状:全体表或 `len(item_ids)` 行批内对齐;给出时加 `0.5 × orbit_loss`(仅 `coverage_neighbour>0` 的行);
- 返回 `(metrics_dict, total_tensor)`,`metrics` 里逐项记录数值,便于日志。

权重来自 v3 契约,不由本层重新设定:

```json
{"relation": 1.0, "neighbour_relation": 0.5, "neighbour_orbit": 0.5,
 "neighbour_family": 0.0, "own_orbit": 0.0, "own_subtype": 0.0}
```

## 3. 边界(硬)

- RQ-VAE Encoder 输入仍是 **768 维 Stage1 向量**;不含邻接、Teacher hidden、实例标签。
- 冻结 Teacher 只在 **训练损失**中使用;SID 推理协议不变。
- `targets_for` 读的是冻结表;`predict_targets` 只是无图回退,不用于替代图导出目标。
- 本层不训练 RQ-VAE/Stage2/Stage3,不宣称 SID 或 Recall。

## 4. 测试

`tests/test_structure_teacher_bundle.py`(7 项):

1. 全部 SHA 与载体参数校验通过;
2. 五种 family 的识别;
3. 目标行对齐与越界拒绝;
4. 载体冻结且确定性;
5. `structure_loss` 有限,且 Teacher 关系几何的损失低于随机;
6. 一个 4 维线性 student 用 bundle 训练 40 步,总损失下降(监督可端到端使用);
7. bundle 契约声明了输入边界与权重。

全量 teacher 测试:**91 passed**。
