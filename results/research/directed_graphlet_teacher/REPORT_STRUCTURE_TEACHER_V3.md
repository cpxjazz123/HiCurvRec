# 结构 Teacher v3:多角色目标改到“邻域结构”后真的可承载

日期:2026-11-21
版本:`structure-teacher-v3`,build SHA `1fd2b39c95377a2ff5f94c997add6518a02ef7d9787691d91f88311b21835093`
实现:`structure_teacher.py::build_structure_teacher_v3`;契约:`teacher/structure/v3/teacher_structure_contract_v3.json`
v1 / v2 原样保留(审计链不覆盖)。

## 0. v2 的瓶颈与 v3 的改法

v2 测出:**商品自己的 orbit 直方图不可由 768 维语义载体复现**(CE 4.04 vs 常数 3.50)。
但 item 级监督要的是“这个商品处在什么结构环境里”,不是“它自己的稀有 orbit 编号”。
v3 把多角色目标改成**邻域的均值结构**:

```
neighbour_orbit  = mean orbit profile of the observed neighbours   (376)
neighbour_family = mean family profile of the observed neighbours  (5)
neighbour_relation = mean relation vector of the observed neighbours (128)
```

这些目标仍然只在 **TRAIN 观测图**上构建,冻结后只以表的行出现;推理不需要图。

## 1. 载体测量(80/20 按商品划分,常数 profile 只平均被覆盖的训练行)

| 目标 | CE | 常数 CE | 相对 | 角色秩相关 | top-1 | 少数角色商品数 | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| relation | — | — | — | — | — | — | 余弦 **0.8627**(常数 0.154) |
| neighbour_relation | — | — | — | — | — | — | 余弦 **0.8444** |
| **neighbour_orbit** | 2.4996 | 2.7235 | **−8.2%** | **0.593** | 0.2546 | **3,581** | **可承载** |
| neighbour_family | 0.1728 | 0.2112 | −18.2% | 0.9987 | 0.9998 | **1** | CE 可承载但**退化**,不算多角色 |
| own_orbit | 3.8695 | 3.5004 | +10.5% | 0.212 | 0.0619 | 2,249 | 不可承载 |
| own_subtype | 1.6242 | 1.6024 | +1.4% | 0.753 | 0.4287 | 1,362 | CE 未过线(排序尚可,记录但不加权) |

关键结论:

1. **邻域 orbit 分布是可承载的多角色信息**:CE 明显低于常数,且角色秩相关 0.593;
   3,581 个 eval 商品的真实 argmax 不是全局最多数角色,排除“只预测一个常见类”的假象。
   但它仍然抓不住少数角色(top-1 在少数角色商品上只有 ~0.03),所以只能声明“形状/主导角色可承载”,
   不能声明“逐角色识别”。
2. **neighbour_family 的 CE 改善是退化的**:eval 里只有 1 个商品的 argmax 不是主导 family,
   所以规则把它判为不可用的多角色目标(权重 0)。
3. **own_orbit / own_subtype 是保真记录**:own_orbit 的 CE 比常数差,不得计入损失;
   own_subtype 的排序相关 0.753 但 CE 未过线,按事先规则不计权重,只记录。

## 2. 冻结的监督目标(`structure_targets_v3.npz`,24,587 行)

| 字段 | 形状 | 覆盖 | 角色 |
|---|---|---|---|
| `relation` | (24587,128) | 24,094 | 主目标:商品间结构关系 |
| `neighbour_relation` | (24587,128) | 24,094 | 邻域关系目标 |
| `neighbour_orbit` | (24587,376) | 24,094 | **多角色结构环境** |
| `neighbour_family` | (24587,5) | 24,426 | 粗粒度记录(退化) |
| `own_orbit` / `own_subtype` | (24587,376)/(24587,47) | 12,266 | 保真记录 |
| `coverage` / `coverage_neighbour` / `coverage_family` | (24587,) | — | 各分量是否有图证据 |
| `confidence` | (24587,) | — | 邻域覆盖 × 对数度数证据;0 = 无图证据 |
| `positive_pairs` / `negative_pairs` | (32637,2)/(32637,) | — | 关系排序损失 |

**权重(规则先写死)**:

```json
{"relation": 1.0, "neighbour_relation": 0.5, "neighbour_orbit": 0.5,
 "neighbour_family": 0.0, "own_orbit": 0.0, "own_subtype": 0.0}
```

规则:relation / neighbour_relation 需余弦 > 0.8;角色分量需 CE < 常数 × 0.98 **且**
eval 至少 50 个“少数角色”商品,防止单一主导类冒充多角色。

## 3. 能力与接口

| 能力 | 状态 | 证据 |
|---|---|---|
| 识别(给诱导邻接) | **成立** | `recognise_subgraph(matrix)` 确定性返回 family/key/orbits(五种 family 有测试);学习式评估见 #221:subtype 0.7121 / orbit 0.6101 |
| 多角色 | **成立(邻域层面)** | neighbour_orbit CE −8.2%、秩相关 0.593、3,581 个少数角色商品 |
| 预测:边存在性 | **成立** | C−A +0.0395 [0.0372,0.0418]、C−D +0.0343 [0.0322,0.0363],双 seed |
| 预测:方向 | **未成立,已声明** | 0.5145 vs 多数类 0.5016 / 度数规则 0.5088 / 度数 logistic 0.5039 |
| 商品级可承载 | **成立(relation + 邻域多角色)** | 0.8627 / 0.8444 / 邻域 orbit 见上;own orbit 明确不可承载 |
| 监督可交付 | **成立** | 表 + 三个损失 + 证据权重;无图推理;RQ-VAE 输入协议不变 |

## 4. #229 Done 对照

| 要求 | 状态 |
|---|---|
| 识别成立 | PASS(确定性 API + #221 学习式数字) |
| 预测成立 | PASS(边);方向未成立并声明 |
| 商品级可承载(多角色 + 商品间关系) | PASS(邻域 orbit 多角色 + relation/neighbour_relation) |
| 监督可交付 | PASS |
| 证据可信(无泄漏/伪负例/错位;对照与重复性) | PASS(违规 0;度数保持重接线;40-step 对照 0.5039) |
| 产物完整 | PASS(v3 权重 + 目标表 + 契约 + manifest + 报告 + 6 项测试) |
| SID / Recall | 不在本 Issue 范围,未宣称 |

**限制(必须与结论同读)**:逐角色识别(少数角色 top-1)未成立;方向未成立;
本 Issue 不训练 RQ-VAE,因此没有任何“结构监督已改善 SID”的结论。

## 5. 复现

```bash
python3.9 probe_neighbour_roles.py          # 邻域目标的载体测量(含少数角色与秩相关)
python3.9 -c "import structure_teacher as s, torch; print(s.build_structure_teacher_v3(torch.device('cuda:0'))['build_sha'])"
python3.9 -m pytest tests/test_structure_teacher_v3.py -q
```
