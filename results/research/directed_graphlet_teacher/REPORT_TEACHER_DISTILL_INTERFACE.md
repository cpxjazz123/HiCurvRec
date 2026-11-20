# 冻结子图 Teacher 与无邻接 Student 蒸馏接口:可迁移性 Gate(Issue #222)

日期:2026-11-20
范围:Teacher 权重保存 + 严格冻结加载 + 无邻接 Student 接口 + 低成本可迁移性 Gate。
**本轮不训练任何 Stage 2 / Stage 3**,全部是冻结基线上的离线实验(§19g 不受影响)。

## 1. Teacher checkpoint:保存、冻结、复现

| 项 | 值 |
|---|---|
| 构件 | `teacher/teacher_directed_v1.pth`(payload:format / config / vocab / state_dict) |
| SHA-256 | 见 `teacher/teacher_metadata.json` 的 `sha256` 与 `MANIFEST.json` |
| 结构 | `SubgraphEncoder`(2 层 in/out 分离消息传递,hidden 128)+ `Readout`(池化 subtype + 上下文混合 orbit) |
| 训练配方 | 与 #221 `subgraph_predictor.py` 逐项相同:1500 steps、batch 512、seed 20261120、池 `instances.npz` |
| 加载语义 | `load_frozen_teacher`:format 校验 → `load_state_dict(strict=True)` → `eval()` → `requires_grad_(False)` |
| 重载证明 | `save_teacher` 内固定 probe 批次的前向必须与保存前 **逐位相等**,否则写入失败 |
| 词汇表 | `mode=directed`、`n_subtypes`、`n_orbits` 写进 payload;调用方给出期望词汇不一致即拒绝 |

同口径复跑(`teacher_controls.json`,与 #221 的独立运行逐位一致):

| 输入 | subtype macro-F1 | 无条件 orbit macro-F1 | 有 subtype 掩码的 orbit macro-F1 |
|---|---:|---:|---:|
| 真实有向邻接 | **0.7121** | **0.6101** | 0.9536 |
| 方向按无序对翻转 | 0.1694 | 0.1462 | 0.9387 |
| 邻接全零 | 0.0297 | 0.0117 | 0.9364 |

三行与 #221 报告的数字完全相同(0.7121 / 0.6101、0.1694 / 0.1462、0.0297 / 0.0117),
所以这份 checkpoint 就是产生那些数字的那次运行,而不是另一次碰巧接近的运行。

## 2. Student 接口:输入契约与蒸馏损失

```text
x = stage1_embeddings[item_ids]              # (B,k,768)   Trainer 拥有
t_type, t_orbit = teacher(x, directed_adj)   # 只在 TRAIN 侧;torch.no_grad()
z = rqvae.encoder(x)                         # (B*k,32)
q = sum_level codebook[level][token]         # 真实三层硬量化的码字和,不是连续 z
s_type, s_orbit = student_head(q)            # 无邻接、无标签、无位置 id
L_KD = T^2 * [ KL(p_T(type)||p_S(type)) + alpha * mean_j KL(p_T(orbit_j)||p_S(orbit_j)) ]
```

- `frozen_sid.FrozenSID` 从基线 `rqvae_best.pth` 读编码器(含 bias)与三张码本,
  SimVQ 经 `embed_proj`;`item_sids.json` 的第四项是导出碰撞计数器,不参与量化。
- `kdistill.kd_loss` 用 `kl_div(log_softmax(student/T), softmax(teacher/T))`,
  其解析梯度 $T(p_s-p_t)/N$ 由测试逐元素固定(符号检查挡不住写反的实现)。
- 两个头:`StudentQuantisedHead`(池化读出头)与 `StudentRelationalHead`(附加成对
  $|q_i-q_j|$、$q_i\odot q_j$ 的 mean/max)。后者是**容量正对照**:它能在合成关系标签上
  达到 0.9+,因此它在真实码字上失败只能说明码字本身,不能归咎于读出头。
- 两个头都在第一层前加 `LayerNorm`。这不是装饰:码字每维量级约 0.05,不加归一化时
  GELU 停留在近似线性区,整个头退化成码字的线性映射 —— 实测在真实标签上训练 1200 步,
  训练准确率卡在多数类 0.555,而随机标签也学不动(0.19);加上归一化后同一批数据
  训练准确率到 0.998。这个对照留在 `student_head.py` 的注释里,也是下面 Gate 数字可信的前提。

## 3. 数据审计:held-out 实例与训练池的节点对重叠

`instances.npz` 的 `*_split` 是**实例级**划分(训练与 held-out 的节点集合不同),
但不是**节点对级**划分。按完整训练池统计(不使用 cap):

| 族 | 训练实例 | held-out 实例 | 至少共享一条无条件节点对的 held-out 实例 |
|---|---:|---:|---:|
| K3 | 101,142 | 25,242 | 24,527(0.972) |
| P3 | 366,058 | 91,010 | 91,004(1.000) |
| C4 | 199,853 | 50,275 | 50,250(1.000) |
| P4 | 81,343 | 20,367 | 20,367(1.000) |
| K1_3 | 224,257 | 56,636 | 56,617(1.000) |
| 合计 | 972,653 | 243,530 | **242,765(0.997)** |

结论:**这份池子没有 pair-disjoint 的 held-out**。#221 的 0.7121 是「实例不同、节点对重叠」的
指标,不能当作「没见过的结构」。Gate 里另给一列 pair-clean 子集,但它的参照是 Gate 实际采样
的 5000/族训练集(而不是完整 972k 训练池),所以那一列只是「相对本次训练集未见过的节点对」,
比完整池的 0.3% 宽松,两个数在报告里必须并列写明。

## 4. 可迁移性 Gate

同数据、同 seed、同优化预算(每族 2000 个全批量 epoch,训练集 5000/族),只改监督信号。
指标为 subtype macro-F1(47 类全局词汇,出现过的类才计入)与**无条件** orbit macro-F1;
`_spec` 是每族一个专用头,`_rel` 是带成对项的读出头。

| 变体 | 头 | 训练 acc | held-out acc | held-out macro-F1 | pair-clean acc | 无条件 orbit F1 |
|---|---|---:|---:|---:|---:|---:|
| `random` | — | 0.0088 | 0.0090 | 0.0043 | 0.0095 | 0.0005 |
| `majority` | — | - | - | 0.0777 | - | 0.0131 |
| `degree` | — | - | - | 0.0182 | - | - |
| `hard_ce` | pooled | 0.3669 | 0.2210 | 0.0208 | 0.2075 | 0.0078 |
| `teacher_kd` | pooled | 0.3483 | 0.2392 | 0.0277 | 0.2249 | 0.0087 |
| `shuffled_kd` | pooled | 0.2774 | 0.2331 | 0.0285 | 0.2209 | 0.0087 |
| `teacher_kd_z` | pooled | 0.3441 | 0.2499 | 0.0280 | 0.2370 | 0.0097 |
| `hard_ce_rel` | relational | 0.5924 | 0.1730 | 0.0213 | 0.1615 | 0.0083 |
| `teacher_kd_rel` | relational | 0.5204 | 0.2000 | 0.0268 | 0.1890 | 0.0085 |
| `teacher_kd_z_rel` | relational | 0.4968 | 0.2226 | 0.0295 | 0.2027 | 0.0090 |
| `hard_ce_spec` | specialist | 0.9199 | 0.3403 | 0.1356 | 0.3373 | 0.0562 |
| `teacher_kd_spec` | specialist | 0.9283 | 0.3475 | 0.1136 | 0.3468 | 0.0461 |
| `shuffled_kd_spec` | specialist | 0.3482 | 0.3243 | 0.1166 | 0.3338 | 0.0437 |
| `teacher_kd_z_spec` | specialist | 0.9290 | 0.3471 | 0.1215 | 0.3422 | 0.0501 |
| `teacher_kd_z_rel_spec` | specialist | 0.9604 | 0.3449 | 0.0844 | 0.3406 | 0.0492 |


专用头逐族(held-out / pair-clean):

| 族 | hard_ce_spec | teacher_kd_spec | 多数类 |
|---|---:|---:|---:|
| K3 | 0.1566 / 0.1456 | 0.1556 / 0.1516 | 0.1037 |
| P3 | 0.1961 / 0.1933 | 0.1691 / 0.1679 | 0.1383 |
| C4 | 0.0676 / 0.0753 | 0.0568 / 0.0604 | 0.0424 |
| P4 | 0.1438 / 0.1266 | 0.0853 / 0.0897 | 0.0393 |
| K1_3 | 0.1137 / 0.1239 | 0.1015 / 0.1115 | 0.0649 |

## 5. 判定

**结论一:结构信息确实能迁移。** 在只读冻结基线硬码字 `q` 的专用头上,
`hard_ce_spec` 的 held-out macro-F1 为 **0.1356**、`teacher_kd_spec`
(没有任何硬标签,只有 Teacher 的软分布)为 **0.1136**,而多数类基线是
**0.0777**、度数查表是 **0.0182**。pair-clean 子集上的排序相同
(0.1329
/ 0.1162 vs
0.0842)。专用头的训练准确率
0.92-0.96,held-out 0.34-0.35,说明这不是欠拟合造成的假阴性:共享头(训练 acc 0.35-0.59)
的早期数字已作废。

**结论二:迁移到的主要是共享的无向骨架,不是方向。** 同一 Teacher 权重、把每条无向对的方向
按 1/2 翻转后生成的目标,让专用头得到 **0.1166**,与真实方向教师的
**0.1136** 同带。必须同时读的两条限制:(i) 打乱教师的训练拟合只有
0.348,真实教师是
0.928,所以这不是「同等强度的对照」;
(ii) 真实教师自己也只有 0.7121/0.6101,而它的学生停在与硬标签天花板相同的水平。合起来的
读法是:**学生能从码字里拿到的结构量,已经被硬标签上限钳住,方向打乱的目标足以把它填满**,
因此方向特异的蒸馏收益不存在,不能用「打乱也差不多」反过来宣称方向无用。

**结论三:量化不是瓶颈。** 同一专用头在连续 `z` 上是
**0.1215**,在硬 `q` 上是 **0.1136**;
若把量化当作瓶颈,`z` 应当明显更高,事实不是。

**做法**:按 §C.5,方向特异的信号不足以支撑 §D 的完整 Stage2 因果对照,本轮停在接口可行性层;
若要继续,应以「无向骨架蒸馏」为新假设重新立题,并把 pair-disjoint 数据缺口先补上
(§3:当前池子 99.7% 的 held-out 实例与训练池共享节点对)。

## 6. 复现入口与测试

| 入口 | 作用 |
|---|---|
| `fit_teacher.py` | 训练并保存 Teacher、三行对照、metadata、probe |
| `gate.py` | 可迁移性 Gate(全部变体与基线) |
| `manifest.py` | `MANIFEST.json`(输入、checkpoint、每个脚本与 JSON 的 sha256) |
| `tests/test_teacher_contract.py` | 重载逐位一致、全参数冻结、反向不落梯度、排列不变/等变、词汇不一致与截断 payload 拒绝 |
| `tests/test_student_interface.py` | Student 签名/源码无邻接、带邻接的 bundle 必须抛错、KD 梯度精确、KD 非对称、反向边按无序对拆分、关系头容量正对照 |

测试在 Stage 2 解释器下执行:
`/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 -m pytest tests/ -q`
