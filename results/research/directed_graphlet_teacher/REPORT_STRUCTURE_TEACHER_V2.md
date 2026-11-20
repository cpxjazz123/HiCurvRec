# 结构 Teacher v2:证据决定监督权重(Issue #229 续)

日期:2026-11-21
版本:`structure-teacher-v2`,build SHA `401d091213f6025af6cce04fe4dd80f9ace7174440efd090a5cc28859d451893`
实现:`structure_teacher.py::build_structure_teacher_v2`;契约:`teacher/structure/v2/teacher_structure_contract_v2.json`
诊断探针:`probe_structure_v2.py` → `teacher/opt/structure_probe_v2.json`

v1 保留不动(`structure-teacher-v1`,`build_sha ba66a98a…`)。v2 不是“换一个更大的模型再试一次”,
而是把**哪些结构信息真的能被固定商品表示承载**测清楚,并据此决定冻结监督的权重。

## 0. 为什么有 v2

v1 的契约把四个目标写在一起,但只有一个测过载体:relation。v2 回答三个 v1 没回答的问题:

1. 768 维 Stage1 向量能否承载 **角色**(subtype 47 / family 5 / orbit 376)?
2. 方向能否在**专用训练**下超过度数基线(而不是只超过 0.5)?
3. 被承载的 relation 目标是否顺带保留角色信息(两者是否同一个信号)?

## 1. 结果

### 1.1 载体(held-out 商品,80/20 按商品划分)

| 目标 | 载体 CE | 常数 profile CE | 相对 | 载体 macro-F1 | 多数类 macro-F1 | 判定 |
|---|---:|---:|---:|---:|---:|---|
| relation(128) | — | — | — | 余弦 **0.8809** | 常数向量余弦 0.1533 | **成立** |
| family(5) | 0.1421 | 0.1399 | +1.6% | 0.4996 | 0.4996 | 未成立 |
| subtype(47) | 1.6196 | 1.5957 | +1.5% | 0.0919 | 0.0669 | 未成立 |
| orbit(376) | 3.9150 | 3.4913 | +12.1% | 0.0271 | 0.0037 | 未成立 |

同一切分上的**识别探针**(768 → 47/376,单独训练):

| 识别任务 | CE | 常数 CE | macro-F1 |
|---|---:|---:|---:|
| 768 → subtype | 1.6178 | 1.6064 | 0.0836 |
| 768 → orbit | 3.6157 | 3.5031 | 0.0211 |

结论:**商品自身的 768 维语义不携带它的局部角色直方图**(三种粒度都不如常数),
但它**确实携带商品间的关系几何**(0.881,常数向量 0.153)。这不是模型容量问题:
同一个载体用 1024 宽、1200 步、dropout,relation 学得很好,角色目标仍停在常数附近。

### 1.2 relation 与角色是否同一个信号

1,000 个覆盖商品的两两余弦:relation 空间相似度 vs orbit-profile 相似度,
**Spearman = 0.0125**。⇒ 两者几乎正交;relation 目标**不能**当作角色目标的代理。

### 1.3 方向(专用训练,冻结 C 状态 + 非对称头,valid 早停)

| 方法 | 测试 balanced accuracy |
|---|---:|
| 非对称方向头(1200 步上限,按 valid 选) | 0.5145(最好 valid 0.5198,step 50) |
| 度数规则(valid 上定方向) | 0.5088 |
| 度数 logistic | 0.5039 |
| 真实多数类 | 0.5016 |

判定:**NOT_ESTABLISHED**。比多数类高 0.013、比度数规则高 0.006,
低于事先写死的“超过多数类与度数基线各 0.02”的门槛;而且 best step 只有 50,
之后下降,说明学到的是弱且不稳的度数相关信号。

## 2. 因此冻结的监督是什么

`structure_targets_v2.npz`(24,587 行,商品 id 对齐):

| 字段 | 形状 | 角色 |
|---|---|---|
| `relation` | (24587, 128) | **推荐使用的主目标**;两两余弦 = 商品间结构关系 |
| `subtype` | (24587, 47) | 多角色保真记录;载体未成立 |
| `family` | (24587, 5) | 最粗多角色视图;载体未成立 |
| `orbit` | (24587, 376) | 全保真 orbit 分布;载体未成立 |
| `coverage` / `confidence` | (24587,) | 49.89% 覆盖;无图证据的商品置信度 0 |
| `positive_pairs` / `negative_pairs` | (32637, 2) / (32637,) | 关系排序损失的正例与度数匹配负例 |

**权重规则(先写死,不是调出来的)**:某分量只有在载体优于常数 profile ≥2% CE
(或 relation 余弦 > 0.8)时才给非零权重。实测:

```json
{"relation": 1.0, "family": 0.0, "subtype": 0.0, "orbit": 0.0}
```

⇒ 推荐给 RQ-VAE 训练损失的是 **relation 主目标**(`relation_ranking_loss` /
`relation_distill_loss`);角色目标保留为可复核的保真记录,但**不推荐**作为损失项,
因为固定输入不含可学信号,加进去只会增加噪声。这一条直接写在契约的
`supervision_recommendation` 里,并附上判据。

## 3. 与 #229 Done 的关系(不降低标准)

| 要求 | v2 状态 | 证据 |
|---|---|---|
| 识别成立(给结构) | **PASS** | 给诱导邻接:subtype 0.7121 / orbit 0.6101;每商品 2.08 角色,46.1% 多角色 |
| 预测成立 | **PASS** | 边存在性:C−A +0.0395 [0.0372,0.0418]、C−D +0.0343 [0.0322,0.0363],双 seed |
| 商品级可承载 | **PASS(relation)/ 未成立(角色)** | relation 0.8809、跨 seed Spearman 0.967;角色三粒度均不如常数,已声明 |
| 监督可交付 | **PASS** | 目标表 + 三个损失 + 证据决定权重;推理不需图;不改 RQ-VAE |
| 证据可信 | **PASS** | 泄漏违规 0;度数保持重接线;40-step 对照 0.5039;方向有度数基线对照 |
| 方向 | **未成立,已声明** | 0.5145 vs 0.5016 / 0.5088 / 0.5039 |
| 产物完整 | **PASS** | v2 权重 + 目标表 + 契约 + manifest + 本报告 + 测试 |

**必须同时读的限制**:角色直方图**在目标里**、**不在固定表示的可学范围里**。
任何“RQ-VAE 能保住多角色结构”的说法都必须等到用 relation 目标训练后的
独立 Stage2/Stage3 验证,本 Issue 不宣称。

## 4. 复现

```bash
python3.9 probe_structure_v2.py                 # 识别探针 + 载体 + 方向
python3.9 -c "import structure_teacher as s, torch; print(s.build_structure_teacher_v2(torch.device('cuda:0'))['build_sha'])"
python3.9 -m pytest tests/test_structure_teacher_v2.py -q
```

前置:`teacher/opt/C_residual_edge_main_seed226.pth`(SHA 见契约 `build` 块)、
`behavior_graph_directed/graph.npz`、`sentence_t5.npy`。
