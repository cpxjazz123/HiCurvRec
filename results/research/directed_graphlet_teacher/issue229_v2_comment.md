## v2:用证据决定冻结监督的权重

v1 之后补做了三项 v1 没测的事(脚本 `probe_structure_v2.py`,产物 `teacher/structure/v2/`,
build SHA `401d0912…`;v1 原样保留)。

### 1. 768 维固定表示能承载什么(held-out 商品)

| 目标 | 载体 CE | 常数 CE | macro-F1 | 判定 |
|---|---:|---:|---:|---|
| relation(128) | 余弦 **0.8809** | 常数向量 0.1533 | — | **成立** |
| family(5) | 0.1421 | 0.1399 | 0.4996 | 未成立 |
| subtype(47) | 1.6196 | 1.5957 | 0.0919 | 未成立 |
| orbit(376) | 3.9150 | 3.4913 | 0.0271 | 未成立 |

⇒ **商品自身语义携带商品间关系几何,不携带局部角色直方图**。1,000 个覆盖商品上,
relation 相似度与 orbit-profile 相似度的 Spearman 只有 **0.0125** —— 两者几乎正交,
relation 不能当作角色目标的代理。

### 2. 方向:专用训练后仍未成立

| 方法 | test balanced accuracy |
|---|---:|
| 非对称头(valid 早停) | 0.5145 |
| 度数规则 | 0.5088 |
| 度数 logistic | 0.5039 |
| 多数类 | 0.5016 |

比多数类 +0.013、比度数规则 +0.006,低于写死的 +0.02 门槛;best step 仅 50 后即下降。

### 3. 冻结监督(权重由上面的规则推出,不是调的)

```json
{"relation": 1.0, "family": 0.0, "subtype": 0.0, "orbit": 0.0}
```

`structure_targets_v2.npz`:relation(128)+ subtype(47)+ family(5)+ orbit(376),
coverage 49.89%,confidence 0 表示无图证据,正/负对 32,637。角色目标保留为**保真记录**,
但固定输入不含可学信号,不推荐作为 RQ-VAE 损失项。

### Done 对照(v2,未降低标准)

识别(给结构)PASS(subtype 0.7121 / orbit 0.6101,2.08 角色/商品);
预测 PASS(C−A +0.0395、C−D +0.0343,双 seed);
商品级可承载 PASS(relation 0.8809、跨 seed Spearman 0.967)/ 角色部分**未成立并已声明**;
监督可交付 PASS;证据可信 PASS;方向 **未成立并已声明**。
本 Issue 仍不训练 RQ-VAE/Stage2/Stage3,不宣称 SID 或 Recall 效果。

报告:`REPORT_STRUCTURE_TEACHER_V2.md`;测试:`tests/test_structure_teacher_v2.py`。
