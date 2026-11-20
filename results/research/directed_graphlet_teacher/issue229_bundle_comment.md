## 交付物:一个可直接加载的冻结 Teacher Bundle

前几轮的产物(目标表、载体、契约)现在打包成一个对象,消费者不需要知道文件怎么拼:

```python
from structure_teacher_bundle import load_structure_teacher
teacher = load_structure_teacher(".../teacher/structure/bundle")
teacher.verify()                                # 源文件 SHA + 载体参数校验
teacher.recognise(matrix)                       # 诱导邻接 -> family/key/orbits
teacher.targets_for([12, 900, 24001])           # 图导出的监督行
teacher.predict_targets(embeddings)             # 768 维载体(无图回退)
metrics, loss = teacher.structure_loss(student_vectors, role_logits=logits)
```

### 能力矩阵(v3 证据,原样带进 bundle)

| 能力 | 状态 |
|---|---|
| 识别(给诱导邻接) | 成立(五 family 有测试;#221 subtype 0.7121 / orbit 0.6101) |
| 多角色(邻域) | 成立(CE −8.2%、秩相关 0.593、3,581 少数角色商品) |
| 边存在性预测 | 成立(C−A +0.0395、C−D +0.0343,双 seed) |
| 方向 | **未成立并声明**(0.5145 vs 0.5016 / 0.5088) |
| 商品级可承载 | relation 0.8627 / neighbour_relation 0.8444 |
| 逐角色(少数角色) | **未成立**(top-1 ≈ 0.03) |
| 监督可交付 | 成立(ranking + distill + orbit,权重来自 v3 契约) |

### 边界

RQ-VAE Encoder 仍只吃 768 维 Stage1 向量;Teacher 只在训练损失中使用;SID 推理协议不变;
本层不训练 RQ-VAE/Stage2/Stage3,不宣称 SID 或 Recall。

### 证据

- `structure_teacher_bundle.pth` / `_contract.json` / `_manifest.json`,SHA 互相校验;
- `verify()` 同时校验 v3 的 teacher / targets / contract 三个源文件与载体参数摘要;
- 测试 `tests/test_structure_teacher_bundle.py`(7 项),含“4 维线性 student 用 bundle 训练 40 步损失下降”;
- 全量 teacher 测试 **91 passed**。

报告:`REPORT_STRUCTURE_TEACHER_BUNDLE.md`。
