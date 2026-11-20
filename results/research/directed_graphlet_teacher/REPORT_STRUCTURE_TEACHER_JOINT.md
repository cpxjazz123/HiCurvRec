# 联合训练 Teacher:角色识别是否只是“没联合训练”的问题?

日期:2026-11-21
脚本:`train_teacher.py`;结果:`teacher/opt/joint_teacher_report.json`;
checkpoint:`teacher/structure/joint/teacher_joint_v1.pth`(SHA `9a812aa0…`)

## 设置

一个编码器(残差 GNN,128 维)+ 四个头,同一次训练:

```
edge BCE       (masked train pairs, 度数匹配负例)
direction CE   (one-way train pairs, balanced)
orbit CE       (376 角色,只在 80% 覆盖商品上)
subtype CE     (47 子类型,同一批商品)
loss = edge + 0.5·direction + 0.2·orbit + 0.1·subtype
1200 steps, accum 3, micro 4096, lr 3e-4
```

评估:边/方向用 #228 的 held-out;角色头在**训练时排除的 20% 覆盖商品**上评估。

## 结果

| 指标 | 联合 Teacher | 参照 |
|---|---:|---|
| 边 test AUC | 0.6947 | 边专用 C:**0.7473**(联合训练稀释) |
| 方向 balanced acc | 0.5036 | 多数类 0.5016;度数规则 0.5088 |
| orbit CE(held-out 商品) | **4.3015** | 常数 profile **3.5031**;冻结态探针 3.5488 |
| orbit 秩相关 | 0.1752 | — |
| subtype CE | 1.6670 | 常数 1.6064 |
| subtype 秩相关 | 0.7387 | — |
| subtype macro-F1 | 0.1108 | 多数类 0.0687 |

## 结论

1. **“没联合训练”不是角色识别失败的原因。** 联合训练后 orbit CE 反而更差(4.30 vs 3.50),
   说明该架构无法从图表示中恢复商品自己的 orbit 直方图;这不是 head 或训练方式的锅。
2. subtype 的排序相关 0.74、macro-F1 0.111 > 0.069,说明粗粒度角色顺序有一点点可学,
   但校准 CE 未过线,按规则不计入监督权重。
3. 联合训练把边 AUC 从 0.747 拖到 0.695 —— 多任务稀释。因此交付的 Teacher 保留
   **边专用编码器**产出 relation 目标,角色目标仅作保真记录(v3 契约的权重 0)。

这一结果写进符合性审计的 `joint_role_training` 检查:状态为 **NOT_ESTABLISHED**,
与 `minority_role_identification` 并列,不再作为待办。

## 可加载模型(不是只有一张表)

`joint_teacher_model.py` 把该 checkpoint 作为**模型**加载并复算:

```python
from joint_teacher_model import load_joint_teacher
teacher = load_joint_teacher()      # teacher/structure/joint/teacher_joint_v1.pth
teacher.verify()                    # 参数摘要 ed9bccc7…
state = teacher.encode(features, adjacency)
scores = teacher.score_pairs(state, positives, negatives)
teacher.recognise(matrix)
```

复算结果(`teacher/opt/joint_teacher_reload.json`):

| 指标 | 加载后复算 | 训练时记录 |
|---|---:|---:|
| edge test AUC | 0.6946952832 | 0.6946952724 |
| direction balanced acc | 0.5035671435 | 0.5035671435 |
| 参数摘要 | `ed9bccc7…` | 同 |

⇒ 模型权重、头部与评估通路可逐位重载并复现记录值(测试 `tests/test_joint_teacher_model.py`)。
