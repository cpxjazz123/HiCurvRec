## 联合训练答案:角色识别不是“没联合训练”的问题

一个编码器 + 四个头(edge / direction / orbit / subtype)同一次训练 1200 步,
角色头在训练时**排除的 20% 覆盖商品**上评估。

| 指标 | 联合 Teacher | 参照 |
|---|---:|---|
| 边 test AUC | 0.6947 | 边专用 C:**0.7473**(稀释) |
| 方向 balanced acc | 0.5036 | 多数类 0.5016 / 度数规则 0.5088 |
| orbit CE | **4.3015** | 常数 **3.5031**;冻结态探针 3.5488 |
| orbit 秩相关 | 0.1752 | — |
| subtype CE | 1.6670 | 常数 1.6064 |
| subtype macro-F1 | 0.1108 | 多数类 0.0687 |

结论:

1. 联合训练**没有**修复角色识别,orbit CE 反而更差 ⇒ 该架构无法从图表示恢复商品自己的 orbit 直方图,
   不是 head 或训练方式的问题。
2. subtype 排序相关 0.739、macro-F1 0.111 > 0.069,粗粒度顺序有一点点可学,但 CE 未过线,不计权重。
3. 联合训练把边 AUC 从 0.747 拖到 0.695(多任务稀释),所以交付的 Teacher 保留边专用编码器产出
   relation 目标,角色目标仅作保真记录(v3 权重 0)。

审计新增 `joint_role_training`:状态 **NOT_ESTABLISHED**,与 `minority_role_identification` 并列,
不再作为待办。报告 `REPORT_STRUCTURE_TEACHER_JOINT.md`,测试 `tests/test_joint_teacher.py`。
