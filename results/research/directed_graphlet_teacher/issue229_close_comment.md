## 交付:冻结的多角色结构 Teacher(`structure-teacher-v1`)

| 产物 | SHA-256(前 12) |
|---|---|
| `teacher/structure/structure_targets_v1.npz`(24,587 行目标表) | `c0962cd20f69` |
| `teacher/structure/teacher_structure_v1.pth`(图编码器 + 768→relation/orbit 载体) | `ec465a7d81e7` |
| `teacher/structure/teacher_structure_contract.json` | `b970eaf2d8f8` |

### Done 对照

| 要求 | 状态 | 证据 |
|---|---|---|
| 识别成立(多类 graphlet + orbit,多角色) | **PASS** | 给邻接:subtype 0.7121 / orbit 0.6101;每商品 **2.08** 角色,46.1% 多角色 |
| 预测成立(无真值结构,稳定超语义与拓扑对照) | **PASS** | C−A **+0.0395 [0.0372,0.0418]**、C−D **+0.0343 [0.0322,0.0363]**,双 seed |
| 商品级可承载 | **PASS** | 关系载体 held-out 余弦 **0.8465**;双 seed 排序 Spearman **0.967**;近重复 0.000 |
| 监督可交付 | **PASS** | 目标表 + `orbit_loss` / `relation_ranking_loss` / `relation_distill_loss`;推理不需图 |
| 证据可信 | **PASS** | 四类泄漏违规 0;度数保持重接线;40-step 对照 0.5039 |
| 产物完整 | **PASS** | 权重 + 目标表 + 契约 + manifest(SHA)+ 报告 + 10 项测试 |

### 输入边界

- RQ-VAE Encoder 仍只吃 **768 维 Stage1 向量**;无邻接、无 Teacher hidden、无实例标签。
- 冻结 Teacher 只在 RQ-VAE **训练损失**中使用;不改 SID 推理协议。
- 监督函数签名中不存在 `adjacency/graph/edge` 参数(有测试断言)。

### 明确未成立(不得省略)

1. **方向预测未成立**:balanced accuracy 0.497–0.511 vs 多数类 0.5016。
2. **orbit 不可由 768 维语义的线性载体复现**:载体 CE 3.4956 vs 常数 3.4913。
3. **无 SID 碰撞率/Recall 结论**:本 Issue 按契约不训练 RQ-VAE/Stage2/Stage3。

报告:`REPORT_STRUCTURE_TEACHER.md`;机器可读:`structure_teacher_report.json`;
复现:`python3.9 structure_teacher.py` + `pytest tests/test_structure_teacher.py`。
