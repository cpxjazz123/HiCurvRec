## 符合性审计:机器可复核的 Done

`verify_conformance_229.py` 逐条读取已交付产物并给出 PASS / NOT_ESTABLISHED / FAIL,
退出码 0 表示没有 FAIL。当前结果(`conformance_229.json`):

```
verdict: DONE_WITH_DECLARATIONS
failed: []
declared_not_established: [prediction_direction, minority_role_identification]
```

| 检查 | 状态 | 数字 |
|---|---|---|
| recognition API(五 family) | PASS | 全部正确 |
| 识别(给邻接) | PASS | subtype 0.7121 / orbit 0.6101 |
| 多角色 | PASS | 2.08 角色/商品,46.1% 多角色 |
| 边存在性预测 | PASS | C−A / C−D 双 seed CI 下界 > 0,GRAPH_SIGNAL |
| 方向 | **NOT_ESTABLISHED** | 0.5145 vs 0.5016 / 0.5088 / 0.5039 |
| 商品级载体 | PASS | relation 0.8627 / neighbour_relation 0.8444 / 邻域 orbit CE −8.2% |
| 少数角色识别 | **NOT_ESTABLISHED** | top-1 ≈ 0.03 |
| 监督可交付 | PASS | 4 维 student 40 步损失下降 |
| bundle 完整性 | PASS | 全部 SHA + 载体参数摘要一致 |
| 边界 | PASS | `structure_loss` 无 adjacency/graph 参数 |
| 覆盖 | PASS | 邻域 98.0% / own orbit 49.9% |

未做到的三项(方向、少数角色、SID/Recall)已逐条声明,不降低验收标准。
报告:`REPORT_STRUCTURE_TEACHER_CONFORMANCE.md`;测试:`tests/test_conformance_229.py`。
