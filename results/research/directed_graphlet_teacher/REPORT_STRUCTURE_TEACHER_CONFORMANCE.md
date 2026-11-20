# #229 符合性审计(机器可复核)

日期:2026-11-21
脚本:`verify_conformance_229.py`;结果:`conformance_229.json`

## 结论

```
verdict: DONE_WITH_DECLARATIONS
failed: []
declared_not_established: [prediction_direction, minority_role_identification]
```

每一条都读取**已交付产物**而不是本文件的说法;审计脚本以退出码 0 表示无 FAIL。

| # | 检查 | 状态 | 决定它的数字 |
|---|---|---|---|
| 1 | `recognition_api`(诱导邻接 → family) | PASS | K3/P3/C4/P4/K1_3 全部正确 |
| 2 | `recognised_type_from_adjacency` | PASS | subtype macro-F1 0.7121 / orbit 0.6101(#221 产物) |
| 3 | `multi_role` | PASS | 覆盖商品平均 2.08 角色,46.1% 多角色 |
| 4 | `prediction_edge` | PASS | C−A / C−D 的双 seed CI 下界 > 0,平均效应 > 0.01,#228 判决 GRAPH_SIGNAL |
| 5 | `prediction_direction` | **NOT_ESTABLISHED** | 0.5145 vs 多数类 0.5016、度数规则 0.5088、度数 logistic 0.5039 |
| 6 | `item_carrier` | PASS | relation 0.8627 / neighbour_relation 0.8444;neighbour_orbit CE −8.2%、minority 3,581 |
| 7 | `minority_role_identification` | **NOT_ESTABLISHED** | 少数角色 top-1 ≈ 0.03 |
| 8 | `supervision_deliverable` | PASS | 4 维线性 student 40 步,总损失下降 |
| 9 | `integrity` | PASS | bundle 的全部 SHA 与载体参数摘要一致 |
| 10 | `boundary` | PASS | `structure_loss` 无 adjacency/graph 参数;监督不需要图 |
| 11 | `coverage` | PASS | 邻域覆盖 98.0%,own orbit 49.9% |

## 与 #229 Done 的对应

- **识别成立** → 1、2、3。
- **预测成立** → 4;方向按 #229 允许的方式单独声明未成立(5)。
- **商品级可承载** → 6(relation + 邻域多角色);own orbit 明确不可承载并记录。
- **监督可交付** → 8、10;bundle 提供 `structure_loss`。
- **证据可信** → 4 的对照与 CI、#228 的零泄漏审计、6 的常数与退化门槛。
- **产物完整** → 9 的 SHA 校验 + 本审计。

## 明确未做到(不降低标准)

1. 方向预测:低于多数类 + 0.02 的门槛,且弱于度数规则+0.02 的门槛。
2. 少数角色识别:邻域 orbit 的形状可承载(秩相关 0.593),但逐少数角色 top-1 仅 ~0.03。
3. 任何 SID / Recall 结论:本 Issue 不训练 RQ-VAE/Stage2/Stage3。

复现:`python3.9 verify_conformance_229.py`(退出码 0 = 无 FAIL)。
