## 节点 3:结论对账与逐条验收(C1–C10)

逐条验收表:`results/research/issue247/VERDICT.md`(每条给判据、实际值、证据路径、判定)与 `CHECKLIST.md`。**10/10 PASS**,另记 1 项 **NA**(梯度/参数更新检查——本轮不训练、不定义 loss,不是漏做)。

### 头条数字的独立重算

| 数字 | 重算路径 | 一致 |
|---|---|---|
| 有效曲率商品 24,326 / 缺失 261 | `prepare_v3.validate` 从可见图重抽三元组、参考点、距离,重算全部曲率值与逐商品摘要 | ✓ |
| 逐商品摘要(count/mean/std/负-零-正比例/分位数) | `sectional.summarize` 在三元组表上重跑,逐键与发布值比对 | ✓(`MODULE_CHECKS.json`) |
| 严格评估 ∩ 训练 bank = 0 | 全部 `strict_*` 与 `train_*` bank 的无向 key 交集 | ✓ |
| 有向二跳成员关系(全部 24,587 中心) | 矩阵级全等 + 2,000 个随机中心逐点重算 | ✓ |
| TIGER 前三码碰撞 0.1692764469 / 重复份额 0.2653028023 | 从 `item_sids.json` 现场重算 | ✓ |
| 手算图曲率(路径 0、星 −0.5、三角 0.25、四环 1) | 与 networkx + `Fraction` 独立实现对照 | ✓ |

### 七项强制机制落点

| 机制 | 文件 |
|---|---|
| 数学契约 + 论文/官方实现对照 | `results/research/issue247/CONTRACT.md`(含 GraphMoRE A.1 Eq. A.1 公式、边界、容差、对照表) |
| 独立参考实现 | `results/research/issue247/reference.py`(networkx BFS + `Fraction`,不 import 生产模块) + `tests/test_sectional_v3.py` |
| 确定性测试(手算期望值写死) | `tests/test_sectional_v3.py` |
| 变异测试 | `tests/test_mutation_v3.py` + `MUTATIONS.md`/`MUTATIONS.json`(**7/7 被指定断言捕获**) |
| 分模块验证 | `MODULE_CHECKS.json`(真实数据上 **12 PASS / 0 FAIL / 1 NA**) |
| 结论复核 | `VERDICT.md` 复核段 + 重算脚本 `prepare_v3.validate` |

### 关键事实(供后续使用,勿再被旧数字误导)

1. **可见用户 45,951**(不是 51,695);探针 62 对(不是 64);严格评估组最大 2,940(不是 2,938)。
2. **#246 没有任何曲率资产文件**;`curvature.py` 是 Ollivier–Ricci,不能当作截面曲率。本轮新建的是 GraphMoRE A.1 统计量。
3. 官方 GraphMoRE 仓库**未发布曲率估计器**,数值对照用的是论文公式的独立实现,已在契约里写明。
4. 本结果是**无向投影上的截面统计量**,不是有向截面曲率;方向信息另存于 `directed_edges.npz` 与有向 1/2 跳邻域。
5. **261 个商品没有有效曲率**(116 孤立点 + 无可达参考点的小分量),以掩码表示,**不填 0**。
6. 参考点预算敏感:16→64 参考时 4.67% 的三元组符号翻转,64→256 时 2.18%。任何下游使用必须固定参考预算并声明。
7. 4 节点实例仍是抽样(500 中心、覆盖 1.96% 中心、占总体 9.3e-7),选择对节点 ID 有偏,**不给无偏置信区间**。
8. TIGER 历史 Recall 与重建误差标 **NA**(无可绑定的不可变 SID 哈希;checkpoint 属另一轮)。
9. **PPR 全程未计算、未读取、未引用**;v2 里的 `_ppr_cache.npz` 被显式跳过。

### 复验

```bash
bash tools/graph_infra/verify_v3.sh   # 退出码 0
```

单元/契约 9 条 PASS、变异 7/7、模块检查 12 PASS + 1 NA、重建复现三元组/距离/曲率/摘要/二跳成员、manifest 50 项哈希校验通过;既有 #246 的 17 个套件全部 PASS。

**commit**:`5d9d930c63445f80f7f83382cc9bca99812fb997`,本地与远端 `main` SHA 一致。
