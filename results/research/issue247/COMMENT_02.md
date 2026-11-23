## 节点 2：审计结论 + 曲率资产 + 隔离接口（构建与校验已 PASS）

### 1) #246 审计（以文件实际内容为准，不覆盖冻结资产）
| 项 | 此前报告 | 文件实测 | 结论 |
|---|---|---|---|
| 可见用户数 | 51,695 | **45,951**（struct_val 5,744 / struct_test 5,744） | 此前数字有误 |
| 梯度探针对数 | 64 | **62** | 此前数字有误 |
| 严格评估组规模 | ≤2,938 | 2,726–**2,940** | 此前数字有误 |
| v2 曲率资产 | 称"近似 Ollivier–Ricci 已生成" | **v2 无任何曲率文件**；`curvature.py` 是 Ricci，不是 A.1 截面统计 | 本轮新建 |
| 3 节点实例总数 | 28,391,948 | 一致（逐对精确计数） | 一致 |

逐项复核方式：47/47 v2 manifest 哈希未漂移；每个用户划分用**独立 Counter 重算原始转移**与 CSR 比对（自环丢弃 0）；严格 bank 与训练 bank 的无向 key 交集 = 0。证据 `results/research/issue247/AUDIT_246.json`。

### 2) 截面曲率（GraphMoRE A.1，非 Ricci）
- 公式：`kappa(m;b,c) = mean_a[ d(a,m)^2 + d(b,c)^2/4 − (d(a,b)^2+d(a,c)^2)/2 ]`，见 arXiv 2412.11085v1 附录 A.1 Eq. A.1。
- 距离：**可见有向图的无向投影**上的无权最短路。因此结果**不称为有向截面曲率**；方向特征另行保留（`directed_edges.npz` 的 `reciprocal` 与有向 1/2 跳邻域）。
- 官方实现局限：GraphMoRE 公开仓库（commit `1948d3c`）**未发布截面曲率估计器**，只有 `utils.cal_shortest_dis`（networkx 最短路）。因此数值对照用**论文公式的独立实现**（`results/research/issue247/reference.py`，networkx BFS + Fraction 有理数），不伪称官方曲率代码。
- 三元组：每个合格中心一个均匀无序邻居对（seed 247001）；参考点：全局无放回置换（seed 247002）的前 16/64/256，嵌套做敏感性检查。
- 缺失处理：不可达参考点**掩码缺失，绝不填 0**；空邻域中心无三元组，摘要为 NaN 且 `valid=false`。

**真实数据结果**：24,587 商品中 24,326 个有有效曲率，261 个缺失（116 个孤立点 + 小分量无参考点）；分量数 117；合格邻居对总体 28,391,948，抽样覆盖 0.086%。
**参考点敏感性（同一三元组、嵌套参考集）**：16→64 参考时平均绝对变化 0.475、最大 2.811、符号翻转比例 4.67%；64→256 时 0.233 / 1.243 / 2.18%。参考量少时符号不稳，是必须记录的成本事实。

### 3) 局部拓扑与接口
- 精确 3 节点族计数（复用 v2）、有向 1/2 跳稀疏邻域（24,587 中心全量校验）、精确半径 1/2 的诱导 ego 加载器（`root_local`/hop/in/out mask/node_mask/edge_mask）。
- 4 节点实例仍是**抽样**（500 中心、481 实际有实例、覆盖率 1.96%、占总体 9.3e-7），选择机制对节点 ID 有偏，**不宣称全量、不给无偏置信区间**（`coverage.json` 明确写 NA）。
- 轨道列是**抽样实例中的中心轨道参与数**，不是全量轨道统计。
- 训练接口 `TrainingAssets` 无法读取 `use=evaluation` 的资产；严格评估走 `HeldOutEvaluation`。严格评估 key ∩ 训练 bank = 0（`ACCEPTANCE.json`）。

### 4) TIGER 溯源
Stage3 实际读取的是 `item_sids.json`（sha256 `025ad8bd…`）；`sids_for_hgrec.npy`（`6f2682c1…`）与 checkpoint（epoch 300）属**另一轮**且从未被 Stage3 读取，二者单列。历史 Recall 与重建误差标 **NA**：无法把任何历史 recall 与不可变 SID 哈希绑定，checkpoint 也无法重建 Stage3 用过的表。

### 5) 验证
- 单元/契约测试 9 条 PASS（路径 0、星 −0.5、三角 0.25、四环 1、孤立分量、随机图与独立实现 1e-12 一致、节点重排、方向反转、参考点自重合、重复三元组、NaN/Inf、ID 错配、manifest 篡改）。
- 变异 **7/7 被指定断言捕获**（数值 3 / 结构 2 / 路由 2），见 `MUTATIONS.md` 与 `MUTATIONS.json`。
- 分模块验证 **12 PASS + 1 NA**（NA = 本轮不训练，故无梯度/参数更新检查）。
- 既有 v2 的 17 个测试套件全部 PASS（0 失败）。
- 一条命令复验：`bash tools/graph_infra/verify_v3.sh`。

产物：`results/research/issue247/{CONTRACT,INFRA,AUDIT_246.json,CHECKLIST,README,MUTATIONS.md,MODULE_CHECKS.json,ACCEPTANCE.json}`、`results/research/issue247/assets/`（manifest 50 项）、生产模块 `tools/graph_infra/{sectional,interfaces_v3,prepare_v3,module_checks_v3}.py` 与 `tests/test_{sectional,interfaces,mutation}_v3.py`。

未做（issue 明令）：训练、PPR、基线重训、新 loss。
