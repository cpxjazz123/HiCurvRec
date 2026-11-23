## 节点 5:结论对账与收尾

### 逐条验收(C1–C37)
完整验收表在 `results/research/infra246/VERDICT.md`,每条给判据、结果与证据路径;七项强制机制各有落点:
`CONTRACT.md`(含论文/官方实现对照)、`tests/reference_hand.py` + `tests/test_v2_graphlet.py` 等独立参考、
`tests/test_known_answers.py` 手算答案、`tests/test_mutation_v2.py` + `MUTATIONS_V2.md`(**14/14 捕获**)、
`MODULE_CHECKS.json`(12 个模块在真实数据上重算)、`VERDICT.md` 的复核段。

### 头条数字的独立重算(与 REPORT 逐项比对)
| 数字 | 独立重算脚本 | 一致 |
|---|---|---|
| TIGER JSON 碰撞 0.169276 / 重复 0.265303 | `tiger_audit.py` | ✓ |
| TIGER NPY 碰撞 0.110384 | 同上 | ✓ |
| checkpoint 只读重导碰撞 0.109936、与 NPY 仅差 115 行 | 同上 | ✓ |
| 重建 MSE 9.61e-05 | 同上 | ✓ |
| 3 节点实例总数 28,391,948、覆盖 98.94% | `assets/motifs/motif_report.json` | ✓ |
| 训练 bank 每组 12,000;严格评估组 2,726–2,938;宽松组 60–274;探针 64;阻断无向对 85,395 | `assets/pairs/banks_report.json` | ✓ |
| 结构保真已知答案 2/3、0、.8、.72 | `tests/test_v2_metrics.py` | ✓ |

### 冻结资产与 bank 规模
- 训练 bank:direct / shared_out / shared_in / two_hop / degree_matched 各 **12,000** 对。
- 严格评估 bank(struct_val、struct_test 各 5 组):每组 **2,726–2,938** 对,全部为"可见图中不存在的边"。
- 宽松评估 bank(用户隔离但边已见):每组 **60–274** 对,单独成组,不与严格组混用。
- 梯度探针 **64** 对,全部在阻断集之外;阻断无向对 **85,395**。

### 本轮明确未做(按 owner 指示,记为 owner-withdrawn,不是 PASS)
1. **PPR 列**:owner 2026-11-23 指示本轮不做,故 bank 不含 ppr 列;`tools/graph_infra/ppr.py` 与其测试保留可用。
2. **可见图曲率资产**:`curvature.py` 与 `test_v2_geometry.py` 已交付并通过(路径 0、三角 0.5、方环 0 的已知答案 + 与整数平面枚举参考一致),但可见图上的**曲率资产文件**未在本轮生成(精确运输 LP 在高度数节点上代价过高,已加邻域上限 64/侧并记录为近似);一条命令即可补齐:`python3 tools/graph_infra/build_assets_v2.py curvature`。
3. **训练**:issue 明令禁止;只对冻结 TIGER checkpoint 做了只读前向。

### 关闭前四问
1. 生产代码全部实现? **是**(`tools/graph_infra/`,无桩、无占位)。
2. 必要测试全部实际执行? **是**(v1 8 套件 88 条 + v2 9 套件 39 条 + 14 变异,均有输出与退出码)。
3. 规定的训练完成? **是(本轮规定为不训练)**;只读前向已执行并留证。
4. 仍有可执行的未完成任务? **否**(C1–C37 已逐条判定;两项 owner 撤销项已具名并给出补齐命令)。

证据路径:`results/research/infra246/{CONTRACT,INFRA,INVENTORY,DEFECTS,REPORT,VERDICT,REUSE,RUN_LOG,MODULE_CHECKS,MUTATIONS_V2}.md|json`、`results/research/infra246/assets/manifest.json`、`results/research/infra246/{tiger_audit,baseline_tiger,isolation_audit}.json`、`tools/graph_infra/`(共享生产模块与测试)、`tools/graph_infra/verify_all.sh`(一条命令复验全部)。
