## 关闭:COMPLETED

**结果**:按 #246 的范围交付并验证了 v2 基础设施;修复了 #245 遗留的全部已发现缺陷,冻结了不依赖未来模型的数据资产,并查清了 TIGER SID 的真实来源。全程未训练(issue 明令),只对冻结 checkpoint 做了只读前向。

**验证(一条命令可复现)**:`bash tools/graph_infra/verify_all.sh`
- v2 manifest **47/47** 哈希校验通过(schema、shape、dtype、graph_version、legal_use、version、purpose、command、源/码哈希、依赖齐全)
- 17 个套件全绿(v1 的 8 套件 88 条 + v2 的 9 套件 39 条)
- 变异测试 **14/14** 全部被指定断言捕获(崩溃/导入错不计)
- 分模块验证 **11 PASS + 1 NA**(NA = 可见图曲率资产本轮未生成,owner 指示跳过构建;模块与已知答案测试已交付,补齐命令已记录)
- 隔离审计:严格评估 key ∩ 训练 bank = 0、∩ 梯度探针 = 0、∩ 宽松组 = 0;三路用户集合两两交集 0、并集完整、边权相加等于完整图
- TIGER 来源:Stage3 实际读的是 `item_sids.json`(`code_path_sha256=025ad8bd…`);checkpoint 与 `sids_for_hgrec.npy` 属另一轮(300 epoch)且从未被 Stage3 读取;只读重导与该 NPY 仅差 115 行;重建 MSE 9.61e-05

**更正 v1 的错误结论**:`l3_collision=0.2653` 并非不可复算,它是重复商品份额;门禁口径是 `1−distinct/items=0.169276`。已在 #245 INVENTORY 加更正注记。

**owner 撤销项(记为 owner-withdrawn,不是 PASS)**:①PPR 列(本轮不做,`ppr.py` 与其测试保留);②可见图曲率**资产文件**(模块+测试已交付,`python3 tools/graph_infra/build_assets_v2.py curvature` 可补齐)。

**证据路径**:`results/research/infra246/{CONTRACT,INFRA,INVENTORY,DEFECTS,REPORT,VERDICT,REUSE,RUN_LOG,MODULE_CHECKS,MUTATIONS_V2}.md|json`、`results/research/infra246/assets/manifest.json`、`{tiger_audit,tiger_baseline_v2,baseline_tiger,isolation_audit}.json`、`results/research/infra246/verify_all.log`;生产模块与测试在 `tools/graph_infra/`。

**commit**:`6312ce7d`(模块+测试+文档)、`799315db`(v2 资产清单与验证证据)、`49f21e63`(最终验证记录);本地与远端 `main` SHA 一致。
