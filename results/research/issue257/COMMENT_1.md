已冻结验收清单（12 条，逐条映射到可执行证据），并记录**本轮不需要正式训练**：本 Issue 的验收是"可复用流水线 + 对已完成 run 的只读复算 + 受控短四卡入口"，正文明确禁止新增正式 3000/150 训练。

- 清单与正文哈希：`results/research/issue257/RUN_LOG.md`（正文 sha256 `f4509b74…f4d5ad`，附复核命令）。
- 故障台账（12 行，分类/复现/根因/修复/证据）：`results/research/issue257/FAULT_LEDGER.md`，每行指向仓库中已有的证据文件（如 `rounds/r02/FLOOR_LIFETIME_*`、`BATCH_PRECISION_DIAGNOSIS.json`、`rounds/r03/FINAL_STAGE2.json`），不重算、不删历史。
- 公共边界：`tools/issue257/{identity,timeline,registry,depgraph,analysis,replay,report,archive}.py`；启动仍只用 `tools/research/stage2/launcher.py`，归档仍只用 `results/research/issue139/upload_stage3.py`。

下一步：依赖图/状态机实测 + R01/R04 只读复算 + 受控短四卡入口。
