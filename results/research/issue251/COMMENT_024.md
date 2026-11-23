最终CHECK_4：8 PASS、1 BLOCKED、0 NOT_RUN；逐C原文/实际值/生产路径:SHA见 `results/research/issue251/VERDICT.md:6a2b7e17cd264c06a51222793a3b995f499635d75661d37c972a8a2f97e3e3ed`，本轮清单 `CHECK_4.md:59cfaa4659aeb43ccfaa52917f66451d7c30b19621ea5272fd0bd5db4189965c`。源码/全部最终运行与评估证据已在commit `a529589d71bb936cba244cc62d9f5a6fbab51362` 推送，实测本地main=远端main。

结论复核也已实际执行：`<Stage2-python> -m tools.issue251.recompute_report` 退出0，不import模型/评估器，以Python fsum从每商品计数/误差独立复算宏/微MAE，从原始SID/入口日志重算三码碰撞，并对账报告和10个strict bank；最大曲率差约1.2e-14<1e-9。`REPORT_RECHECK.json:9d96fe1a2928ff66a67e3865e2ee7003272022ba0c6d5115fcd3efe216bb5cf3`；该复算器先红后绿，2个手算回归覆盖macro/micro混淆及第四列计数器冒充零碰撞。

未满足的不是待做工作：实际公开请求
`verification_check({"source_dir":"TIGER_gnn_attention_sectional_rq","verification_id":"machine:issue251"})`
原始结论为 `entry: PASS; behavior_graph: FAIL; passed: false`，拒绝原文 `reads a graph outside the canonical source in model/mechanism.py, tests/test_structure.py`。完整回执SHA `12675d046954f62e3dab7ed81fae02a29a3b14111ea857b436c2bb47c8e6f87f`。冻结图和规范图有62501有向边/120318无向弧的精确差集，训练输入SHA相同不代表同图。

已排除：GPU/数据/凭证缺失（真实四卡、全目标与入口完成）；可修复实现/OOM（显存回归与4096入口通过）；随机端口（实际50202）；MCP超时或丢回执（持久化缓存经公开工具0.179秒取回，当前图仍重新检查）；测试未跑或仅有参考函数通过（真实模型Loss、参数更新、全目标和重导出均有绑定证据）。没有剩余可自行完成的实现/测试/评估/训练任务，不以时长、工作量或上下文作为阻塞。

最小owner决定：A，明确授权本候选使用当前冻结图/ego/目标的具名§89b例外；或B，批准新的规范图一致图/ego/全曲率目标/strict bank版本，并重冻结公式、计数与校准。Agent不能自行选一条规则放宽。按硬合同冲突以BLOCKED关闭，不伪称COMPLETED；本轮没有正式Stage2/Stage3、TIGER重训或收益主张。
