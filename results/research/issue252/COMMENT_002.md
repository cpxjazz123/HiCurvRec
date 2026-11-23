先红后绿已开始，证据 `results/research/issue252/RED_GRAPH_ARCHIVE.log`，命令 `<Stage2-python> -m unittest tools.issue252.test_graph_audit tools.issue252.test_archive_binding -v` 退出1。

- 图对账生产入口尚缺；手算用例固定“每用户最长序列、去自环但保留跨用户次数”，同时检查val-only/test-only/both差异和不能把短前缀重复计权。将独立读取原始train复算，不拿旧汇总自证。
- 发现实际归档正确性缺陷：现有`upload_stage3.stage`对同名目录中与Stage2 SID SHA不相同的Stage3结果照样归档，回归明确失败 `ValueError not raised`。当前源码还未把完整HG_Rec.log/metrics流纳入recall块，且可能上传第五件Stage3权重，均须按既有四件套规则修复。仅在隔离本地fixture执行，未读取凭证内容、未上传任何东西。

来源研究支持合法图Case A：#246将训练用户按80/10/10分为可见/结构验证/结构测试，canonical多出的62501边来自结构留出；先完成原始逐类证明后注册具名图，保留全部正确旧资产。历史0.3677秒窗口有真实日志，但旧计时排除评估/导出且缺执行源/硬件绑定；不会用它直接通过本轮0.5秒完整墙钟门槛。
