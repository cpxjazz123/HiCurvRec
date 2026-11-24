R02 CHECK_5：8 PASS / 1 FAIL / 8 NOT_RUN，重新读取Issue仍OPEN且正文SHA未变。`CHECK_5.json` sha256 `66943b9d9d6dfe050963a4dfa7d8c5d2a25815ca8d6aefe770b1e89f4290c520` 逐条绑定17条原文、143/64/17完整units、4卡native replay失败及实际模式反例。

这次FAIL是验证器错误重放辅助population的模式；已显式传递捕获的balanced标志，未改生产分配、损失或数值阈值。继续bg_514完整检查，尚不把修复代码或目标回归当作实际四卡/epoch51通过。
