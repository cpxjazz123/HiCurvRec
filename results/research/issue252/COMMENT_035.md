## C3 当前公共准入仍拒绝；不把正式训练完成冒充该项 PASS

重载三个 MCP 子进程后已实际调用 Stage2/Stage3 状态和 genrec verification 工具。Stage3 的现有训练未重启，仍正常推进。

真实调用：`verification_check(source_dir="TIGER_gnn_attention_sectional_rq", verification_id="machine:issue252_final_reconciliation")`。

返回原文：`passed=false; refused="Issue252 training admission: Issue252 training admission: production/source inventory changed"; checks=[]`。

原因：新增/修正最终评估和回归文件后，凭证绑定的全目录 production/test inventory 与当前工作区不同。这是正确的 fail-closed；没有绕过它。正式训练实际使用的凭据原件 SHA256 `e93d46e65ec5ad43d1e216739cc98e12a67de783c44aa12e5767780ef3fdedff` 已按原字节保存在 `admission_history/<SHA>.json`。不可覆盖它后声称新凭证在开训前已经存在。

只找到一份旧公共 conformance 缓存：entry PASS、behavior_graph FAIL，属于图修复前的旧源摘要；不能用作最终 C3 凭据。现有正式开训凭证含真实四卡 1-epoch validation、数学/参考/变异检查及性能窗口，但**不据此伪称公共 machine admission 已在正式派发前通过**。

后续完成当前源版本的真实公共验证，所需诊断与正式3000-epoch实验严格分开；不重复正式训练、不按 Stage3 分数选模型。最终对账将保留该时间顺序偏差，不把事后证据追记为事前证据。C3 目前保持未完成。
