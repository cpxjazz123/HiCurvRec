## C16 评估路径缺陷:归档绑定(绝对路径)与归档导出清单(仓库相对路径)的同代身份比较失败

**现象**:历史 #252 归档的 `archive_binding.json` 记录绝对路径,而同一归档中的 `export_manifest.json` 记录仓库相对路径。`tools/issue253/final_evaluation.py::load_historical_checkpoint` 用字符串相等比较两者,抛出 `historical exact export proof missing`。该缺陷由历史侧干跑在正式评估前发现,不涉及任何训练数值。

**修复**:`final_evaluation.py` 新增 `_identity_path(value, base)`,把仓库相对身份按绑定记录自身的 `repository` 根解析为绝对路径再比较;这与 #252 既有 `load_bound_checkpoint`("相对路径按 `receipt['repository']` 解析")的处理一致,不是放宽校验——两侧身份仍必须指向同一代、同一 SHA。

**TDD**:先写复现测试并确认 RED(`ImportError: cannot import name '_identity_path'`),再实现,再 GREEN:

- `test_relative_and_absolute_identities_resolve_to_one_path`
- `test_resolution_uses_the_bound_repository_root`
- `test_empty_or_non_string_identities_fail_closed`

**连带修复**:`tools/issue252/layer_metrics.py` 的分层 SID 记录原硬编码 #252 的臂名;现由不可变运行工作区派生(`resolve_arm`),新增 3 项回归测试(显式臂名逐字使用、历史路径回退、畸形 `_runs` 路径 fail-closed)。

**证据链处理**:该缺陷只影响训练后的 CPU 评估模块,不改变训练源码与训练数值;但训练准入回执的 `final_sources` 覆盖整个 `tools/issue253/` 与 `tools/issue252/` 清单,且 `reconcile_results._admission` 要求运行凭据与当前回执逐字节一致。因此本次修复之后必须重新跑合同测试、真实四卡验证与基准、重新生成回执并**重新派发正式训练**,以保证回执、运行凭据与归档三者一致。被取代的那次派发在缺陷修复前完成 3000 epoch 但不作为结果报告,只作为启动链路的诊断证据。

合同测试重跑结果:确定性 **181 项**(新增 6 项回归测试)、参考 11 项、变异 2 项,全部通过,`source_hashes` 未变。
