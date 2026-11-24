新增 C12 工程FAIL已复现：`python3 -m unittest tools.issue253.test_admission_policy -v`，2个测试、1个指定断言失败，exit=1。`training_admission.required(new_arm)` 返回False，导致新三专家树落入旧单卡入口检查，而不是#252已验证的source-bound四rank证据路径；现有模块还硬编码5个Query及issue252证据位置。

证据：`results/research/issue253/ADMISSION_POLICY_RED.log`、`tools/issue253/test_admission_policy.py`。修复保持四rank、实际loss/更新/开关/性能与源码哈希门槛，不放宽准入；将独立候选身份、专家数及本issue证据路径显式登记。之后需要公开MCP实测和相应服务重载，不绕过拒绝直接派发。
