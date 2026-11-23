真实入口OOM的激活保存问题已红→绿：实际训练Criterion、真实32商品batch，保存浮点元素114,910,562→1,980,234；硬输出和Loss严格相同，38个参数梯度与直算在声明容差内一致（`RED_ACTIVATION_MEMORY.log`、`ACTIVATION_MEMORY_FINAL.log`）。未减batch、未截断ego、未抽样参考。

最终源45项候选测试+30项评估/图/端口检查全部PASS（`UNIT_MEMORY_BOUND.log`）；外部EgoStore直接/别名导入不再漏为NOT_APPLICABLE，非法源失败关闭，注释/字符串和无图模型不误判。独立复核 `FinalGradient251` 无代码问题；`FinalAdmission251` 也无代码问题，但明确整体准入被冻结图/规范图合同冲突阻断，绑定见 `INDEPENDENT_REVIEW.json`。

`tools/issue251/audit_graphs.py` 已从原始CSR独立复算两图差集并核对训练来源，证据 `GRAPH_AUDIT.json`。接下来仅做最终源真实入口与四卡重验，不绕过图门槛。
