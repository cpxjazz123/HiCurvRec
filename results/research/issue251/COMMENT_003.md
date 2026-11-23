新增 C7/C9 FAIL：独立审查发现启动源码绑定错误、训练 parquet 未校验、最终保存可能绑定执行中途的新源码，以及rank0的8商品校准会推进小batch轮转分配状态。已取消刚启动的验证（无成功微步/正式训练），原始终止日志保存在 `results/research/issue251/runtime_cancelled_review/logs/controlled_validation.log`。

`REVIEW.md` 逐条记录4项；正在执行新增复现测试 `tests/test_execution.py`、`tests/test_provenance.py`，红灯证据写入 `RED_REVIEW.log`。先修复再重跑，未把候选测试绿误记为真实入口或四卡通过。
