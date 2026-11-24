## 新 FAIL：集成候选全量61项检查

`python3.9 -m unittest discover -s stage2_RQ-VAE/TIGER_gnn_attention_sectional_rq/tests -v`：59 PASS、1 FAIL、1 ERROR，exit1；保留 `CANDIDATE_TESTS_1.log`，不视为准入通过。

1. `test_actual_launcher_changes_invalidate_source_evidence`：旧测试仍把 ROOT/源绑定接口挂在 validation_entry，重构后实际绑定函数在 benchmark_entry；需迁移到真实生产函数并继续验证 launcher 源码变化使凭据失效，不补兼容别名/不删除判据。
2. `test_initial_feature_cache_real_root`：提供冻结特征根的 synthetic EgoStore 在静态布局打包时又调用 node_features，触发 `AssertionError: refit`。冻结特征已经读过，拓扑打包不应再次拟合/读取；修复该实际边界路径，不放宽断言。

同代优化代码的四卡诊断已真实完成1个1024-item/rank优化器步并返回4份 PROFILED；这不替代全量测试、30-epoch性能或正式3000epoch。当前继续修复上述集成失败。
