候选诊断已完成：全57,439样本的候选覆盖表、40,716行错推明细、12个确定规则选取的退化/改善/两对/两错案例均持久化。退化928例中，253的top10有677例无目标L1、233例仅覆盖L1、17例覆盖L2、1例同L3但目标未出现；这是最终输出覆盖，不是T5内部剪枝/概率的因果分类。原始beam轨迹未保存，记NA。

统计执行曾报 `KeyError: results/stage0_build_parquet/train.parquet`：上游train SHA在 `assets/manifest.json.inputs.train`，不是生成资产列表项。已保留失败日志并先加独立回归测试 `test_training_frequency_input_hash_not_asset_entry`，再修复来源读取和真实训练频次计数；只重跑统计，不重跑已成功的配对/结构/门控。没有改变科学判据、模型或数据。
