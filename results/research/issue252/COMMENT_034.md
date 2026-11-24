## 正式 Stage2 完成；分层指标已在读取 Stage3 分数前持久化

正式 run `issue252_formal_r2` 已完成 3000 epochs（四卡、global batch 4096、seed 42）。`train_end` 记录完整运行 1979.702 s；checkpoint 选择为 `final_epoch`，没有按测试分数筛选。

Checkpoint SHA256 `0ca5cd32c307460cad7ecd72734538dff2ca3427f64286063ece8c290f8e28f0`；SID SHA256 `037766704565c5143ab721fd5ca5f61bbba3e800baa9dd561d3ae5601987cff2`。独立重新编码 24587 项，0 项不一致（`EXPORT_CHECK_FORMAL.log` / 本 run 的 `export_manifest.json`）。

复用已冻结的 `tools/issue252/layer_metrics.py` 与 `LAYER_METRICS_CONTRACT.md`，没有另造统计定义。报告：`results/stage2_RQ-VAE/TIGER_gnn_attention_sectional_rq/stage2_metrics/037766704565c5143ab721fd5ca5f61bbba3e800baa9dd561d3ae5601987cff2.json`，SHA256 `0afbacf06a90bb1e1b621cc16be7e30de302a25c4ce3e9810b356fe7d8fce658`。

- 三码碰撞率 0.07284337251393014；L2 前缀 9130，L3 三元组 22796；三级占用均为 256。
- 验证集 57439 行；L1 正命中率 0.05329131774578248，合法目录均匀负例的精确期望概率 0.007784630099238625，ER 6.845709695441359。
- L3 refinement `H(C3|C1,C2)` = 2.029564844129057 bits。
- 旧 Conditional G3 的定义实现已删除，仍明确标记 MISSING；另名记录的条件富集不是它的替代品。

修复两个后处理工程故障，不改变训练权重、SID 或冻结训练源码：

1. Stage3 的层指标钩子仍调用 §94 已删除模块；现在只读取并核验 SID/immutable run 绑定的手工报告，缺失明确报告。`LAYER_RECORD_RED.log` → `LAYER_RECORD_GREEN.log`（5 个回归通过）。删除了未产出结果且口径错误的重复 `sid_layer_metrics.py`，仅保留已有受测实现。
2. 全量评估错误地拿工程优化前的 intake manifest 哈希检查正式训练使用的新版本 manifest。改为保留原始数据 intake 哈希不变，manifest 严格使用 checkpoint 训练前已绑定的版本；同时报告两个版本，不能接受训练后的修改。`FINAL_EVALUATION.log` 保留失败；`FINAL_INPUT_RED.log` → `FINAL_INPUT_GREEN.log`（4 个回归通过）。全量评估已从精确绑定的正式 run 启动，不读取 Stage3 成绩。

原 0.5 s 性能目标仍未达到；正式准入窗口实测 0.6688767219893634 s/epoch。放宽至 0.8 s 及只保留第三级 Sinkhorn 的 owner 决策仍以既有 `CONTRACT.md` 最后一节为准，不能把它写成原配方完全不变。Stage3 尚在训练；没有提前宣称最终 Recall、R2 归档或研究主张成立。
