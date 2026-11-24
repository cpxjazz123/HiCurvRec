Infra/归档与隔离节点：109/109既有assets哈希一致；新建资产0、重建0。完整path:sha256/use/coverage与数据/冻结TIGER源码清单见`results/research/issue256/INFRA.json`；manifest SHA `cfbd9d9c0d3c5282f9fdd0c0b28798cbf7a86479c2626b702e499b589acc5b6b`。四张L40S空闲、50201/50202空闲、两解释器存在。

确认无运行依赖后已`git mv`将旧255树归档到`stage2_RQ-VAE/_archive/TIGER_prefix_curvature_gate_rq`，顶层只剩baseline/_archive，55个源码身份保留；冻结_runs未动。仍用路径已同步，结果目录遮蔽回归1 passed。详见`PARENT_ARCHIVE.json`/`RELOCATION_TEST.log`/`INFRA.md`。

固定验证/解码/词表派生与测试隔离见`EVALUATION.md`；`TEST_ACCESS.json`登记历史暴露。没有独立未访问确认集，不把反复使用的test称无偏泛化；不读254具体测试错误清单来选择机制。历史R2编号100/101的199个远端对象已逐个读取并核验SHA一致（`PARENT_R2_AUDIT.json`）；新机制尚未登记或训练。
