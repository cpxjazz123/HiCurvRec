## C7 新 FAIL：真实60epoch完整窗口平均2.2026s，未达0.5s

`genrec_env_v2/bin/python3.9 -m tools.issue252.launch_benchmark`，exit1；实际run `issue252_benchmark_4185524_577609457`，四rank均完成60完整epoch，无正式checkpoint/SID。

rank0原始报告：初始化65.7685s；连续31–60 epoch完整窗口66.078675s，平均2.2026225s/epoch，中位2.1600068s，P952.2846093s，按该窗口3000epoch约6607.9s。真实epoch50全目录评估已包含（该epoch3.4169s），每epoch实际结构更新/采样和四rank覆盖均保留。证据 `results/research/issue252/benchmark/issue252_benchmark_4185524_577609457/rank_0.json`。不能删评估或减batch/epoch来达标；继续定位精确GNN计算/反传热点，不开始正式训练。

同代源绑定数学套件：49项确定性为48PASS/1ERROR；11参考PASS；2变异测试捕获全部15个有效变异。ERROR是缺静态资产fixture原先只接受FileNotFoundError，新增严格字节检查在任何array读取前以ValueError拒绝；保留CONTRACT_TEST_RUN.log，已把该合同测试迁移为实际拒绝类型，不放宽缺资产不可重建的判据。最终源码测试凭据仍需重新产生。

新增分层诊断契约独立于训练：LAYER_METRICS_CONTRACT.md，已有训练/验证数据只用于导出后诊断；未找到被删除的legacy Conditional G3定义时记录MISSING而非伪造。新entropy/uniform-negative边界3个手算测试RED→GREEN，LAYER_METRICS_TESTS.log。这不代表正式结果已存在。