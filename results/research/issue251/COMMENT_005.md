C7 新增协议缺陷已红→绿修复：框架torchrun的`--standalone`会忽略`--master_port=50202`。真实PyTorch参数解析复现 endpoint=`localhost:0`；取消验证的rank0实际记录端口33455（`runtime_cancelled_port/rank_0.json`），未把该运行算通过。

`tools/research/stage2/launcher.py` 改用静态127.0.0.1 rendezvous，保留world4与50202；新增 `tools/research/stage2/tests/test_frozen_port.py` 检查真实PyTorch解析后的有效端口，`RED_PORT.log` FAIL→`PORT_FINAL.log` PASS。现在重新执行同一受控验证，不是正式训练或调参。

独立C8评估也已改为全28,391,948目标、全参考集分别重算硬q和连续z，不用16个中心代替全量。新增手算/独立ambient几何/掩码/宏微归一测试2项PASS（`FULL_EVALUATION_TEST.log`），实际指标仍待checkpoint。
