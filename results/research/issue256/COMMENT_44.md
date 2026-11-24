R02 验证器修复的窄绿测已通过：2/2手算/反例回归；源码身份/完整unit覆盖/完整校准台账12/12。模型、真实loss、硬分配和原有数值容差均未改变。原生不同float32批次下的离散ID差异会完整记录，不再错误声称贪心目标最优值等价；同一实际hard分支上的独立GNN loss/VJP、真实规则精确重放及全目录硬导出仍需分别验证。

`GREEDY_ORACLE_GREEN_RECEIPT.json` sha256 `9c00dfecfaaf954096aa8ec7daadc34f9a53996b5be5c6b456f0b15651aef348`；`CONTROL_GREEN_RECEIPT.json`与VERIFIER_REPAIR.md同目录。新的完整流程正在执行（bg_505），尚未宣称整体验收PASS。

此外，unit suite中的CUDA kernel测试现统一走唯一四卡launcher；每个实际收集用例只执行一次，四rank逐例清单必须无漏项/重复/skip地覆盖完整suite。它不冒充DDP训练证明；后续真实Trainer四卡模块、epoch51、inclusive性能与机器入口仍完整执行。
