四卡三专家原serial-kernel/原双分支profiling完成：run `issue253_steady_profile_3781546_580856978`，真实workers [3782958, 3782959, 3782960, 3782961]；四rank各完成3epoch/18optimizer steps，epoch2完整插桩6steps，source前后不变。

插桩epoch2 wall 1.456435–1.460536s；未插桩epoch3 0.648966–0.654901s。后者只有1epoch，不能冒充≥30epoch性能验收。每rank该插桩epoch处理完整2-hop约11.92–12.02M展开节点/165.52–166.07M边；详细CPU/CUDA、分支/层/内存见 `BASELINE_PROFILE.json` 及其四rank原件SHA。嵌套span不求和。

已派发60epoch实际基准与四卡CUDA独立数值检查序列，尚无最终回包；正式Stage2 3000/Stage3 150未启动。独立最终评估契约/旧回归42tests PASS，`EVALUATION_GREEN.log`；inclusive中位/P95漏算interepoch成本的回归已RED→GREEN，`PERFORMANCE_RED.log`/`PERFORMANCE_POLICY_GREEN.log`，阈值未放宽。
