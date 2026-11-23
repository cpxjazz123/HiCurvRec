## C3 子项：旧缓存与版本漂移回归已红→绿

`python3 -m unittest tools.research.stage2.tests.test_machine_cache -v`：10 tests，OK，exit0，证据 `results/research/issue252/GRAPH_CACHE_TESTS.log`。
新增复现曾实际失败：具名图 A→B 仍复用旧 entry；entry 执行期间图身份漂移仍判 PASS；包装器丢失已验图 identity。红灯保留 `RED_GRAPH_CACHE.log`。修复后 entry 缓存同时绑定源码、检查策略、图代际；检查期间漂移为 NOT_RUN，不入缓存。

第一轮 `PHASE1_TESTS.log` 为42项中41 PASS、1 fixture ERROR（模拟的 PASS 漏写真实返回必有的 detail）；修正测试 fixture，不放宽生产校验。`python3 -m unittest tools.research.stage2.tests.test_named_graph -v`：11 tests，OK，exit0，证据 `NAMED_GRAPH_TESTS.log`。其余41项先前已经执行通过，未把原始 ERROR 隐去。

这些单测不代替实际 verification_check/entry。真实 MCP 准入、性能窗口与正式训练仍未完成。
