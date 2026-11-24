## C11 修复 PASS：元数据恢复不重建二进制

真实 object-NPZ 描述错误补回归后先红再绿：`METADATA_UNIT_TESTS.log` 的8项布局测试+1项 profiler 单位测试，9 tests OK、exit0。元数据现在仅安全读取 NPZ headers，不解码 object/pickle；没有启用 allow_pickle。

恢复命令 `python3.9 -m results.research.issue252.recover_layout`，exit0，`RUNTIME_LAYOUT_RECOVERY.log`：六份既成布局的SHA前后相同，binary_regenerated=false。全量审核24587 directed rows、24587完整2-hop populations/root positions、28391948原始有效 triple 的 b/c/float64 bits/顺序/重复，均 PASS；261个零有效行中心明确保留。没有重算原始图或曲率目标。

证据 `assets/v8_runtime_layout/metadata.json:116068365f7acb74ebb0f4aa0dcfa9249a2fbe968b0721371c8a2d7505caff60`，`audit.json:74188af783ae98f97b38dabedffc5e82bc181dc61cd1a5ff155dc0df47afeb91`；原失败生成源码也保存，未覆盖旧证据。布局待根 manifest 登记后才可用于真实训练。

另外纠正了 profiler 原生产器把 native device 微秒误标为秒的报告错误，新增明确单位转换回归。原始4-rank报告/源码不改，collector显式除1e6；没有重跑成功的训练步来掩盖数据。原 profile 仅记录 rank allocator peak，未记录独立 phase peak；报告保留该测量限制，不能编造逐阶段显存峰值。
