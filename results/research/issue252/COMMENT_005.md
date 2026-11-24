## 新 FAIL：静态布局元数据生成器错误（C11）

命令 `/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 -m tools.issue252.runtime_layout`，exit1，原文证据 `results/research/issue252/RUNTIME_LAYOUT_BUILD.log`。

失败位置 `_metadata -> _array_description(blob[key])`，输出：`ValueError: Object arrays cannot be loaded when allow_pickle=False`。复用的既有 node-feature NPZ 包含 object 类型的描述字段；生成器不应为描述 shape/dtype 而解码该字段，更不能打开 pickle 来绕过。

先前17项布局/最终评估确定性测试实际全部 PASS（`LAYOUT_EVALUATION_TESTS.log`），但未覆盖这个真实元数据格式。六份新布局二进制已经生成，元数据/全量审计尚未签发；不算通过，不登记为可训练资产。保留失败日志及二进制，下一步补安全 NPZ header 回归、修复描述器，再逐项核对既有布局后恢复元数据；不重建原始图/监督，不把缓存失败写成外部阻塞。
