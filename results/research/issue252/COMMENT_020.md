## C8 / C11：实际校准编译故障已定位，尚未替代真实入口验证

真实一轮入口曾在 `calibrate_curvature_weight → sampled_curvature → fixed_squared` 的 28×1×32 配对布局失败；失败产物保留于 `validation/issue252_validation_379303_577903371/`。仅压平单例轴仍失败，原生 Triton 日志定位为 `logaddexp` 分支保存谓词的 SSA dominance 错误，不是 OOM。

生产有限输入改用恒等式 `max(a,b) + log1p(exp(-abs(a-b)))`，不改曲率、维度、权重、dtype、估计器或分配规则。命令 `genrec_env_v2/bin/python3.9 -m tools.issue252.launch_geometry_validation` 实际四卡退出0；证据 `geometry_validation/geometry_1795476629191404831_467881_b5841f2166ec/rank_{0,1,2,3}.json`：每卡116项数值/梯度对照（含1、28、32行配对），16项硬选择、18项原始 floor 对照均通过；9次实测 geometry forward/backward 的 CUDA compute kernel 数为64/64/63（原路径551）。这些是算子证据，不是 epoch 达标或真实 Trainer 通过。

另发现旧 geometry 报告仅列20个精选源码，遗漏导入的编译策略。新增行为回归：只改被导入的 `cuda_math.py` 时，旧 inventory 不变，指定断言 RED；改为复用正式准入的完整源码/import 闭包后4项驱动测试 GREEN。记录 `GEOMETRY_SOURCE_BINDING_RED.log` / `GEOMETRY_SOURCE_BINDING_GREEN.log`。正在重跑完整绑定的四卡算子验证及全数学套件，随后执行真实 Trainer、60个完整 epoch 性能验证。现有 C7 的2.2026225秒失败仍有效；没有正式训练启动。

所有相对证据路径均位于 `results/research/issue252/`。