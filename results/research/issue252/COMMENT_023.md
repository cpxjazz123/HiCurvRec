## C5 PASS；C8 的真实四卡入口、实际 Loss 梯度/更新、DDP 与同 checkpoint 导出通过

实际命令 `genrec_env_v2/bin/python3.9 -m tools.issue252.launch_validation` 退出0，完成完整1个诊断 epoch，不是正式训练。run_id `issue252_validation_642254_578105758`；四卡各首批1024，global4096，冻结3000epoch调度保持不变。

- `validation/issue252_validation_642254_578105758/rank_0.json:b4a273eccf83745700a73d3001f62969a273b18beb27dfba4aad527b92aad9c1`，其余3卡均PASS，实际完成后参数SHA均为 `2ba2d851c67a4c75bc4905b389dcb64020ed4c92505cb4ccfe7ca44811f02fce`。
- 20个真实参数组逐卡记录：Encoder、Decoder、GNN各层、五Query、共享K/V/score、三级码本。实际训练总Loss的梯度与实际AdamW更新均有限且非零；另以真实原始/结构Loss分别做VJP，结构Loss不达Decoder。`MODULE_CHECKS.json:ed2dc3c7d055b2d1e40d2a01917cd9ffef5c086cf1da1175cba42a26ba466212` 保存79文件源码绑定、完整forward观察和每模块数值。
- 真正改写Query后四卡分别有71/37/68/40个商品的硬码改变，并改变真实代价，不是旁路标志。
- 24587个商品从该诊断checkpoint重新加载/编码严格一致；SID SHA `bac17b6c53e1ab85fcdb8156e19af985e7d457e4bbd905128a23b76d41ad70ce`。这是诊断导出，绝不送Stage3或冒充最终SID。

C5 的静态完整CSR/特征均值、批量读取、共享距离/采样计算均不缓存学习输出，真实GNN/门控更新已证实。原图/全量28391948目标仍复用，不重建。具体源码与数值/梯度对照见上述绑定和现有lossless/static/sparse测试。

编译保留图故障也 RED→GREEN：新四卡geometry回归每卡130项数值/梯度（含14项保留图双VJP）、16硬选码、18原floor对照通过，`geometry_validation/geometry_1795477867061548663_628781_434a38a67d78/rank_0.json:cb61a97afaa507a8257d208929e1ca9b696359b989eeb1590eb38e9e7c3889fd`。修复策略 `tools/issue252/cuda_math.py:7e3e8d54f89eb64836d3b730bf30c44618c3e98f07952dd41582d1082c1c800b`。

当前完整数学套件正在重新生成最终源码凭证；之后执行60个完整epoch计时。C7未达标状态不变；实际 public verification_check / 正式准入仍待性能凭证，未开始正式Stage2。证据相对路径均在 `results/research/issue252/`。