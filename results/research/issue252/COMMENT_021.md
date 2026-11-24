## C8：当前源码的完整数学检查通过，真实 Trainer 仍在检查

命令 `/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 -m tools.issue252.run_contract_tests` 退出0：确定性60、独立参考11、变异测试方法2全部通过；15个数值/结构/路由变异均由指定断言捕获，无跳过。报告均绑定当前完整源码，执行前后无漂移：

- `contract_tests/1795477450193264969/deterministic.json:823911921bb4fc6efe9e8da5b83fd59dcd78dee8dbb9a651648cdb4f5130b48e`
- `contract_tests/1795477450193264969/reference.json:82c2b35cc48fab33507cb4255087134bea59f53ca4f47fee458345a76a4c9777`
- `contract_tests/1795477450193264969/mutation.json:04c84c7d7dfd542ae3f93eb9b05119f4940ff6b45385c86c33161af936d70f2d`

完整绑定的四卡 CUDA 检查也退出0：`geometry_validation/geometry_1795477391529808182_565443_6c4b74eb48c9/rank_0.json:562413b6d1e15936f7a79538f4db79f6bb97f9eacad9f14c299155b272d2d760`，四卡各116数值/梯度、16硬选择、18原始floor对照，通过；81个源码/契约/归档输入前后相同，实际包含导入的编译策略。原始CUDA launch数64/64/63，不是编译配置回显。

当前生产文件：`model/geometry.py:62ee1328ec2504e787bdea6e10b1393e055f86d10c271ee311a6af4346dc4c8f`、`model/structure.py:6afb386e5a70718db40235bdde1726323a1dbc41df668399898bec3cdd9651fb`（均在活跃候选树）。正式凭证/不可变快照/禁止单卡绕过等31项生命周期回归实际退出0，`FORMAL_LIFECYCLE_FINAL_TESTS.log:6dc2ea259092bf97a57636113374c02ed004f78a54b7e45e2188869963197493`。

真实四卡 Trainer 一轮已通过标准 launcher 启动检查（不是正式训练），尚未取得完成结果。C8 不因单元/算子测试通过而提前判整项PASS；C7仍等待新一轮包含全部成本的60epoch实测。以上证据相对路径均在 `results/research/issue252/`。