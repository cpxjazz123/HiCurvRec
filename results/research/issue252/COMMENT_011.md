## 独立审查：正式启动和归档仍存在实际缺口

两份独立只读审查指出以下可修复问题，当前不启动正式训练、不宣称准入完成：

1. checkpoint 已绑定 issue252 数学契约，但 archive_audit 仍核对 issue251 路径，正式产物会被拒绝。
2. 静态 CSR loader 仅比对 manifest 声明，训练消费者还需在首次加载前核对所读文件的实际 size/SHA；保留原生成器与所有既有二进制不变。
3. 正式 Stage2 worker 仅执行 entry+graph，尚未强制消费本 issue 的数学/梯度/DDP 和连续30 epoch ≤0.5s 证据；新增显式源绑定凭据门。
4. immutable src 后的 launcher floor patch 可能改写快照；须前置补丁或冻结后 verify-only。
5. Stage3 之前必须生成并验证实际 source/checkpoint/SID/export 的 archive_binding，不能等150 epoch结束才发现缺失。
6. R2 上传须逐个远程读取实际字节核 SHA，并拒绝缺项/多项；仅把本地 SHA 写入 ledger 不算远端验证。

已完成的集成修复：第二次候选全量61项为60 PASS/1测试 fixture ERROR（有效 AST 解析拒绝了无效 Python fixture）；fixture 改为有效 VALUE 赋值后，目标 test_execution.py 2/2 PASS，见 CANDIDATE_TESTS_2.log 与 CANDIDATE_EXECUTION_FIX.log。其他失败日志保留。接下来先记录上述缺陷的 RED，再修实现。