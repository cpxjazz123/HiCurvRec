## C1 PASS：原始交互重建与隔离对账

执行 `/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 -m tools.issue252.graph_audit`，CPU 审计 PASS；无训练、无二进制重建。

- canonical 323633 有向边，visible 261132；差62501 = struct_val 独有31100 + struct_test 独有31241 + 二者共有160。unexplained=0、visible_only=0。
- 整数转移权重：339519 = 272244 + 33559 + 33716；weight_residual=0。held-out 用户落在已有 visible 边上的权重4461亦单独核对，未忽略权重泄漏。
- 57439 用户的确定性划分45951/5744/5744互斥完整；推荐未来交互未进入构图。十份 strict 结构评估 bank 与训练 pair 重叠均0。
- 全24587商品的冻结 ego 节点/诱导边、可见特征与曲率矩度数均已核对。visible 正确；因此适用 issue 授权的 Case A，不重建正确图和监督资产。

生产审计源码：`tools/issue252/graph_audit.py:22a0f91b2ba8282ad84d1c3c6a6ab32f0c52baacc5f00f5ea3b12248d80ddd67`。
不可变审计证据：`assets/v7_graph_contract/audit.json:0918afe35cddf209441fff4fe80a558a2d562e14027326d124ec2f51000acb1b`。
具名契约：`assets/v7_graph_contract/contract.json:d61a4ca54f6e7ea29331652aceae1b0d9495e215c44fb8415ae507ed52e38687`。
原有68项资产仅按叶子哈希复用；新增仅这两份元数据，尚待根 manifest 登记。C2/C3 不以本节点替代真实准入。
