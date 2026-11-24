## C2 PASS：具名可见图契约真实生效

按 owner #252 Case A 授权，已修订 CLAUDE §89c、资产说明、具名校验器，并将两份不可变审计/契约元数据登记根 manifest。68项原有资产叶子及二进制保持不变；候选显式声明 `BEHAVIOR_GRAPH_NAME = "infra246-visible-v2"`，不是按臂名放行。

实际系统 Python 调用 `tools.research.stage2.checks.check_behavior_graph('stage2_RQ-VAE/TIGER_gnn_attention_sectional_rq')`：exit0、`status=PASS`，图代际 `d61a4ca54f6e7ea29331652aceae1b0d9495e215c44fb8415ae507ed52e38687`。原始输出 `results/research/issue252/NAMED_GRAPH_ADMISSION.json`；该代际绑定 `assets/v7_graph_contract/contract.json` 同SHA和其审计源码/输入/隔离/训练资产全部叶子。

图错、混图、评估输入、源码变更、新缓存无训练祖先、旧 entry 代际和执行中漂移均有已执行回归。当前新增运行布局仍因描述字段错误等待恢复/全量审计，未登记为训练资产。C3 的真实 verification_check + entry 尚未 PASS，正式训练未派发。

已推送的前置源码/红绿证据提交 `521c5e22868f181b3d5283c0cd29cae91b13a9e9`，当时本地/远端 main 同SHA；本节点的 manifest/候选声明将在下一次集成提交保存。
