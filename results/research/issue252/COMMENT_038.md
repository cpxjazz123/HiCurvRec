## 公共 C3 已实际通过；最终分析读者的旧 ledger 兼容故障已修复

真实 MCP `verification_check(..., verification_id="machine:issue252_final_reconciliation")` 返回 `passed=true`：

- entry PASS：`source-bound real four-rank Trainer epoch and exact checkpoint SID verified`，绑定真实四卡 validation `issue252_validation_3110912_580007597`；
- behavior_graph PASS：原始 train/split/graph/audit source/training dependencies 核验；
- candidate digest `4aeae9dec8f3cbbf4ee4b78fc10191e09a5d12cc5bf96180c881a334daad7a85`；graph identity `c6fd4a78ef03e1374dcb9d701163e29e9e25237c617fd987733760aaabe0231b`；checks fingerprint `b2877a41335a`。

该版本凭据 `866b804ff83204775db9e57ef52c0d7c5b04d5c188d59382896f84cf30c56f01` 及公共回复 `3a0bb8fc429af7a8251f403738252f8d046b08cd294c8d5d398455ea367e434e` 分别按原字节保存在 admission_history / public_verification_history。此凭据包含真实 1-epoch 四卡验证、60-epoch 四卡窗口（inclusive mean 0.6787061656 s）、191 deterministic + 11 reference + 2 mutation suites（15 个指定变异全部捕获）。这些是事后诊断，不冒充正式开训前的公共工具调用。

独立逐用户复算已经确认四个 test shard 的样本数为 14360/14360/14360/14359；从 target_rank 重算 R@5、R@10、NDCG@5、NDCG@10，与正式记录在 1e-10 容差内一致。R@10 仍为 0.0492870697609638，结论不变。

完整 reconciliation 运行继续暴露一个分析侧缺陷：历史 R2 ledger 的部分旧项没有 key，原过滤器 `entry['key']` 直接 KeyError（保留 `RECONCILIATION.log`）。修复读者仅按显式 key 选当前项，再严格核对完整 SID/checkpoint SHA；旧无关联键记录不能冒充当前来源。新增回归覆盖无 key 历史、缺失、歧义、12位前缀相同但完整 SID 不同、checkpoint 混代；`LEDGER_SELECTION_RED.log` → `LEDGER_SELECTION_GREEN.log`。

因为新回归文件改变全目录凭据绑定，最终源码版本再生成诊断凭据后重新走公共 MCP；不绕过 source-drift 拒绝，不重训正式模型，不改已完成的 SID、checkpoint、Stage3 结果或 R2 bundle。最终文件会保留每代原始证据，而不覆盖过去。
