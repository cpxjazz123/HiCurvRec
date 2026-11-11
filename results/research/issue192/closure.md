# Issue #192 关闭依据

机制：A（单次组合臂）——R48 的**源-目标对对比项** + `struct_mixed` 的**欧氏三元组结构项**，两项同时在线（`PAIR_OFF`/`STRUCT_OFF` 各自可取关），λ 由仓库既有 `calibrate_lambda_block_budget` 标定，仅一次正式训练（seed=42、3000 epoch、四卡、冻结全局 batch 4096）。

依据 commit：
- `562b55b90` 合并接线（combo 树创建、两项装入同一 trainer）
- `09029710` 冒烟派发与依据留痕、否决路径
- `551d4053b` `forward_with_codes` 补齐
- `4230edc7e` `VQLayer.quantize_with_raw`
- `631473052` 位置改原始码字和（码本可导）
- `ce85114b4` 双通路位置（编码器与码本同时可导）
- `e3cb5d190` 冒烟 PASS、λ = 2.56423e-07、入训
- 本次提交：Stage2 完成、§63 分层指标记录、§89 门槛拒绝、本文件

验收对照（Issue 正文的三条）：
1. 机制实现并训练 —— **是**，3000 epoch 完成并导出 SID。
2. 代码验证 —— 冒烟 PASS，两块结构梯度实测非零，λ 由冻结规则标定。
3. Stage3 —— **未启动**：`collision = 0.5043722292 ≥ 0.40`，§89 门拒绝（实测 MCP refusal 已记录）。

结论：本臂组合形式不降低三级碰撞，Stage3 `test_R@10` = `NOT_RUN`；该线以实测数字关闭，不做调参、不做多种子、不改 λ 重跑。
