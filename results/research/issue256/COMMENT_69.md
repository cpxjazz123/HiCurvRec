R03 正式 Stage2 已在推送后的准入源码上启动（run_id `issue256_r03_formal`，arm `TIGER_role_conditioned_l1_rq`，源码提交 `bcf124e6f79cdc3cd178f6f6736250d4be978ff6`，四卡/world4/global4096/seed42/3000 epoch/端口50202，冻结源码快照在 `results/stage2_RQ-VAE/_runs/issue256_r03_formal/TIGER_role_conditioned_l1_rq/src`）。

准入凭据（全部实际执行）：152 candidate + 64 asset + 22 controller/framework 用例无 skip；真实单 epoch 入口导出 SID；行为图契约 PASS；四卡 anchor1/51、17 个耦合模块组、4 个 stale 系数突变全部被断言捕获；inclusive 31–60 均值 **0.6767370127451916s**（0.8s 上限通过，0.4–0.5s 目标未达）；dispatch 绑定探针覆盖 **133** 个精确源文件并退出 0。

对照预注册预测，机制签名已出现：三码碰撞率在 epoch100/200/300 分别为 **0.329/0.426/0.274**，而 R02 终值是 0.0649、混合族最好臂是 0.257——role-conditioned level-1 确实把近乎逐物品的 SID 变粗了。这不是结论：最终判据仍是 validation Recall@10 ≥0.0583 与完整流程，test 只在决策冻结后读一次。
