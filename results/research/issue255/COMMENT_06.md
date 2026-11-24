唯一一次正式 Stage2 已正常结束：`launcher_exit=0`，3000/3000 epoch，四卡 global batch4096、seed42、端口50202；launcher实耗2117.47秒，训练流 `train_end.total_wall_time_s=2103.351`。

- run_id：`issue255_formal`。
- 四个真实 rank PID：2277692/2277693/2277694/2277695。
- 不可变源码：`results/stage2_RQ-VAE/_runs/issue255_formal/TIGER_prefix_curvature_gate_rq/src/`。
- 正式结果：同级 `result/`，含最终 `rqvae_best.pth`、`item_sids.json`、`sids_for_hgrec.npy`、`logs/training_metrics.jsonl`。
- 最终三码碰撞率：**0.16354170903322895 < 0.4**；SID形状 `[24587,4]`，checkpoint选择 `final_epoch`。该门槛通过不是推荐收益结论。
- 最后一轮训练批次平均门控（负/零/正，不冒充全目录测量）：L1 `[0.929897,0.041129,0.028975]`；L2 `[0.004799,0.978047,0.017154]`；L3 `[0.003838,0.735398,0.260764]`。

正继续做最终 checkpoint/SID 逐项重导出、不可变身份与分层指标记录、全套测试与真实逐层损失/四卡诊断。Stage3尚未启动；测试及真实验证完成后，才执行一次冻结150 epoch Stage3。全量三层/两臂 common_fixed3 评估与独立审查也会真实执行。未使用中途checkpoint、没有调参或重训。
