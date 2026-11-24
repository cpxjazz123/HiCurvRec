## C8 / C11 新 FAIL：真实 Trainer 校准需要保留的图被编译器捐赠

新一轮真实四卡入口命令 `genrec_env_v2/bin/python3.9 -m tools.issue252.launch_validation` 退出1；未执行正式训练。证据 `results/research/issue252/validation/issue252_validation_604437_578075250/rank_0.json`，59.415274秒，仍在初始校准，未完成 epoch、未进行模型挑选。

失败原文：
```
RuntimeError: This backward function was compiled with non-empty donated buffers which requires create_graph=False and retain_graph=False. Please keep backward(create_graph=False, retain_graph=False) across all backward() function calls, or set torch._functorch.config.donated_buffer=False to disable donated buffer.
```
调用为实际 `calibrate_curvature_weight` 的 `autograd.grad(reconstruction, parameters, retain_graph=True)`。此前算子检查只有单次 backward，未覆盖真实消费者的图生命周期，不能替代入口通过。

已先扩展四卡生产 geometry 回归：同一图两次不同目标的 VJP（第一次 retain_graph=True），逐输入与独立 ambient 实现比较；正在观察原编译策略的 RED。将按编译器支持的显式保留缓冲策略修复，不删除真实损失/梯度校准、不吞错误或退回未计时路径。当前81文件完整执行源已保存在 `source_generations/4b6e58f0f9b18d28e9c1d44f000906287fdc617eecf296f281872dbe77d3e0f1/`，归档SHA256 `622d6daaf4a792e287ce07818c74300e71d3ae5a1e6a87f540e9d5dfd2db43e4`；不覆盖之前失败证据。