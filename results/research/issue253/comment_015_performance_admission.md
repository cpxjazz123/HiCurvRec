## C13/C14 真实四卡性能、门控/裁剪与逐卡诊断(机器准入通过,正式训练已派发)

**性能(与 #252 正式稳态同一口径:连续 60 epoch、epochs 31-60、含 epoch 内全部工作)**
- mean **0.624 s/epoch**、median 0.579、p95(nearest-rank) 0.764;目标 0.4 **未达**、期望 0.5 **未达**、授权上限 0.8 **满足**(`passed=true`)
- 对比 #252 正式稳态 0.64058 s/epoch → **0.974x**(本机制多两个几何专家,每 epoch 时间未增加)
- 对比本轮优化前同一基准 0.912 s/epoch → **0.684x**
- 归因:周期性全量评估 epoch 50 由 **7.407 s → 1.667 s**;去掉 epoch 50 的窗口均值 0.686 → 0.586 s
- 初始化 30.83 s;总墙钟 90.15 s;峰值 allocated 8.85 GB;每 epoch 每 rank 6 个 optimizer step(窗口合计 1440 步)

**C14 门控/裁剪/逐卡诊断**
- 门控实测:population 1024,三专家 per-expert 均值 0.336/0.340/0.325,行和最大误差 1.19e-07;训练后 `gate_mean` = [0.510, 0.330, 0.160] → 门控未坍塌
- 梯度裁剪:epoch 60 `gradient_norm_before_clip` = 0.002996
- 逐卡:4 rank 各 18 optimizer step / 3 epoch,profile 窗口墙钟 1.60–1.61 s,峰值 allocated 8.57–8.65 GB,通信 CUDA stream span 0.047–0.054 s;每 rank 每次 token 调用唯一商品 ~1270,1-hop ~28k 节点/~100k 边,2-hop ~2.0M 节点/~27M 边;`rank_local_populations_preserved=true`

**准入**
- 合同测试:确定性 175 项、参考 11 项、变异 2 项全部通过,`source_hashes` 与冻结清单未变
- machine admission:`entry` PASS(source-bound 真实四卡 epoch + checkpoint SID 精确导出)、`behavior_graph` PASS
- 训练准入回执 `results/research/issue253/TRAINING_ADMISSION.json` 已生成并通过 `check_receipt`

正式训练已派发:dispatch `stage2_6e4324cd326e45eba067d30295f8e4bd`,4 卡 DDP、global batch 4096、seed 42、3000 epoch。
