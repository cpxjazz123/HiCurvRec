C9 准入与性能通过，正式 Stage2 已启动。

**机器准入**（`tools/issue255/admission.py` → `results/research/issue255/ADMISSION.json`）：`entry` PASS —— 真实入口在单卡契约下跑通1 epoch并导出标识（levels 256/256/256，三码碰撞 0.051165，远低于 Stage3 门槛0.40）；`behavior_graph` PASS —— 训练可见图、split、审计源与训练依赖核验通过。两次准入失败均为实现缺陷且已修复并记录：方向ego资产最初未按命名图登记（graph_name/graph_sha256 缺失）→ 登记；注册项缺 source_hashes → 补齐并使本资产注册可自刷新（不触碰他人资产条目）。

**四卡性能**（`results/research/issue255/benchmark/issue255_benchmark_*/rank_*.json`，四rank均 status=pass）：连续30个稳态 inclusive epoch（第31–60），mean 0.687462 s、median 0.651593 s、P95 0.837740 s（nearest-rank）、窗口合计20.6239 s、epoch间额外0.04996 s；初始化27.374 s 单列；第50 epoch的周期全目录评估已含在窗口内；显存峰值见profile；按0.6875 s推算3000 epoch约2062 s。**授权硬上限0.8 s/epoch 达成**；**目标0.4 s与期望≤0.5 s 未达成**（#253为0.6396 s，本臂因四类token与逐层结构项约+7.5%），如实报告，未以删2-hop、冻结GNN、减采样或隐藏工作达标。

**四卡逐模块剖析**（`results/research/issue255/steady_profile/*/rank_*.json`）：profiled epoch2，含全部结构计算；四类token均在实测路径（GNN_1hop/2hop/3hop/4hop即一/二跳与入/出两个方向token）；production_forward 1.6428 s/6步、full_ego_tokens 0.6588 s、backward 0.5098 s、assignment_guards 0.1931 s、all_reduce 0.0366 s。该剖析自带声明不做速度验收，验收以60-epoch窗口为准。

**正式运行**：唯一一次四卡 Stage2，3000 epoch、全局batch4096、seed42、端口50202，经 `tools/research/stage2/launcher.py` 从不可变快照 `results/stage2_RQ-VAE/_runs/issue255_formal/TIGER_prefix_curvature_gate_rq/` 启动；不重训TIGER、不建第二个臂、不追分调参、无多种子。训练中每epoch记录总损失/重建/结构项、裁剪前梯度范数、每层三专家门控均值、门控熵、注意力熵、每层码本梯度范数、level_scale_calibration 与 curvature_calibration（含重建与结构项梯度余弦）。

剩余工作：正式训练完成→checkpoint导出与SID逐项复核→三码碰撞<0.4→一次冻结Stage3（150 epoch，beam20，final test）→机制指标与逐用户配对→报告/VERDICT/R2归档→推送与关闭。未完成前不关闭issue。
