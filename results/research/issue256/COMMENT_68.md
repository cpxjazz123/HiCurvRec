R03 已在**任何训练之前**预注册（`results/research/issue256/rounds/r03/CONTRACT.md` sha256 `ba983b1423bffb50fb635f04963e424e6d56b569354e77a32bf6c8c66363d89c`），并登记进 ROUNDS.json（run_id `issue256_r03_formal`，arm `TIGER_role_conditioned_l1_rq`，状态 PREREGISTERED_NOT_RUN）。

单一主要改动：**level-1 的曲率混合不再取自每个物品自由学习的 gate，而取自该物品的 graphlet orbit type**；level 2/3 保持 R02 不变；Sinkhorn 均衡分配、assignment floor、损失、预算标定与刷新节奏、容量、3000/150 冻结协议全部不变。
- 行为信号：训练期行为图算出的 orbit type `r(i)`；
- 曲线运算：三专家混合曲率距离 `d_e`，权重为角色混合 `m_{r(i),e}`；
- 耦合通路：`r(i) → m_{r(i)} → D_1(i,c) → level-1 码 → SID 首 token → Stage3 解码器`；
- 三个开关（BEHAVIOR_OFF / CURVE_OFF / BOTH_ON）写进代码但按政策不训练关闭态。

预注册的可证伪预测（用 validation 选择，test 最后只读一次）：二码前缀数 ≤6000（R02: 8840）、绝对 L1 前缀命中率 ≥0.10（R02: 0.0643）、validation Recall@10 ≥0.0583（R01: 0.05764 + 协议中位重复噪声 0.000949）；失败判据：L1 命中率 <0.08 或 validation Recall@10 ≤0.0576。

同时写明因果限制：R01/R02 带数值修复，所以 R03 vs R02 不是单因素隔离，唯一干净的单因素对照是 R03 vs 冻结基线 TIGER_RQ-VAE。

R02 归档与结论证据均已推送（`fedb8718`、`7e96c64`，本地与远端 main SHA 一致）。Issue 保持 OPEN：关闭条件（一次合法完整的 Stage3 test_R@10 > 0.065）本轮未达成，实测 0.05266456588728913。
