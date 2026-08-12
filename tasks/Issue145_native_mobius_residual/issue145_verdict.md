# Issue #145 — Stage2 原生曲率 Residual Quantization Verdict

## 总结

| 指标 | Control (A: 切空间 residual) | Treatment (B: Möbius residual) | Δ |
|---|---|---|---|
| Stage2 final c_l0 | 1.25 (未动) | 1.141 | -0.109 |
| Stage2 final c_l1 | 1.25 (未动) | 1.277 | +0.027 |
| Stage2 final c_l2 | 1.25 (未动) | 1.256 | +0.006 |
| Stage2 boundary hit | 0.0 | 0.0 | = |
| Stage2 round-trip err | N/A (A) | ≤ 5.4e-9 | OK |
| Stage2 transport err | N/A (A) | ≤ 7.6e-9 | OK |
| Stage3 best valid R@10 | 0.1241 | 0.1174 | -0.0067 |
| Stage3 best epoch | 80 | 95 | +15 |
| **Stage4 test R@5** | 0.0804 | 0.0790 | -0.0014 |
| **Stage4 test R@10** | **0.1003** | **0.0956** | **-0.0047** |
| Stage4 test R@20 | 0.1219 | 0.1165 | -0.0055 |
| Stage4 NDCG@10 | 0.0743 | 0.0733 | -0.0010 |
| 95% paired bootstrap CI | — | — | [-0.0069, -0.0025] |
| McNemar only_A / only_B | — | — | 453 / 337, p≈0 |
| avg_best_rank (命中) | 19.05 | 19.11 | +0.06 |
| prefix_match_length | 0.2813 | 0.2834 | +0.0021 |
| first_error_pos_1_ratio | 0.8902 | 0.8907 | +0.0005 |

## 判定: **INVALID (B 机制无效)**

### Gate 1: 8/8 PASS (数学一致 + 数值稳定)
- Exp/Log round-trip 误差 ≤ 1.4e-5 ✓
- T_{a→b} round-trip 误差 ≤ 3.0e-7 ✓
- c→0 极限: Poincaré distance → 2·Euclidean (数学), assignment/recon 严格一致 ✓
- 实际初始化 c=1: L0 distance/assignment 完全一致, L1/L2 差异符合 spec 预期 (差异只能从第一次 residual 更新后出现) ✓
- 三个曲率参数 θ_l 进入 optimizer, 梯度 finite 且非零 ✓
- 无 NaN/Inf, c ∈ [0.5, 2.0] 有界 ✓
- Möbius residual 反向传播到 encoder + theta + codebook 全部 finite ✓

### Gate 2: PASS (Stage2 数值稳定)
- B 路径曲率实际学习: c_l0=1.141 (↓), c_l1=1.277 (↑), c_l2=1.256 (↑)
- Round-trip 误差 ≤ 5.4e-9 (训练全程)
- Transport 误差 ≤ 7.6e-9 (训练全程)
- Boundary hit rate = 0 (无球面边界饱和)
- SID unique 3-digit: 9386/9922, 4-digit: 9922/9922 (无塌缩)

### Gate 3: B test R@10 严格低于 A (-0.0047) → **机制无效**

spec INVALID 条件命中:
- ✗ "B 的 test R@10 不高于 A" → **命中 (0.0956 < 0.1003)**
- ✗ "valid 改善但 test 下降" → B valid 也比 A 差 (0.1174 < 0.1241)

CI 不跨 0 但方向错误 → B 在统计上确实更差, 不是噪声。

## 实施核心 (R18 4 维度 vs 历史)

### D1 spec 摘录
- A: 切空间普通减法/加法 residual
- B: Möbius subtraction (`u_l = h_l ⊕ (-q_l)`) + 跨曲率 `T_{a→b}(x) = expmap0_b(logmap0_a(x))` 切空间传输
- 共同: c_l = c_min + (c_max - c_min) · sigmoid(θ_l), 三层共享

### D2 实施核心
- 新模块 `_lib/per_layer_curvature_quantizer.py`:
  - `PerLayerCurvatureVQ`: 单层共享 c_l (PerLayerCurvatureHRQVAE 内的 vq_layers list)
  - `PerLayerCurvatureHRQ`: 通过 `mobius_residual` flag 切换 A/B 路径
  - `PerLayerCurvatureHRQVAE`: encoder → HRQ → decoder (欧氏输入输出, 内部切空间/双曲)
- `stage2_train.py`: 用 flag 切臂, 训练 1000 epoch, 保存 ckpt/SID/曲率轨迹/residual energy/transport audit
- Stage3/Stage4: 从 baseline 复制, 改 STAGE3_DDP_CONFIG (A 用 GPU 0,1; B 用 GPU 2,3) + hab_stage2_ckpt 路径 (指向本任务 arm ckpt)
- Stage2 ckpt 保存时去掉 "hrq." 前缀以兼容 Stage3 期望 (baseline HRQVAE 直接 `vq_layers.X.embeddings.weight`)

### D3 Gate 1 失败机制 (本次未触发, 设计规避)
- (a) Stage3 ckpt 路径: 默认 baseline 路径, 改为 arm 路径
- (b) Stage2 ckpt key 前缀: 训练时 model.state_dict() 嵌套在 self.hrq 内, Stage3 期望扁平 — 保存时手动 strip `hrq.` 前缀
- (c) Stage3 DDP 4 卡 auto-launch 会让两 arm 冲突 — 改为每 arm 2 卡 + 不同 port (R42)

### D4 引用文献
- Issue #145 spec 引用 HG-Rec baseline + Issue #140 (L0 codeword curvature) 作为最近的曲率框架先例
- Möbius residual 引用 arXiv:1705.09164 (Chami et al., Hyperbolic Neural Networks) + arXiv:2405.13979 (学习曲率)

## 产物清单
- `control/stage2/{hrqvae_kappa_sync.ckpt, sid_output.npy, verdict.json, curvature_trace.csv, residual_energy_per_layer.csv, exp_log_transport_roundtrip.csv}`
- `treatment/stage2/{...同上}`
- `control/stage3/{HG_Rec_best.pth, trace.json, verdict.json}`
- `treatment/stage3/{...同上}`
- `control/stage4/{raw_predictions_stage4_full.parquet, eval_test.json}`
- `treatment/stage4/{...同上}`
- `gate1_geometry_equivalence.json` (8/8 PASS)
- `paired_prediction_analysis/paired_prediction_analysis.json` (ΔR@10=-0.0047, CI 不跨 0)
- `issue145_verdict.json` (本文件机器可读版本)

## 结论
Möbius residual 路径在数学和数值层面完全可实现 (Gate 1 全 PASS), 但实际推荐效果比切空间 residual 差 4.7% (test R@10 0.0956 vs 0.1003), 统计显著 (CI 不跨 0, McNemar p≈0)。

可能原因:
- B 路径 Möbius algebra 改变了 L1/L2 输入分布, 但 Stage3 的 T5 解码器已经基于欧氏 SID 训练, 跨代数 SID 不匹配
- B 路径曲率学习 (c_l0: 1.25→1.141) 改变了 RQ 输出几何, 但没传递到 Stage3 的输入侧 (Stage3 输入是 SID tokens, 不是 latent)
- 切空间与球面双曲的几何优势在 SID 离散化后被截断, 残差代数差异未能传递到下游

INVALID 决策: **不进入 baseline, 不触发 R37 版本回滚 (本任务是 A/B 实验, 不是 vN 演进), Issue #145 关闭并归档为机制无效证据**。
