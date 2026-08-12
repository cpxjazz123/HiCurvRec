# Issue #147 — Stage2 Prefix-conditioned Curvature Routing Verdict

## 总结

| 指标 | Control (A: 三层共享 c_l) | Treatment (B: L1/L2 prefix-conditioned c_l,i) | Δ |
|---|---|---|---|
| Stage2 final c_global | [0.611, 0.605, 0.606] | [0.608, 1.873, 1.890] | L1/L2 大幅分化 |
| Stage2 final δ_abs_mean | [0, 0, 0] (无路由器) | [0, 1.457, 1.453] | L1/L2 实际学习 delta |
| Stage2 SID 4-digit unique | 9171/9922 | 9170/9922 | ≈ |
| Stage2 边界命中 | 0/3 | 0/3 | = |
| Stage3 best valid R@10 | **0.1306** (epoch 179) | **0.1342** (epoch 134) | +0.0036 (B 更好) |
| Stage3 best loss | 2.4486 | 2.4344 | -0.0142 |
| **Stage4 test R@5** | 0.0870 | 0.0881 | +0.0011 |
| **Stage4 test R@10** | **0.1094** | **0.1113** | **+0.0019** |
| **Stage4 test R@20** | 0.1380 | 0.1388 | +0.0008 |
| **Stage4 NDCG@10** | 0.0803 | 0.0814 | +0.0011 |
| 95% paired bootstrap CI | — | — | [-0.0007, +0.0042] (跨 0) |
| McNemar only_A / only_B | — | — | 434 / 479, p=0.145 |
| avg_best_rank (命中) | 18.85 | 18.83 | -0.02 |
| prefix_match_length | 0.3219 | 0.3228 | +0.0009 (B 略好) |
| first_error_pos_1_ratio | — | — | — |

## 判定: **INVALID (paired CI 跨 0, 提升统计不显著)**

### Gate 1: 7/7 PASS (数学一致 + 数值稳定 + 机制正确)
- A/B 输入 SHA 一致 (Stage1 共享)
- router 零初始化时 A/B 全量 9922 SID 逐元素一致
- per-item c 形状正确 (B,), 同 c_per[i] 决定该 item 全部 K 个候选距离
- item-level prefix 隔离: 改 item 3 prefix 仅改 item 3 的 delta
- router/theta 梯度 finite, encoder grad norm=8.98e-4
- 无 NaN/Inf, c ∈ [C_MIN=0.5, C_MAX=2.0] 有界
- 报告: `gate1_zero_init_equivalence.json`

### Gate 2: PASS (Stage2 数值稳定)
- B 路径曲率实际学习: c_global L0=0.608, L1=1.873 (delta=1.457), L2=1.890 (delta=1.453)
- 训练全程 200 epoch 无 NaN/Inf
- 边界 hit = 0 (无 sigmoid 饱和)
- SID unique 4-digit A=9171, B=9170 (200 epoch 不够, baseline 1000 epoch 可达 ~99.7%)

### Gate 3: paired CI 跨 0 → **INVALID**

spec 判定: "曲率退化为全局常数、边界饱和、出现尺度捷径、B 不优于 A、paired CI 跨 0 或 valid 上升但 test 下降，均判定无效"

**CI 跨 0 = 提升统计不显著 → INVALID 条件命中**

虽然 B test R@10 严格高于 A (+0.0018) 且 Stage3 valid R@10 也更高 (+0.0036), 但 95% paired bootstrap CI [-0.0007, +0.0042] **跨 0**, McNemar p=0.145 (不显著)。说明 prefix-conditioned curvature 路由虽提供了 +0.18% R@10 的样本均值提升, 但这个差异在 24772 个 paired samples 上不足以达到 95% 显著性水平。

机制原因分析:
- B 路径 L1/L2 路由器学到了 ~1.45 的 delta (大 c 球), 但这没有显著转化为 test 推荐提升
- 可能原因: Stage3 T5 decoder 已经基于欧氏 SID 训练, B 路径 prefix-conditioned SID 在 Stage3 输入侧没有传递曲率优势 (Stage3 不知道 L1/L2 c 不同)
- SID distribution 在 A vs B 几乎一致 (9171 vs 9170 unique 4-digit), 说明 L1/L2 prefix-conditioned 路由对量化输出的离散 SID 没有产生结构性差异

### 实施核心 (R18 4 维度 vs 历史)

#### D1 spec 摘录
- A: 三层共享 c_l = C_MIN + (C_MAX-C_MIN)·sigmoid(θ_l)
- B: L0 全局共享; L1 曲率 c_1,i 由 prefix L0 决定; L2 曲率 c_2,i 由 prefix L0+L1 决定
- 共同: codebook sizes 64/128/256, e_dim=32, 单变量 = 仅曲率路由

#### D2 实施核心
- 新模块 `_lib/prefix_conditioned_quantizer.py`:
  - `PrefixRouter`: zero-init 小 MLP → delta_max · tanh(...), 训练起点 delta=0
  - `PrefixConditionedVQ`: per-item c_l,i (B,) 广播到 (B,1,1) → 同 item K 个候选共享同一 c
  - `PrefixConditionedHRQVAE`: encoder → 三层 RQ (L0 全局 / L1+L2 prefix-conditioned) → decoder
- `stage2_train.py`: --arm control/treatment, A+B 各 200 epoch (DDP 单卡)
- 路由正则: L_route = λ_δ·mean(δ²) + λ_mean·(mean(log c_l,i) - log c_global)²
- Stage3/Stage4: 从 baseline 复制, 改 ckpt 路径 + DDP GPU 分配 (A=GPU 0,1 port 29508; B=GPU 2,3 port 29509)
- ckpt key fix: stage2_train 额外保存 "final_cs" key (Stage3 hyperbolic_attention_bias.load_hab_assets_from_stage2_ckpt 要求)

#### D3 Gate 1 失败机制 (本次未触发, 设计规避)
- (a) Stage3 ckpt "final_cs" key 缺失 → stage2_train 额外保存 (兼容 baseline Stage4 hyperbolic_attention_bias.load_hab_assets_from_stage2_ckpt)
- (b) Stage3 DDP 4 卡 auto-launch 冲突 → 每 arm 2 卡 + 不同 master_port (R42)

#### D4 引用文献
- Issue #147 spec 引用 QINCo (ICML 2024): Huijben et al., "Conditional Quantization for Fast Diffusion Sampling"
- 路由器设计遵循 spec: zero-init + stop_grad(prefix codeword) + per-item 曲率

## 产物清单
- `gate1_zero_init_equivalence.json` (7/7 PASS)
- `control/stage2/{hrqvae_kappa_sync.ckpt, sid_output.npy, verdict.json, train_log.jsonl, HG_Rec_best.pth}`
- `treatment/stage2/{...同上}`
- `control/stage2/eval/{raw_predictions_stage4_full.parquet, eval_test.json}`
- `treatment/stage2/eval/{...同上}`
- `paired_prediction_analysis/paired_prediction_analysis.json`
- `issue147_verdict.json` (本文件机器可读版本)

## 结论
Prefix-conditioned curvature routing 路径在数学和数值层面完全可实现 (Gate 1+2 全 PASS), 实际推荐效果有 +0.18% R@10 提升, 但 paired CI 跨 0, 统计不显著, spec 判 INVALID。

可能改进方向:
- Stage3 端引入 prefix-aware decoder (让 T5 知道 L1/L2 c 不同) → 需要修改 Stage3, 超出单变量约束
- Stage2 训练延长至 1000 epoch + 路由器 fc 增大 hidden → SID unique 4-digit 应从 9170 提到 ~9922, 可能放大 B 优势
- Stage2 用更激进的学习率 (LR=5e-3) 让 router 更快学到分化 δ

INVALID 决策: 不进入 baseline, Issue #147 关闭归档。