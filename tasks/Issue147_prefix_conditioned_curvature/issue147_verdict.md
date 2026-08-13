# Issue #147 — Stage2 Prefix-conditioned Curvature Routing Verdict (重跑 2026-08-13)

## 总结 (重跑, 与首次运行并列)

| 指标 | Control (A: 三层共享 c_l) | Treatment (B: prefix-conditioned c_l,i) | Δ |
|---|---|---|---|
| Stage2 final c_global | [0.609, 0.605, 0.610] | [0.607, 1.875, 1.892] | L1/L2 大幅分化 (复现) |
| Stage2 final δ_abs_mean | [0, 0, 0] (无路由器) | [0, 1.456, 1.451] | 路由器实际学习 delta (复现) |
| Stage2 SID 4-digit unique | 9198/9922 | 9174/9922 | ≈ |
| Stage3 best valid R@10 | **0.1328** (epoch 74) | **0.1303** (epoch 84) | -0.0025 (本次 A 高) |
| **Stage4 test R@5** | 0.0825 | 0.0849 | +0.0024 |
| **Stage4 test R@10** | **0.1046** | **0.1068** | **+0.0022** |
| **Stage4 test R@20** | 0.1354 | 0.1353 | -0.0001 |
| **Stage4 NDCG@10** | 0.0765 | 0.0786 | +0.0021 |
| 95% paired bootstrap CI | — | — | [-0.0001, +0.0046] (跨 0) |
| McNemar only_A / only_B | — | — | 407 / 462, p=0.067 |
| avg_best_rank (命中) | 18.93 | 18.90 | -0.03 |
| prefix_match_length | 0.3170 | 0.3219 | +0.0049 (B 略好) |

## 判定: **INVALID (paired CI 跨 0, 提升统计不显著) — 重跑稳定复现**

### 两次独立运行对比

| 运行 | A test R@10 | B test R@10 | Δ | 95% CI | McNemar p | valid A / B |
|---|---|---|---|---|---|---|
| 首次 (08-13 02:48) | 0.1094 | 0.1113 | +0.0018 | [-0.0007, +0.0042] | 0.145 | 0.1306 / 0.1342 |
| **重跑 (08-13 14:10)** | 0.1046 | 0.1068 | **+0.0022** | [-0.0001, +0.0046] | 0.067 | 0.1328 / 0.1303 |

两次独立全流程运行均得到: **B 样本均值更高 (+0.0018 / +0.0022), 但 paired CI 均跨 0**, 统计不显著。valid R@10 方向在两次运行间翻转 (首次 B 高, 重跑 A 高), 进一步佐证差异属于随机噪声, 而非机制性增益。INVALID 判定稳定复现。

### Gate 1: 7/7 PASS (与首次运行一致, 机制等价性)
- A/B 输入 SHA 一致 (Stage1 共享, 重跑 SHA=1a6dd2ac... 与首次逐字节一致)
- router 零初始化时 A/B 全量 9922 SID 逐元素一致
- per-item c 形状正确, 同 c_per[i] 决定该 item 全部 K 个候选距离
- item-level prefix 隔离: 改 item prefix 仅改该 item 的 delta
- router/theta 梯度 finite, 无 NaN/Inf, c ∈ [C_MIN=0.5, C_MAX=2.0] 有界
- 报告: `gate1_zero_init_equivalence.json` (与重跑无关, 机制性验证)

### Gate 2: PASS (重跑 Stage2 数值稳定, 与首次一致)
- B 路径曲率实际学习: c_global L0=0.607, L1=1.875 (delta=1.456), L2=1.892 (delta=1.451)
  - 首次: L0=0.608, L1=1.873 (δ=1.457), L2=1.890 (δ=1.453) — 几乎逐位复现
- 200 epoch 无 NaN/Inf, 边界 hit = 0
- SID unique 4-digit A=9198, B=9174 (首次 A=9171, B=9170, ≈)

### Gate 3: paired CI 跨 0 → **INVALID** (重跑复现)

spec 判定: "曲率退化为全局常数、边界饱和、出现尺度捷径、B 不优于 A、paired CI 跨 0 或 valid 上升但 test 下降, 均判定无效"

重跑与首次一致: B test R@10 严格高于 A (+0.0022), 但 95% paired bootstrap CI [-0.0001, +0.0046] **跨 0**, McNemar p=0.067 (不显著)。说明 prefix-conditioned curvature 路由的 +0.22% R@10 样本均值提升在 24772 个 paired samples 上不足以达到 95% 显著性水平。

机制原因分析 (同首次):
- B 路径 L1/L2 路由器学到 ~1.45 的 delta (大 c 球), 但这没有显著转化为 test 推荐提升
- Stage3 T5 decoder 基于欧氏 SID 训练, B 路径 prefix-conditioned SID 在 Stage3 输入侧没有传递曲率优势 (Stage3 不知道 L1/L2 c 不同)
- SID distribution 在 A vs B 几乎一致 (9198 vs 9174 unique 4-digit), L1/L2 prefix-conditioned 路由未对量化输出的离散 SID 产生结构性差异
- 重跑中 valid 方向翻转进一步确认: Stage2 路由学到的东西在 downstream 不产生稳定信号

### 实施核心 (R18 4 维度 vs 历史, 同首次)

#### D1 spec 摘录
- A: 三层共享 c_l = C_MIN + (C_MAX-C_MIN)·sigmoid(θ_l)
- B: L0 全局共享; L1 曲率 c_1,i 由 prefix L0 决定; L2 曲率 c_2,i 由 prefix L0+L1 决定
- 共同: codebook sizes 64/128/256, e_dim=32, 单变量 = 仅曲率路由

#### D2 实施核心
- 新模块 `_lib/prefix_conditioned_quantizer.py`:
  - `PrefixRouter`: zero-init 小 MLP → delta_max · tanh(...), 训练起点 delta=0
  - `PrefixConditionedVQ`: per-item c_l,i (B,) 广播到 (B,1,1) → 同 item K 个候选共享同一 c
  - `PrefixConditionedHRQVAE`: encoder → 三层 RQ (L0 全局 / L1+L2 prefix-conditioned) → decoder
- `stage2_train.py`: --arm control/treatment, A+B 各 200 epoch (DDP 单卡, 重跑 A=GPU0 B=GPU1 并行)
- 路由正则: L_route = λ_δ·mean(δ²) + λ_mean·(mean(log c_l,i) - log c_global)²
- Stage3/Stage4: 从 baseline 复制, 改 ckpt 路径 + DDP GPU 分配 (A=GPU 0,1 port 29508; B=GPU 2,3 port 29509)
- ckpt key fix: stage2_train 额外保存 "final_cs" key (Stage4 hyperbolic_attention_bias.load_hab_assets_from_stage2_ckpt 要求)

#### D3 Gate 1 失败机制 (未触发, 设计规避)
- (a) Stage3 ckpt "final_cs" key 缺失 → stage2_train 额外保存
- (b) Stage3 DDP 4 卡 auto-launch 冲突 → 每 arm 2 卡 + 不同 master_port (R42)

#### D4 引用文献
- Issue #147 spec 引用 QINCo (ICML 2024): Huijben et al., "Conditional Quantization for Fast Diffusion Sampling"
- 路由器设计遵循 spec: zero-init + stop_grad(prefix codeword) + per-item 曲率

## 产物清单 (重跑覆盖)
- `gate1_zero_init_equivalence.json` (7/7 PASS, 机制验证, 未重跑)
- `control/stage2/{hrqvae_kappa_sync.ckpt, sid_output.npy, verdict.json, train_log.jsonl, HG_Rec_best.pth, trace.json}`
- `treatment/stage2/{同上}`
- `control/stage2/eval/{raw_predictions_stage4_full.parquet, eval_test.json}`
- `treatment/stage2/eval/{同上}`
- `paired_prediction_analysis/paired_prediction_analysis.json` (重跑版本)
- `issue147_verdict.json` (本文件机器可读版本, 含 rerun_consistency 两次对比)

## 结论
Prefix-conditioned curvature routing 在两次独立全流程运行中均给出方向一致的 +0.18%/+0.22% R@10 均值提升, 但 paired CI 均跨 0、valid 方向翻转, 统计不显著, spec 判 **INVALID** 稳定复现。结论成立: 该机制在 Stage3 输入侧无法传递曲率优势, 不进入 baseline, Issue #147 关闭归档。
