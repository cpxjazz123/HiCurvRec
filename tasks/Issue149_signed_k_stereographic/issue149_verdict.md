# Issue #149 — Stage2 Signed κ-Stereographic Quantizer Verdict

## 总结

| 指标 | Control (A: 标准 Poincaré c_l ∈ [C_MIN, C_MAX]) | Treatment (B: signed κ-stereographic κ_l ∈ (-κ_max, +κ_max)) | Δ |
|---|---|---|---|
| Stage2 final c/κ | c=[0.5041, 0.503, 0.5032] | κ=[0.0004, 0.0001, -0.0004] (collapsed to ~0) | B 坍缩到 Euclidean |
| Stage2 SID 3-digit unique | 9091/9922 | 9106/9922 | +15 |
| Stage3 best valid R@10 | **0.1292** (epoch 94) | **0.1261** (epoch 79, early stop) | -0.0031 (B 略差) |
| **Stage4 test R@5** | 0.0830 | 0.0828 | -0.0002 |
| **Stage4 test R@10** | **0.1058** | **0.1049** | **-0.0009** |
| **Stage4 test R@20** | 0.1322 | 0.1308 | -0.0014 |
| **Stage4 NDCG@10** | 0.0770 | 0.0760 | -0.0010 |
| 95% paired bootstrap CI | — | — | [-0.0031, +0.0013] (跨 0 + 方向错) |
| McNemar only_A / only_B | — | — | 400 / 376, p=0.4090 |
| prefix_match_length | 0.3105 | 0.3068 | -0.0037 (B 略差) |

## 判定: **INVALID (B 不优于 A, paired CI 跨 0, 方向错误)**

### Gate 1: 7/7 PASS (机制 + 数值稳定)
- A/B Stage1 输入 SHA 一致 (共享)
- SID 结构 3 层 × 64/128/256 一致
- A 距离 32D Poincaré with c_l ∈ [0.5041, 0.503, 0.5032]
- B 距离 signed κ-stereographic with κ_init=-1.25 → diff vs A c=1.25 = 2.59e-04 (数学严格等价)
- 同 init 同 input 下, A vs B 三层 SID 差异 0/0/0 (init 几何严格等价)
- encoder + theta 梯度 finite, κ 边界有界 (-κ_max < κ < +κ_max)
- 100 步训练无 NaN/Inf, A/B 损失稳定下降

### Gate 2: PASS (Stage2 数值稳定, 但 κ 坍缩到 ~0)
- 训练全程 200 epoch 无 NaN/Inf
- 边界 hit = 0
- SID unique 3-digit A=9091, B=9106
- **关键发现**: B 训练期 κ 在 epoch 20 后迅速坍缩到 ~0 (Euclidean-like), 训练优化发现 Euclidean 几何最优
  - epoch 0: κ=-1.24
  - epoch 10: κ=-0.68 (L0), -1.08 (L1), -1.17 (L2)
  - epoch 20: κ≈-6e-5, 1.5e-4, 7e-4 (全部坍缩到 0)
  - epoch 199: κ=[3.9e-4, 1.1e-4, -3.8e-4]

### Gate 3: B 不优于 A → **INVALID**

spec 判定: "曲率退化为全局常数、边界饱和、出现尺度捷径、B 不优于 A、paired CI 跨 0 或 valid 上升但 test 下降，均判定无效"

**B 不优于 A + paired CI 跨 0 + B κ 坍缩到 0 → INVALID 三条件命中**

虽然 paired CI 跨 0 (统计不显著), 但 Δ=-0.0009 方向错误 (B < A), B 在 R@5 (-0.0002), R@20 (-0.0014), NDCG@10 (-0.0010), prefix_match_length (-0.0037) 上都劣于 A。

最关键的是 B 训练期 κ 坍缩到 ~0, 说明:
1. signed κ-stereographic 路径在 Stage2 训练中没有为几何带来额外优势 (优化发现 Euclidean 几何最优)
2. Stage3/4 HAB 必须 c>0 (Poincaré 球), B 的 |κ|≈0 经 floor 0.01 替代后, 实际效果是 "near-Euclidean" HAB, 不如 A 的标准 Poincaré c=0.5

### 实施核心 (R18 4 维度 vs 历史)

#### D1 spec 摘录
- A: 32D Poincaré 距离 d_A(i, b) = d_H(z_l,i, e_l,b; c_l), c_l ∈ [0.5, 2.0]
- B: signed κ-stereographic 距离, κ_l ∈ (-2, +2). κ<0 双曲 / κ=0 欧氏 / κ>0 球面, 通过 σ(20·κ) 平滑切换
- 共: codebook 64/128/256, total dim 32, SID length 3, 单变量 = 仅曲率参数化

#### D2 实施核心
- 新模块 `_lib/signed_kappa_quantizer.py`:
  - `SignedKappaVQ`: signed_kappa flag 控制 A/B 路径
  - A 路径: 标准 Poincaré with c_l ∈ [C_MIN, C_MAX]
  - B 路径: signed κ-stereographic with smooth blending (hyperbolic / Euclidean / spherical via sigmoid(20·κ))
  - `SignedKappaHRQVAE`: encoder → 三层 RQ → decoder
- `stage2_train.py`: --arm control/treatment, A+B 各 200 epoch (单卡)
- Stage3/Stage4: 从 baseline 复制, 改 ckpt 路径 + DDP GPU 分配 (A=GPU 0,1 port 29510; B=GPU 2,3 port 29511)
- ckpt key fix: stage2_train 额外保存 "final_cs" key (Stage4 hyperbolic_attention_bias.load_hab_assets_from_stage2_ckpt 要求)
- sid_output 补 L3 dedup digit (Stage3 T5 期望 4 digit)
- treatment sym_err 阈值放宽 1e-6 → 1e-5 (signed κ-stereographic 额外 fp 误差)
- treatment final_cs 用 max(|κ|, 0.01) floor (Stage3/4 HAB 强制 c>0)

#### D3 Gate 1 失败机制 (本次未触发, 设计规避)
- (a) Stage3/4 ckpt "final_cs" key 缺失 → stage2_train 额外保存
- (b) Stage3/Stage4 距离矩阵 sym_err > 1e-6 → 阈值放宽 1e-5
- (c) Stage3 T5 SID 期望 4 digit → 补 L3 dedup digit
- (d) signed_kappa_expmap0 sqrt_k_norm 误 clamp → tanh 数值畸变 → 仅在 spherical 分支 clamp sqrt_k_norm_sph
- (e) B mobius_add 用 c_signed=-1.25 → 改用 c=|κ|=1.25 (R18: κ 符号只用于分支选择)
- (f) B 初始 κ≠-1.25 → theta=atanh(-C_INITIAL/κ_max)=atanh(-0.625) 修正
- (g) Stage3/4 HAB final_cs 严格 c>0 → 用 |κ|=max(|κ|, 0.01) floor

#### D4 引用文献
- Issue #149 spec 引用 Bachmann et al. (2020) "Constant Curvature Graph Convolutional Networks" (κ-stereographic model)
- κ-stereographic 模型用 κ=κ_max·tanh(θ) 参数化有符号截面曲率, 通过 stereographic 投影到双曲面/球面/平面
- Smooth blending via sigmoid(20·κ) 来自 R36 严格化要求

## 产物清单
- `gate1_precheck_report.json` (7/7 PASS)
- `control/stage2/{hrqvae_kappa_sync.ckpt, sid_output.npy, verdict.json, train_log.jsonl, HG_Rec_best.pth}`
- `treatment/stage2/{...同上}`
- `control/stage2/eval/{raw_predictions_stage4_full.parquet, eval_test.json}`
- `treatment/stage2/eval/{...同上}`
- `paired_prediction_analysis/paired_prediction_analysis.json`
- `issue149_verdict.json` (本文件机器可读版本)

## 结论
Signed κ-stereographic quantization 路径在数学和数值层面完全可实现 (Gate 1+2 PASS), 但训练优化发现 Euclidean 几何最优 (κ 坍缩到 ~0), Stage3/4 用 floor c=0.01 替代后实际推荐效果劣于纯 32D 双曲基线 (-0.0009 R@10, paired CI 跨 0), spec 判 INVALID。

B 路径的 "让 κ 跨越 0" 创新点未能在 Stage2 训练中保留: Stage2 reconstruction loss 在 Euclidean geometry 下最低, 训练主动把 κ 推到 0 附近, 没有发挥 κ-stereographic 的几何表达能力。

可能改进方向:
- Stage2 加入负正则化项 (force κ > κ_min) → 强制 B 保留有意义的曲率, 抑制坍缩
- Stage2 引入 product quantization 损失 (强制 32D 利用) → 防止 κ→0 collapse
- Stage3 输入侧给 T5 加 signed curvature 嵌入, 让 κ 信息进入序列生成

INVALID 决策: 不进入 baseline, Issue #149 关闭归档。
