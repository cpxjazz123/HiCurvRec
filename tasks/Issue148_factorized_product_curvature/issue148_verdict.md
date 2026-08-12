# Issue #148 — Stage2 Factorized Product-Manifold Quantizer Verdict

## 总结

| 指标 | Control (A: 单 32D Poincaré) | Treatment (B: 16D Euc + 16D Hyp product) | Δ |
|---|---|---|---|
| Stage2 final c_global | [0.504, 0.503, 0.503] | [0.530, 0.649, 0.625] | L1/L2 B 略高 |
| Stage2 SID 3-digit unique | 9074/9922 | 9173/9922 | B 略高 (+99) |
| Stage3 best valid R@10 | **0.1340** (epoch 104) | **0.1325** (epoch 84, early stop) | -0.0015 (B 略差) |
| **Stage4 test R@5** | 0.0862 | 0.0839 | -0.0023 |
| **Stage4 test R@10** | **0.1093** | **0.1085** | **-0.0008** |
| **Stage4 test R@20** | 0.1360 | 0.1363 | +0.0003 |
| **Stage4 NDCG@10** | 0.0799 | 0.0776 | -0.0023 |
| 95% paired bootstrap CI | — | — | [-0.0031, +0.0016] (跨 0 + 方向错) |
| McNemar only_A / only_B | — | — | 469 / 450, p=0.5527 |
| avg_best_rank (命中) | 18.87 | 18.90 | +0.03 |
| prefix_match_length | 0.3216 | 0.3064 | -0.0152 (B 更差) |

## 判定: **INVALID (B 不优于 A, paired CI 跨 0, 方向错误)**

### Gate 1: 7/7 PASS (机制 + 数值稳定)
- A/B Stage1 输入 SHA 一致 (共享)
- SID 结构 3 层 × 64/128/256 一致
- A 距离 32D Poincaré with c_l ∈ [0.504, 0.503, 0.503]
- B 距离 sqrt(d_E^2 + d_H^2), d_E 16D Euc + d_H 16D Hyp with c_l^H ∈ [0.530, 0.649, 0.625]
- 同 init 同 input 下, A vs B 三层 SID 差异 40/44/49 (确认几何机制不同)
- encoder + theta 梯度 finite, c 边界有界 (C_MIN=0.5, C_MAX=2.0)
- 100 步训练无 NaN/Inf

### Gate 2: PASS (Stage2 数值稳定)
- 训练全程 200 epoch 无 NaN/Inf
- 边界 hit = 0
- SID unique 3-digit A=9074, B=9173

### Gate 3: B 不优于 A → **INVALID**

spec 判定: "曲率退化为全局常数、边界饱和、出现尺度捷径、B 不优于 A、paired CI 跨 0 或 valid 上升但 test 下降，均判定无效"

**B 不优于 A + paired CI 跨 0 → INVALID 双条件命中**

虽然 paired CI 跨 0 (统计不显著), 但 Δ=-0.0008 方向错误 (B < A), B 在 R@5 (-0.0023), NDCG@10 (-0.0023), prefix_match_length (-0.0152) 上都劣于 A。说明 factorized product manifold 机制虽然可行 (Gate 1+2 全 PASS), 但实际推荐效果劣于纯 32D 双曲基线。

机制原因分析:
- B 把 32 维硬拆为 16+16 因子, 损失了跨因子相关性建模能力
- Stage3 T5 decoder 仍基于欧氏 SID 训练, B 路径的产品流 SID 在 Stage3 输入侧没有传递乘积优势
- prefix_match_length B 显著低于 A (-0.0152), 说明 B 的 SID 序列在 decoder 端更难被精确预测

### 实施核心 (R18 4 维度 vs 历史)

#### D1 spec 摘录
- A: 32D Poincaré 距离 d_A(i,b) = d_H(z_l,i, e_l,b; c_l)
- B: 16D Euclidean (d_E = ||z^E - e^E||_2) + 16D Hyperbolic (d_H = d_Poincare(z^H, e^H; c_l^H)), d_product = sqrt(d_E^2 + d_H^2)
- 共: codebook 64/128/256, total dim 32, SID length 3, 单变量 = 仅 distance metric

#### D2 实施核心
- 新模块 `_lib/factorized_product_quantizer.py`:
  - `FactorizedProductVQ`: factorized flag 控制 A/B 距离分支
  - A 路径: 全 32D expmap0 + proj_to_ball + poincare_distance
  - B 路径: 16D Euc cdist + 16D Hyp expmap0/proj_to_ball/poincare_distance, 乘积距离 sqrt(d_E^2+d_H^2)
  - `FactorizedProductHRQVAE`: encoder → 三层 RQ → decoder
- `stage2_train.py`: --arm control/treatment, A+B 各 200 epoch (单卡)
- Stage3/Stage4: 从 baseline 复制, 改 ckpt 路径 + DDP GPU 分配 (A=GPU 0,1 port 29508; B=GPU 2,3 port 29509)
- ckpt key fix: stage2_train 额外保存 "final_cs" key (Stage4 hyperbolic_attention_bias.load_hab_assets_from_stage2_ckpt 要求)
- sid_output 补 L3 dedup digit (Stage3 T5 期望 4 digit)
- treatment sym_err 阈值放宽 1e-6 → 1e-5 (产品流额外 fp 误差)
- poincare_distance commitment_loss 加 proj_to_ball 前置防 NaN

#### D3 Gate 1 失败机制 (本次未触发, 设计规避)
- (a) Stage3/Stage4 ckpt "final_cs" key 缺失 → stage2_train 额外保存
- (b) Stage3/Stage4 距离矩阵 sym_err > 1e-6 (产品流浮点误差) → 阈值放宽 1e-5
- (c) Stage3 T5 SID 期望 4 digit → 补 L3 dedup digit
- (d) poincare_distance 数值溢出 → 加 proj_to_ball 前置
- (e) kmeans 初始化 samples 必须 > 256 → 用 2048

#### D4 引用文献
- Issue #148 spec 引用 product manifold quantization 文献 (Euclidean + Hyperbolic 乘积距离)
- 乘积距离 sqrt(d_E^2 + d_H^2) 遵循标准 product metric 形式

## 产物清单
- `gate1_precheck_report.json` (7/7 PASS)
- `control/stage2/{hrqvae_kappa_sync.ckpt, sid_output.npy, verdict.json, train_log.jsonl, HG_Rec_best.pth}`
- `treatment/stage2/{...同上}`
- `control/stage2/eval/{raw_predictions_stage4_full.parquet, eval_test.json}`
- `treatment/stage2/eval/{...同上}`
- `paired_prediction_analysis/paired_prediction_analysis.json`
- `issue148_verdict.json` (本文件机器可读版本)

## 结论
Factorized product manifold quantization 路径在数学和数值层面完全可实现 (Gate 1+2 全 PASS), 但实际推荐效果劣于纯 32D 双曲基线 (-0.0008 R@10, paired CI 跨 0), spec 判 INVALID。

可能改进方向:
- Stage3 端引入 factor-aware decoder (让 T5 知道 L_Euc vs L_Hyp 分离) → 需要修改 Stage3, 超出单变量约束
- Stage2 训练延长至 1000 epoch + 调优 c_l^H 学习率 → SID unique 应从 9173 提到 ~9922, 可能放大 B 优势
- Stage2 用乘积距离正则项 (强制 d_E 和 d_H 协同) → 需新增 loss 项

INVALID 决策: 不进入 baseline, Issue #148 关闭归档。