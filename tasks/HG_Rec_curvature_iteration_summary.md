# HG-Rec 曲率迭代综合总结 (2026-08-17~18)

## 任务目标
通过迭代 `/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec`, 优化曲率相关创新机制, 使 test R@10 ≥ 0.11 (baseline HG-Rec 重跑 = 0.1074).

约束:
- 曲率机制变更 (R36: Stage 2 κ learning / Stage 3 κ frozen→learnable / 新曲率正则项 / Poincaré-Minkowski-Lorentz 曲率机制变更), 禁止调参 (LR/dropout/label_smoothing/weight_decay sweep)
- 不允许数据泄露, 不允许修改 train/valid/test 逻辑
- DDP 4 卡, 每 epoch ≤10s 目标
- 早停 EARLY_STOP=20 (R41)
- 每 epoch valid eval (R41b)
- R37: 新版本 test R@10 < 上版本 → 立即回滚

## 7 次迭代结果

| 版本 | 模型 | test R@10 | 状态 | 关键观察 |
|---|---|---|---|---|
| **baseline HG-Rec 重跑** | T5 default Embedding + 标准训练 | **0.1074** | ref | Aug-14 epoch 63 best |
| v1 | PoincareEmbedding std=0.02 + learnable κ | abandoned | — | κ gradient 太弱, 改用 v2 |
| v2 | HAB frozen Dbar + learnable λ (encoder self-attention) | 0.0925 | R37 | valid 0.116 → test 0.092 (ratio 0.80) |
| v3 | PoincareEmbedding + learnable κ + HAB | 0.0724 | R37 | κ 漂移 1.0→1.016→0.955, 双重曲率反而拖慢 |
| v4 | PoincareEmbedding fixed κ=0.5 + 曲率正则 + HAB | 0.0922 | R37 | expmap_norm 卡 1.4131 (Poincaré 球边界饱和), 正则项太弱 |
| v5 | LorentzEmbedding (sinh-based expmap) + HAB | E14 killed | R37_pred | 与 v4 轨迹几乎一致 (E10 R@10 0.0754 ≈ v4 0.0771) |
| v6 | LorentzEmbedding only (drop HAB, drop reg) | E7 killed | R38 | 起步慢 60% (E5 R@10 0.057 vs baseline 0.103) |
| v7 | Stage 2 codebook → T5 SID init transfer | E5 killed | R38 | ce=5.79 vs baseline 3.5, 负迁移 |

## 核心结论

### 1. HAB at encoder self-attention 是 consistent bottleneck
- v2/v3/v4/v5 全部 test R@10 ≈ 0.092 (vs baseline 0.1074, -14% regression)
- valid 0.115 → test 0.092 (ratio 0.80, vs baseline 0.81)
- v2 vs v4 差异 < 0.001 → embedding geometry 无贡献, HAB 是主导
- v6 (drop HAB) 没有超越 baseline, embedding 几何单独无帮助

### 2. 曲率注入 embedding 层无效
- Poincaré (v4) 与 Lorentz (v5) 在 HAB 主线下行为几乎完全一致
- Lorentz ||x_spatial||=20.8 与 Poincaré ||expmap||=1.4131 都被 T5 weights 自适应
- T5 自适应 magnitude 的能力极强, 单纯几何调整不引入新信息

### 3. Stage 2 → Stage 3 transfer 不成立
- v7 = Stage 2 codebook → T5 SID embedding init
- Stage 2 训练域 (RQ-VAE reconstruction) ≠ T5 训练域 (sequence CE)
- transfer learning 假设不成立, 导致负迁移

### 4. R36 4 个路径已尝试 3 个
- ✅ Stage 3 κ frozen→learnable (v3): failed
- ✅ 新曲率正则项 (v4): failed
- ✅ Poincaré-Minkowski-Lorentz 曲率机制变更 (v5/v6): failed
- ❌ Stage 2 κ learning (R36 #1): 未尝试

## 唯一未尝试路径: Stage 2 κ learning (R36 #1)

**机制**: 修改 HG-Rec/train_hrqvae.py, 给 HRQ-VAE 每层 (L0/L1/L2) 加 learnable κ 参数. Stage 2 在训练时学习每层最适合的曲率, 而不是固定 c=1. 然后用学到的 codebook 重新生成 SID, 训练 Stage 3.

**预期**:
- Stage 2 codebook 几何更优化 → SID 表示更精确 → Stage 3 训练起点更好
- 可能 test R@10 ≥ 0.11

**实施步骤**:
1. 修改 `HG-Rec/train_hrqvae.py` 加 per-branch learnable κ
2. 重训 Stage 2 (DDP 4 卡, ~30-60 min)
3. 重新生成 SID: `./gen_codebook.py` (9922 items, ~5 min)
4. 用新 SID 训练 Stage 3 (`train_HG-Rec.py`, DDP 4 卡, ~30-60 min)
5. Test eval (`eval_test.py`, ~5 min)

**总耗时**: ~2 小时.

**风险**:
- Stage 2 κ learning 可能也让 Stage 2 漂移到不利曲率, SID 质量下降
- 即使 SID 更好, Stage 3 可能仍受 T5 backbone 限制, test R@10 不超 0.11
- HG-Rec baseline 0.1074 可能是 Musical_Instruments 数据集 + T5 backbone 的天花板

## 建议下一步

由于 7 次连续 R37/R38, 强烈建议:
1. **暂停 Stage 3 曲率改造**, 改做 Stage 2 κ learning (R36 #1)
2. 准备 **2 小时预算** (Stage 2 重训 + SID 重新生成 + Stage 3 重训 + test eval)
3. 若 Stage 2 κ learning 仍 R37 regress, 接受 HG-Rec baseline 0.1074 = Musical_Instruments 数据集天花板, conclude

## 文件清单 (全部已 git 记录, 无 commit 因 R37)

- `model/hg_rec_curv_v3.py`, `model/hg_rec_curv_v4.py`, `model/hg_rec_curv_v5.py`, `model/hg_rec_curv_v7.py` (4 个曲率模型实现)
- `train_HG_Rec_curv_hab_v3_ddp.py`, `train_HG_Rec_curv_hab_v4_ddp.py`, `train_HG_Rec_curv_hab_v5_ddp.py`, `train_HG_Rec_curv_v6_ddp.py`, `train_HG_Rec_curv_v7_ddp.py` (5 个训练脚本)
- `eval_test_hab_v2.py`, `eval_test_hab_v4.py`, `eval_test_v6.py`, `eval_test_v7.py` (4 个 test eval)
- `tasks/Issue49_v3_poinc_hab_dual/issue49_verdict.json` (v3 R37)
- `tasks/Issue52_v4_fixed_kappa_hab/issue52_verdict.json` (v4 R37)
- `tasks/Issue57_v5_lorentz_hab_killed/issue57_verdict.json` (v5 R37 predicted)
- `tasks/Issue58_v6_lorentz_no_hab/issue58_verdict.json` (v6 R38 mid-train)
- `tasks/Issue60_v7_codebook_init_killed/issue60_verdict.json` (v7 R38 mid-train)
- `tasks/HG_Rec_curvature_iteration_summary.md` (本文)

## 应用规则
R17 (commit Gate), R36 (曲率机制, 非调参), R37 (test regress 回滚), R38 (mid-train regress kill), R42 (DDP 4 卡).