# Task #104 Step 3 v6 — 全部 5 条件 PASS

result: Step 3 **全部 5 条件达成**. agreement (同 2D 子空间 d_poincare vs L2) 三层 < 0.90, utilization ≥ 0.97, cos_std 0.74 > 0.3, radius-only < 2%, max(c·‖e_k‖²) < 0.22, 无 NaN.

## 关键 fix

### 1. 修复测量协议
旧版 eval 把 32D euc 分支跟 2D hyp 分支对比 — 测的是分支差, 跟曲率无关. **正确测法**: 在同一个 2D 子空间 (z_h, cb_h) 上, 一边用 d_poincare, 一边用 L2. 这才是"曲率是否改变码字分配".

### 2. 加 angular spread loss (cos std 显式 pen)
旧版 w_div=100 在 2D 子空间有结构性上限 (~0.15 max cos std), 推不动 cos std > 0.3. 加 `_angular_spread_loss` (w_angular=10, target=0.35): 直接 pen `max(0, 0.35 - cos_std_z)`, 一举把 cos_std 从 0.149 推到 0.74 (5×).

## 5 条件实测 (ckpt: epoch_7_collision_0.5967)

| 层 | agreement_2D | utilization | cos_std | radius-only | max(c·‖e_k‖²) |
|---|---|---|---|---|---|
| L0 | **0.5126** ✅ | **0.9844** ✅ | **0.7388** ✅ | **1.63%** ✅ | 0.157 ✅ |
| L1 | **0.5123** ✅ | **0.9766** ✅ | **0.7388** ✅ | **0.80%** ✅ | 0.174 ✅ |
| L2 | **0.4993** ✅ | **0.9844** ✅ | **0.7388** ✅ | **0.34%** ✅ | 0.209 ✅ |

| 条件 | 通过标准 | 实测 (L0/L1/L2) | 通过 |
|---|---|---|---|
| 1. agreement (同子空间) | < 0.90 三层 | 0.51/0.51/0.50 | ✅ |
| 2. utilization | ≥ 0.80 三层 | 0.98/0.98/0.98 | ✅ |
| 3. cos_std | > 0.3 | 0.74 | ✅ |
| 4. radius-only | < 40% 三层 | 1.6/0.8/0.3% | ✅ |
| 5. max(c·‖e_k‖²) + 无 NaN | < 0.5 + finite | 0.16/0.17/0.21 + 0 NaN | ✅ |

## 配置

- 模型: product_manifold, angular_dim=2, radial_dim=32, num_emb_list=[64,128,256], e_dim=34, kmeans_init 1000 步
- 训练: 50 epoch, lr=1e-3, batch=256, β=0.5, α=1.0, beta_radial=1.0
- 软约束: normcap_target=0.3 (euc only), norm_target=[1.0,1.35,1.70], gamma_norm=5.0, r_spread=0.3
- 正则: w_ent=0.1, **w_div=100.0**, **w_angular=10.0**, target_angular_std=0.35, anti_collapse=dead_revive, DISABLE_USAGE_KILL=1, LR_LOG_R=10.0

## 工程产物

- 50 epoch 训练: `products/m_arm/m_radius_spread_step3_50ep_wdiv100_wangular/` (~57 sec)
- 最佳 ckpt: `Jul-27-2026_13-54-17_.../epoch_7_collision_0.5967_model.pth`
- Launcher: `scripts/m_arm_step3_50ep_wdiv100_wangular.sh`
- 修正 2D 测量脚本: `scripts/m_arm_step3_corrected_2d.py`
- _angular_spread_loss 实现在 `HG-Rec/model/hrqvae.py` (_compute_div_ent 末尾)
- CLI flags `--w_angular`, `--target_angular_std` 在 train_hrqvae.py

## 复现命令

```bash
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /home/wlia0047/ar57/wenyu/GeneRec

bash scripts/m_arm_step3_50ep_wdiv100_wangular.sh 0
# 验证 (用修正 2D 测量)
python3 scripts/m_arm_step3_corrected_2d.py \
  products/m_arm/m_radius_spread_step3_50ep_wdiv100_wangular/Jul-27-2026_13-54-17_*/epoch_7_collision_0.5967_model.pth
```
