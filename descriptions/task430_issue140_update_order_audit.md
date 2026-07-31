# Task #430 / Issue #140 [方向A Gate1] 先验冻结的 hard-EMA 更新顺序审计与最小复现

## R18 4 维度路径对比 (vs Issue #134)

| 维度 | Issue #134 (task426, NO-GO 收口) | Issue #140 (本 task, 修复方向A) |
|------|----------------------------------|---------------------------------|
| **D1 spec 摘录** | 5-step audit (R137 κ lock → grad_finite_nz=False) | **1000-step 最小复现 + 更新顺序隔离审计 + kappa 更新后 #47 同步 trace** |
| **D2 实施核心** | HardEMAHRQVAE with geodesic_ema_update + count_regularizer (5-step audit only) | **两阶段更新**: (1) encoder/backprop only, (2) detached hard-count geodesic EMA on full epoch, 每 step #47 scale 同步在 kappa 更新后 |
| **D3 Gate 1 失败机制** | R137 κ lock → grad=0 → EMA 推 boundary → codebook collapse | **隔离更新顺序假设"encoder 不偷渡到 codebook"** (grad=0), 但 **detached EMA 仍推 boundary 塌缩** |
| **D4 引用文献** | Berman-Metzler 2020 κ-Stereographic + 范数边界 | 同文献 + arXiv:2405.13979 Robust Hyperbolic Learning |

**R18 判定**: 4 维度都有差异 (重点在 D1 1000-step + D2 隔离更新顺序 + D3 新机制假设), 必须做新实验, 不允许套用 #134 判决.

## 实施
- `scripts/task430_issue140_update_order_audit.py` (~280 lines)
- HardEMAModel with explicit two-stage update (encoder backward → kappa update → #47 scale recalibration → codebook EMA → distance)
- 1000-step R18 minimal repro (轻量, ~5min 单卡)
- 6 件套: config + SHA256 + trace + raw_log + verdict + commit (R20+R21 强制)
- 产物: `products/task430_issue140_update_order_audit/{config,verdict}.json`

## Gate 1 决策阈值
- encoder→codebook gradient leakage = 0 (3/3 layers codebook.grad max = 0)
- kappa #47 同步 trace 完整 (10 record points × 3 layers)
- 无越域/NaN/Inf
- hard SID round-trip 可复现
- usage >= 90%, max_load < 5%

## 复现命令
```bash
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task430 \
  python3 scripts/task430_issue140_update_order_audit.py
```