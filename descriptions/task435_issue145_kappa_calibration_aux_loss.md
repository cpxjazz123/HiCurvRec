# Task #435 / Issue #145 [方向A Gate1] 硬SID保持下的κ可微几何校准辅助损失

## R18 4 维度路径对比 (vs Issue #143)

| 维度 | Issue #143 (task433, NO-GO 收口) | Issue #145 (本 task, 修复方向A) |
|------|----------------------------------|---------------------------------|
| **D1 spec 摘录** | 修独立 κ 参数所有权 + optimizer group | **保持 hard argmin/SID 不变 + 新增 κ 可微几何校准辅助损失**: 同一批 encoder/候选 codebook 连续双曲距离 + #47 缩放构造 pairwise/ranking calibration, 梯度仅通过距离→κ |
| **D2 实施核心** | RepairedKappaModel: 3 独立 kappa_l_raw_0/1/2 + 3-group optimizer (kappa lr=1e-3, codebook lr=1e-4) | **CalibrationKappaModel**: 复用 RepairedKappaModel 独立 κ + 新增 **aux_loss** = pairwise calibration (||sinh(sqrt(c)*rho)/sqrt(c)|| 跟 c 强相关) + ranking calibration (margin between top-k distances), **autograd graph 强制隔离 hard SID** (硬 assign 通过 .detach()) |
| **D3 Gate 1 失败机制** | loss 不通过 c → κ grad=0 (loss design 是真锁) | **aux loss 设计假设**: aux loss 直接用连续 hyperbolic distance 构造 ranking/pairwise calibration, 让 c 拿到 grad, 但 hard SID 不在 κ gradient path |
| **D4 引用文献** | arXiv:2405.13979 curvature-aware optimizer + fine-tunable hyperbolic scaling | 同文献 + 曲率依赖的可学习几何路径 + 同步 scaling |

**R18 判定**: 4 维度都有差异 (重点在 D1 loss design 修复 + D2 aux loss 构造 + D3 autograd graph 隔离 hard SID + D4 引用角度), 必须做新实验, 不允许套用 #143 判决.

## 实施
- `scripts/task435_issue145_kappa_calibration_aux_loss.py` (~310 lines)
- CalibrationKappaModel (基于 RepairedKappaModel + 新增 aux loss)
- aux_loss = α_pair * pairwise_calibration + α_rank * ranking_calibration
- pairwise_calibration: 用 c-dependent hyperbolic distance sum, 让 c 拿到 grad
- ranking_calibration: 用 c-dependent distance margin (top-1 - top-2), 让 c 拿到 grad
- Autograd graph 隔离 hard SID (z_q_hard 跟 cost 都不在 κ grad path)
- 1000-step no-aux control + calibration 版本同预算
- 8 件套: config + SHA256 + trace + precheck + aux_graph_proof + raw_log + verdict + commit
- 产物: `products/task435_issue145_kappa_calibration_aux_loss/{config,verdict,precheck,aux_graph_proof}.json`

## Precheck 决策阈值
- Autograd graph 验证: aux_loss.requires_grad → distance → c → kappa_l_raw[i] 路径存在
- Hard SID 路径: z_q_hard → cost (但 cost 不参与 aux_loss 计算, 只用于 hard assign)
- Hard SID 不在 κ grad path: 验证 z_q_hard = codebook[assign] (detach 路径) 不贡献 κ grad

## Gate 1 决策阈值
- 三层 κ 均有限非零 grad 和更新
- #47 trace 有限
- hard SID 跟 control 同一导出规则 + round-trip 通过
- 无 NaN/Inf
- usage >= 90%, max_load < 5%

## 复现命令
```bash
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task435 \
  python3 scripts/task435_issue145_kappa_calibration_aux_loss.py
```