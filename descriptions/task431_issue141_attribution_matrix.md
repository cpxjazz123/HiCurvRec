# Task #431 / Issue #141 [方向B Gate1] product 分量归因矩阵与 anchor 限幅最小反证实验

## R18 4 维度路径对比 (vs Issue #135)

| 维度 | Issue #135 (task427, NO-GO 收口) | Issue #141 (本 task, 修复方向B) |
|------|----------------------------------|---------------------------------|
| **D1 spec 摘录** | 5-step audit (R137 κ lock + softmax steady state → grad=0) | **1000-step 最小复现 + component attribution matrix + anchor 限幅 + ranking flip ratio** |
| **D2 实施核心** | LayerProductHRQVAE with 3-component d_mix + bounded_correction_max=0.3 (5-step audit only) | **AttributionMatrixModel**: d_mix = w1*d_anchor + w2*d_fixed_hyp + w3*d_eucl, layer-level 3 scalars per layer, anchor ≥ 0.4 强制, ranking_flip ≤ 5% 强制 |
| **D3 Gate 1 失败机制** | R137 κ lock + softmax 边界饱和 → mixing logits grad=0 | **attribution 矩阵失效假设**: 即使 anchor 限幅生效, 码字塌缩后各分量在该少数码字 top-1 不再有 anchor-routing 信息 |
| **D4 引用文献** | arXiv:2307.04514 Weighted Mixed-Curvature Product Manifold | 同文献 + arXiv:2309.04082 mixed-curvature Transformer |

**R18 判定**: 4 维度都有差异 (重点在 D1 1000-step + D2 attribution matrix + D3 新失效机制假设), 必须做新实验, 不允许套用 #135 判决.

## 实施
- `scripts/task431_issue141_attribution_matrix.py` (~290 lines)
- AttributionMatrixModel + per-layer 3-scalar mixing_logits + bounded softmax + attribution matrix + ranking flip tracking
- 1000-step R18 minimal repro (~5min 单卡)
- 6 件套: config + SHA256 + trace + raw_log + verdict + commit
- 产物: `products/task431_issue141_attribution_matrix/{config,verdict}.json`

## Gate 1 决策阈值
- 三层至少两分量贡献 > 0.1
- anchor-correction 与 ranking flip 均不超过预声明上限 (5%)
- usage >= 90%, max_load < 5%
- kappa/mixing 梯度有限非零
- hard SID round-trip 可复现

## 复现命令
```bash
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task431 \
  python3 scripts/task431_issue141_attribution_matrix.py
```