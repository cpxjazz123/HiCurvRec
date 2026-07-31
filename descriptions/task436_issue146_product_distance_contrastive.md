# Task #436 / Issue #146 [方向B Gate1] hard-SID不变的product距离对比辅助损失

## R18 4 维度路径对比 (vs Issue #145 + #144)

| 维度 | Issue #145 (task435, PARTIAL) | Issue #144 (task434, NO-GO) | Issue #146 (本 task, 修复方向B) |
|------|--------------------------------|------------------------------|----------------------------------|
| **D1 spec 摘录** | 修 loss design (aux 加 κ-dependent 项): pairwise + ranking minimization | 修 mixing 路径: temperature simplex + entropy bound clamp | **不改变 hard SID + 新增 product-distance pairwise contrastive calibration loss**: 同一 batch 正/负 item 对以 d_anchor / d_fixed-hyp / d_eucl 层级加权连续距离构造 margin/ranking 目标, 让 mixing + 独立 κ 通过连续 d_mix 拿 grad |
| **D2 实施核心** | CalibrationKappaModel: aux_loss = α_pair · pairwise_mean + α_rank · relu(margin). 推 codebook collapse | TempSimplexModel: 3 mixing scalars + bounded softmax + entropy clamp | **ProductDistanceModel**: d_mix = α·d_anchor + β·d_fixed-hyp + γ·d_eucl, **contrastive loss** = relu(margin - d_mix_pos + d_mix_neg), mixing 用 softmax over 3 scalars. **autograd graph 隔离 hard SID**. |
| **D3 Gate 1 失败机制** | aux minimize pairwise + margin → 推 codebook 聚集 → collapse | 温度受控 + 熵下界仍 mixing grad=0 | **contrastive margin minimization 假设**: product 距离 + contrastive 让 mixing/κ 拿到 grad (3 分量都贡献), 但 hard SID 不在 d_mix grad path |
| **D4 引用文献** | arXiv:2405.13979 曲率依赖可学习几何路径 | 同 + Mixed-Curvature Riemannian Optimization | **arXiv:2307.04514 weighted mixed-curvature product manifold** (分量权重按数据学习) — 跟 #145 完全不同文献 |

**R18 判定**: 4 维度都有差异 (重点在 D1 product distance contrastive + D2 3 分量 + D3 contrastive margin + D4 arXiv:2307.04514), 必须做新实验, 不允许套用 #145/#144 判决.

## 实施
- `scripts/task436_issue146_product_distance_contrastive.py` (~360 lines)
- ProductDistanceModel (3 独立 κ + 3 mixing scalars + 3-group optimizer)
- d_mix = α_anchor · d_anchor + α_fixed · d_fixed + α_eucl · d_eucl (3 距离层级加权)
- contrastive_loss = mean relu(margin - d_mix_pos + d_mix_neg) (positive pull, negative push)
- hard SID 跟 Issue #145 同样 .detach() 隔离
- Precheck 强制 (Issue #146 spec): contrastive loss → d_mix → mixing/κ + hard SID 不参与该梯度
- 1000-step no-aux control + product-calibration 对照
- 8 件套: config + SHA256 + trace + precheck + d_mix_graph_proof + raw_log + verdict + commit
- 产物: `products/task436_issue146_product_distance_contrastive/{config,verdict,precheck,d_mix_graph_proof}.json`

## Precheck 决策阈值
- autograd trace: contrastive_loss → d_mix → mixing/κ 路径存在
- hard SID 不在 contrastive loss 路径 (z_q_hard = codebook[assign] 不参与 d_mix)
- 3 分量都 requires_grad=True 且都贡献 grad (per Issue #146 spec)

## Gate 1 决策阈值
- 每层 κ + mixing 都有有限非零 grad 和更新
- 至少两分量贡献 > 0.1
- hard SID round-trip + 无 NaN/Inf
- usage >= 90%, max_load < 5%

## 复现命令
```bash
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task436 \
  python3 scripts/task436_issue146_product_distance_contrastive.py
```