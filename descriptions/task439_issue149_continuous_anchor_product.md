# Task #439 / Issue #149 [方向B Gate1] 硬SID不变的连续soft-anchor product校准损失

## R18 4 维度路径对比 (vs Issue #144 / #146)

| 维度 | Issue #146 (task436, PARTIAL/NO-GO) | Issue #144 (task434 NO-GO) | Issue #149 (本 task, 修复方向B) |
|------|--------------------------------------|----------------------------|------------------------------------|
| **D1 spec 摘录** | product distance contrastive (3 分量 d_mix), argmin 选 anchor + F.relu margin (#146 PARTIAL/NO-GO, argmin 切断 + relu 0 grad) | temperature simplex mixing (#144 NO-GO, mixing grad=0) | **continuous soft-anchor product**: softmax(-d_mix/T) 加权 sum (NO argmin) + F.softplus margin (NO relu), T > 0, 负样本 random (NO argmin), 温度预注册 |
| **D2 实施核心** | ProductDistanceModel 3 mixing + softmax + d_mix = α·d_anchor + β·d_fixed + γ·d_eucl + relu(margin - pos + neg) contrastive | 温度受控 + 熵下界 + softmax mixing | **ProductDistanceModel (同 #146) + contrastive_loss 替换**: soft_anchor_dist = sum_k softmax(-d_mix_k / T) * d_mix_k (连续加权), neg = random.randint(K), loss = softplus(MARGIN + soft_anchor - neg_dist). κ/mixing 拿 grad 路径: d_mix → soft_w (full K participation) → soft_anchor → softplus → κ/mixing grad |
| **D3 Gate 1 失败机制** | argmin 切断 + F.relu 频繁 0 → 训练中 κ/mixing grad=0 (#146) | 温度受控 + 熵下界仍 mixing grad=0 (#144) | **新机制**: continuous soft-anchor (NO argmin 切断, 全部 K 个 codebook 拿 grad), F.softplus (处处可微, 无 hard 0 区间), 负样本 random (NO 共享 anchor, NO argmin path). 预期: 全 10 个记录点 κ/mixing grad 都有 finite nonzero |
| **D4 引用文献** | arXiv:2307.04514 weighted mixed-curvature product manifold | arXiv:2307.04514 + Riemannian Optimization | arXiv:2307.04514 weighted mixed-curvature product manifold (同 #146 文献, 实施不同) |

**R18 判定**: 4 维度都有差异 (重点在 D1 continuous soft-anchor vs argmin + softplus vs relu + D2 温度预注册 + D3 反 #146/#144 失败机制 + D4 同文献不同实施), 必须做新实验, 不允许套用 #146/#144 判决.

## 实施
- `scripts/task439_issue149_continuous_anchor_product.py` (~620 lines, fork from task436)
- ProductDistanceModel: 跟 task436 同架构, 仅 contrastive_loss 函数不同
- autograd d_mix_graph_proof: soft-anchor continuous d_mix → κ/mixing (跟 #146 同 proof 逻辑)
- 1000-step no-aux control (跟 #143/#144 复现) + 1000-step with continuous-anchor
- 每 100 步记录: soft-anchor entropy / temperature / softplus loss / κ/mixing grad / δκ / δmixing / component contribution / usage / max_load / count entropy / codebook separation / 域裕量 / NaN-Inf
- 8 件套: config + SHA256(item_emb) + d_mix_graph_proof + train_curve (control + soft-anchor) + precheck + ckpt (TBD) + verdict.json + commit

## Precheck 决策阈值 (Issue #149 spec)
- aux_graph proof: soft-anchor continuous d_mix → κ/mixing grad path 非零
- hard SID branch isolated: z_q_hard = codebook[assign] 不参与 soft-anchor loss 计算
- T > 0 (温度预注册, NO T→0 collapse)
- 至少两分量贡献 > 0.1

## Gate 1 决策阈值 (Issue #149 spec)
- PASS: 全 10 个记录点三层 κ/mixing 均有有限非零梯度; 权重/anchor entropy 不塌缩; 至少两分量贡献 > 0.1; hard SID round-trip; usage >= 90% / max_load < 5%; 无 NaN/Inf; 8 件套齐全
- FAIL: 任一不满足即 STOP, 不做 Stage 2/3/4

## 复现命令
```bash
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task439 \
  python3 scripts/task439_issue149_continuous_anchor_product.py
```