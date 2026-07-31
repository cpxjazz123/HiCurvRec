# Task #438 / Issue #148 [方向A Gate1] 硬SID不变的κ可微双曲三元组分离与多样性损失

## R18 4 维度路径对比 (vs Issue #143 / #145)

| 维度 | Issue #145 (task435, #145 PARTIAL/NO-GO) | Issue #143 (task433, NO-GO) | Issue #148 (本 task, 修复方向A) |
|------|------------------------------------------|------------------------------|----------------------------------|
| **D1 spec 摘录** | aux_loss = pairwise+ranking minimize 推 collapse (#145 NO-GO, codebook collapse to center) | 修所有权 κ grad=0, R137 κ lock 复现 | **triplet hinge + diversity**: hard SID (detach) positive + 随机 negative + relu(margin + pos - neg), batch-level codebook repulsion EXPANDS spread (跟 #145 minimize 反方向) |
| **D2 实施核心** | pairwise = cost.mean() minimize + ranking = relu(top1 - top2) minimize | 参数所有权修复 (lr_codebook=0 切分) | **TripletKappaModel**: anchor = cost.argmin() (detach), neg = random.randint(K), hinge = relu(TRIPLET_MARGIN + pos_dist - neg_dist), diversity = -mean(codebook pairwise hyp distance). κ 拿 grad 路径: pos_dist / neg_dist / diversity 全部 → continuous hyp dist → c → κ |
| **D3 Gate 1 失败机制** | minimize 全局 pairwise 推 codebook 中心聚集 → usage<=1.56% | 参数所有权不修 → κ grad=0 | **新机制 (跟 #145 反方向)**: triplet hinge 让 pos_dist < neg_dist (margin 满足 → loss=0 但 grad 也=0, 跟 #146 argmin+relu 同 family — 但此处 hinge 用 detach anchor, **距离连续**; 负样本是 random 不是 argmin, 梯度路径不会突然归零). diversity 反向推 codebook 展开 (vs minimize collapse) |
| **D4 引用文献** | arXiv:2405.13979 曲率依赖可学习几何 | 内部 spec | arXiv:2405.13979 曲率依赖可学习几何 + 同步 scaling |

**R18 判定**: 4 维度都有差异 (重点在 D1 triplet hinge vs minimize collapse + D2 detach anchor vs 全局 minimize + D3 反方向推 spread + D4 同样 arXiv 但实施不同), 必须做新实验, 不允许套用 #145/#143 判决.

## 实施
- `scripts/task438_issue148_triplet_separation_diversity.py` (~530 lines, fork from task435)
- TripletKappaModel: 跟 CalibrationKappaModel 同架构, 仅 compute_triplet_diversity_loss 不同
- autograd graph proof: triplet → pos_dist/neg_dist → continuous hyp → c → κ (跟 #145 同 proof 逻辑)
- 1000-step no-aux control (跟 #143 复现) + 1000-step with triplet-diversity
- 每 100 步记录: triplet_loss / diversity_loss / pos_dist / neg_dist / κ grad / δκ / #47残差 / usage / max_load / count entropy / codebook pairwise separation / 域裕量 / NaN-Inf
- 8 件套: config + SHA256(item_emb) + aux_graph_proof + train_curve (control + triplet) + precheck + ckpt (TBD) + verdict.json + commit

## Precheck 决策阈值 (Issue #148 spec)
- aux_graph proof: triplet+diversity → continuous distance → c → κ grad path 非零
- hard SID branch isolated: z_q_hard = codebook[assign] 不参与 triplet loss 计算
- detach anchor: positive (argmin) detached, 负样本 (random) 也 detached, 只有 continuous distance 传 grad

## Gate 1 决策阈值 (Issue #148 spec)
- PASS: 全 10 个记录点三层 κ 有限非零 grad 及更新; 三层 usage >= 90%, max_load < 5%; codebook pairwise separation 不塌缩; hard SID round-trip; 无 NaN/Inf; 8 件套齐全
- FAIL: 任一不满足即 STOP, 不做 Stage 2/3/4

## 复现命令
```bash
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task438 \
  python3 scripts/task438_issue148_triplet_separation_diversity.py
```