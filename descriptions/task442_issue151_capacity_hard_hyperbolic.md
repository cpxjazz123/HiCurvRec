# Task #442 / Issue #151 [方向A Gate1] 精确容量约束硬双曲分配与κ校准联合审计

## R18 4 维度路径对比 (vs Issue #148 / Task #438 + #145 / #140 / #131)

| 维度 | Issue #148 (task438 PARTIAL) | Issue #140 (EMA NO-GO) | Issue #131 (soft Sinkhorn NO-GO) | Issue #151 (本 task, 修复方向A) |
|------|-------------------------------|------------------------|----------------------------------|------------------------------------|
| **D1 spec 摘录** | triplet hinge + diversity EXPAND, 但 max_load=100% | EMA codebook (no hard constraint) | soft Sinkhorn (no hard capacity) | **batch精确容量约束的离散最小成本分配**: 在同一双曲 cost 上用硬整数/匹配求解, 每层每个码字有预注册容量上限, 输出仍是硬 SID, 不用 soft transport/Sinkhorn/posterior |
| **D2 实施核心** | TripletKappaModel hinge+repulsion + 64-sample diversity | EMA decay codebook update | soft Sinkhorn transport matrix | **CapacityConstrainedKappaModel**: 跟 #145 同 CalibrationKappaModel 架构 (aux minimize) + Hungarian/auction algorithm 做 batch 级 hard assignment with per-codeword capacity cap. cap[l,k] = floor(B/K_l) 或 ceil, hard constraint enforced via masked cost matrix |
| **D3 Gate 1 失败机制** | hinge 满足 → grad=0 + diversity 未 enforce spread | EMA 衰减但不重置死码 | soft transport 仍部分坍缩 | **新机制**: hard integer assignment 在 cost matrix 上 enforce 每码字 ≤ cap, 不允许 max_load>5%, 直接消除单码字捷径. 预期: 三层 usage>=90%, max_load<5% |
| **D4 引用文献** | arXiv:2405.13979 | (内部 spec) | (内部 spec) | arXiv:2405.13979 (曲率依赖量同步学习) + CrossRef VQ codebook optimization |

**R18 判定**: 4 维度都有差异 (重点在 D1 hard capacity constraint vs soft Sinkhorn/EMA/triplet + D2 Hungarian algorithm vs Sinkhorn-knopp + D3 直接消除单码字捷径 vs 各 NO-GO 路径 + D4 同文献 + CrossRef 新检索), 必须做新实验, **不允许**套用 #148/#140/#131 判决 (R18 强制).

## 实施
- `scripts/task442_issue151_capacity_hard_hyperbolic.py` (~700 lines, fork from task438)
- CalibrationKappaModel (跟 #145 同, aux minimize pairwise) + **匈牙利算法 hard assignment** (scipy.optimize.linear_sum_assignment 或 auction algorithm) with per-codeword capacity cap
- autograd graph proof: aux_loss → c → κ grad path + hard assignment branch isolated (跟 #148 同 proof 逻辑)
- 1000-step old hard-argmin control (跟 #143/#148 复现) + 1000-step with capacity-hard assignment
- 每 100 步记录: capacity feasibility rate / codeword counts / usage / max_load / assignment cost gap / κ grad/delta / #47 residual / codebook margin / loss / NaN-Inf
- 8 件套: config + SHA256(item_emb) + capacity_graph_proof + train_curve (control + capacity) + precheck + ckpt + verdict.json + commit

## Precheck 决策阈值 (Issue #151 spec)
- 容量求解整型性: assignment 是 int tensor, 每码字 count ≤ cap
- 容量可行性: 100% feasible (所有 batch)
- aux_graph proof: aux_loss → κ grad path 非零
- hard SID branch isolated: z_q_hard = codebook[assign] 不参与 capacity loss 计算

## Gate 1 决策阈值 (Issue #151 spec)
- PASS: 每批容量均可行, 三层 usage>=90%, max_load<5%, κ 均有限非零 grad 与更新, hard SID round-trip, 无 NaN/Inf, 8 件套齐全
- FAIL: 任一不满足即 STOP, 不做 Stage 2/3/4

## Gate 4 决策阈值 (Gate 1 PASS 后)
- 六项指标 (R@5/10/20, NDCG@5/10/20)
- 仅 test R@10 > 0.1020 = Target reached

## 复现命令
```bash
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task442 \
  python3 scripts/task442_issue151_capacity_hard_hyperbolic.py
```