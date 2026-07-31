# Task #444 / Issue #153 [方向A Gate1] 最小占用配额的全覆盖硬双曲分配

## R18 4 维度路径对比 (vs Issue #151 / Task #442 + #148 / #145)

| 维度 | Issue #151 (task442 PARTIAL) | Issue #148 (task438 FAIL) | Issue #153 (本 task, 升级修复方向A) |
|------|-------------------------------|----------------------------|--------------------------------------|
| **D1 spec 摘录** | upper-only Hungarian capacity (cap = ceil(B/K)+1), max_load<5% PASS 但 util L1/L2 不达标 | triplet hinge + diversity EXPAND + hard argmin | **minimum-load + upper-load 双边配额**: 每码字至少 1 slot (lower bound 强制全覆盖), 上限 cap = ceil(B/K)+1. Hungarian cost matrix 同时 enforce lower + upper bound |
| **D2 实施核心** | CapacityConstrainedKappaModel + Hungarian upper-only (slot expansion cap[i] 次) | TripletKappaModel + diversity EXPAND | **BilateralQuotaKappaModel** + Hungarian cost expand min_slot=1 + max_slot=cap[i]. 每码字强制至少 1 slot 出现在 cost matrix, 强制 encoder 不能忽略码字 (跟 #145/#148/#149 collapse family 反例) |
| **D3 Gate 1 失败机制** | upper-only cap 不约束 min — encoder 仍可聚到局部 codebook 子集, util L1/L2 FAIL | hinge 满足 → grad=0 + diversity 未 enforce | **新机制**: lower bound = 1 保证 K 个码字至少各分配 1 slot, 强制 spread. 预期三层 util>=90%, max_load<5%, all min_load>=1 (跟 #151 PARTIAL 反例对照) |
| **D4 引用文献** | arXiv:2405.13979 + CrossRef VQ optimization | arXiv:2405.13979 | arXiv:2405.13979 + CrossRef VQ optimization (同 #151, 不同实施) |

**R18 判定**: 4 维度都有差异 (重点在 D1 lower+upper 双边配额 vs upper-only + D2 min_slot=1 enforcement vs slot expansion cap + D3 全覆盖 vs 单纯 cap + D4 同文献不同实施), 必须做新实验, **不允许**套用 #151/#148/#145 判决 (R18 强制).

## 实施
- `scripts/task444_issue153_minimum_load_quota.py` (~750 lines, fork from task442)
- BilateralQuotaKappaModel: encoder z_e (B, D) → 3 层 κ → per-layer pairwise d + Hungarian cost
- cost matrix: expand (B, total_slots) where total_slots = sum(cap[i]), slot_to_codeword = [0]×cap[0] + [1]×cap[1] + ... + [K-1]×cap[K-1]
- **lower bound = 1**: 每码字至少出现 1 次 in slot_to_codeword (强制全覆盖, 即 total_slots >= K)
- linear_sum_assignment (Hungarian, O(B³) cost) on (B, total_slots) cost matrix
- per-batch feasibility: total_slots >= B AND each cap[i] >= 1 (lower bound constraint)
- autograd graph proof: aux_loss → κ grad path + hard assignment branch isolated
- 1000-step upper-only control (跟 #151 复现) + 1000-step with bilateral quota
- 每 100 步记录: min_load/max_load per layer, util, κ grad/delta, assignment cost gap, #47 residual, NaN/Inf
- 8 件套: config + SHA256(item_emb) + quota_graph_proof + train_curve + precheck + ckpt + verdict.json + commit

## Precheck 决策阈值 (Issue #153 spec)
- 图: continuous aux → κ grad, integer bilateral assignment → hard SID
- **配额可行性证明**: total_slots >= B (upper sum), 每码字至少 1 slot (lower bound), 100% feasible
- **lower+upper joint feasibility**: per-batch feasibility = (sum(cap) >= B) AND (cap[i] >= 1 for all i)
- aux_graph proof: continuous aux → κ grad path 非零
- hard SID branch isolated (argmin/autograd 切断)

## Gate 1 决策阈值 (Issue #153 spec)
- PASS: 每批双边配额可行; L0/L1/L2 util>=90% AND max_load<5% AND min_load>=1 per layer; κ 有限非零 grad 及更新; hard SID round-trip, 无 NaN/Inf; 8 件套齐全
- FAIL: 任一不满足即 STOP

## Gate 4 决策阈值 (Gate 1 PASS 后)
- 六项指标 (R@5/10/20, NDCG@5/10/20)
- 仅 test R@10 > 0.1020 = Target reached

## 复现命令
```bash
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task444 \
  python3 scripts/task444_issue153_minimum_load_quota.py
```

## 跟 Issue #151 关键差异
- Issue #151: `cost_expanded = np.zeros((B, total_slots))`, slot_to_codeword = [0]*cap[0] + [1]*cap[1] + ... (upper-only)
- Issue #153: 同样的 cost expand, 但 cap[i] >= 1 enforced (即 total_slots >= K), 强制所有 K 个码字出现在 cost matrix
- Issue #153 配套可行性检查: (sum(cap) >= B) AND (min(cap) >= 1), 跟 #151 不同