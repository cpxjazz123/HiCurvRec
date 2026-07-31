# Task #445 / Issue #154 [方向B Gate1] 受界样本product权重与双边配额硬分配

## R18 4 维度路径对比 (vs Issue #152 / Task #443 + #149 / #144 / #146)

| 维度 | Issue #152 (task443 FAIL) | Issue #149 (task439 FAIL) | Issue #144 (task434 FAIL) | Issue #154 (本 task, 升级修复方向B) |
|------|-----------------------------|----------------------------|----------------------------|--------------------------------------|
| **D1 spec 摘录** | softmax + entropy reg (ALPHA=0.1, ENTROPY_MIN=log(3)-0.1) → weight 退化成 one-hot (entropy ≈ 0) | continuous soft-anchor + softplus + random neg | temperature simplex mixing, mixing grad=0 | **截断simplex投影**: weight ∈ Δ with min/max bounds [w_min, w_max] per component, softmax output projected back to satisfy bounds (clamp + renormalize). 配合双边配额 (lower=1 + upper=cap). |
| **D2 实施核心** | weight_mlps (3 MLPs): encoder z_e → 3 logits → softmax (受 entropy reg lower bound) | ProductDistanceModel 静态 mixing scalars | 温度受控 + softmax mixing | **TruncatedSimplexKappaModel**: weight_mlps → 3 logits → softmax → clamp([w_min, w_max]) → renormalize to simplex. 然后 product d = α_0·d_hyp_l0 + α_1·d_hyp_l1 + α_2·d_eucl (per-sample α 强制在 bounds 内). Hard SID via bilateral quota (跟 #153 同, lower=1 + upper=cap). |
| **D3 Gate 1 失败机制** | entropy reg α=0.1 力度不够 vs contrastive_loss 推 weight 到 single-component (low entropy minima) | loss=0 + util 不达标 | mixing grad=0 | **新机制**: 截断simplex投影强制 weight 在 [w_min, w_max] 区间, 不允许 one-hot. 配合 bilateral quota 强制每码字至少 1 slot. 预期三层 util>=90%, max_load<5%, weight bounds 全程满足 |
| **D4 引用文献** | arXiv:2307.04514 | arXiv:2307.04514 | arXiv:2307.04514 | arXiv:2307.04514 + CrossRef VQ (同 #152, 不同实施) |

**R18 判定**: 4 维度都有差异 (重点在 D1 截断simplex投影 vs entropy reg + D2 clamp+renormalize vs softmax + D3 强制 weight bounds vs 单纯 entropy penalty + D4 同文献不同实施), 必须做新实验, **不允许**套用 #152/#149/#144/#146 判决 (R18 强制).

## 实施
- `scripts/task445_issue154_truncated_simplex_bilateral_quota.py` (~800 lines, fork from task443 + task444)
- TruncatedSimplexKappaModel: encoder z_e (B, D) → per-sample MLP → 3 weight logits → softmax → clamp([w_min, w_max]) → renormalize to simplex
- 截断simplex projection 保证 weight 始终在 bounds 内 (per-sample α_l ∈ [w_min, w_max] AND sum α = 1)
- product d_mix = α_0·d_hyp_l0 + α_1·d_hyp_l1 + α_2·d_eucl (per-sample α 强制 bounded)
- hard SID via bilateral quota (跟 #153 同, lower=1 + upper=cap)
- autograd graph proof: aux_loss → κ/weight_mlp grad path + hard assignment branch isolated + truncated simplex projection 数值稳定
- 1000-step entropy-reg control (跟 #152 复现) + 1000-step with truncated simplex + bilateral quota
- 每 100 步记录: weight min/max/entropy per layer, component contribution, util/max_load/min_load, κ/weight_mlp grad, #47 残差, NaN/Inf
- 8 件套: config + SHA256(item_emb) + truncated_simplex_graph_proof + train_curve + precheck + ckpt + verdict.json + commit

## Precheck 决策阈值 (Issue #154 spec)
- 图: continuous aux → κ/weight_mlp grad, integer bilateral assignment → hard SID
- **截断simplex bounds**: w_min=0.1, w_max=0.8 (Issue #154 spec 强制 — 防止 one-hot 退化, 但允许 dominant component)
- **截断simplex projection 数值稳定**: clamp 后 sum 必须 ≥ 1, 否则 renormalize; bounds 必须非空
- **配额可行性**: total_slots >= B (upper sum), 每码字 ≥ 1 slot (lower bound)
- aux_graph proof: continuous aux → κ/weight_mlp grad path 非零
- hard SID branch isolated

## Gate 1 决策阈值 (Issue #154 spec)
- PASS: 每层 weight 始终在 [w_min, w_max] 且至少 2 分量贡献 > 0.1; 每批双边配额可行; 三层 util>=90% AND max_load<5% AND min_load>=1/K; κ/weight_mlp 有限非零并更新; hard SID round-trip, 无 NaN/Inf; 8 件套齐全
- FAIL: 任一不满足即 STOP

## Gate 4 决策阈值 (Gate 1 PASS 后)
- 六项指标 (R@5/10/20, NDCG@5/10/20)
- 仅 test R@10 > 0.1020 = Target reached

## 复现命令
```bash
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task445 \
  python3 scripts/task445_issue154_truncated_simplex_bilateral_quota.py
```

## 跟 Issue #152 关键差异
- Issue #152: `weight = softmax(logits)` + entropy reg α=0.1 → 退化 one-hot
- Issue #154: `weight = renormalize(clamp(softmax(logits), w_min, w_max))` → 强制 weight 在 [0.1, 0.8] 区间
- Issue #154 配套: `if weight.sum() < 1: weight = weight / weight.sum() + 1e-10` 保证 simplex 约束
- 截断simplex + entropy reg 联合 (entropy 不再是唯一驱动)

## 跟 Issue #153 共享
- 双边配额机制 (lower=1 + upper=ceil(B/K)+1)
- Hungrian assignment (scipy.optimize.linear_sum_assignment)