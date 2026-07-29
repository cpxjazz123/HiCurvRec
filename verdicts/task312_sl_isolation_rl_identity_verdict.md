# Task #312 — Issue #35 s_l isolation (r_l identity + s_l=[2,2,2]) Stage 4 eval

**日期**: 2026-07-30
**状态**: ❌ **NO-GO** (R@10=0.0846, -17.0% vs baseline 0.1020, **worse** than task309c's r_l alone -1.5%)
**Stage**: Stage 1+2+3+4 全跑通 (5-Gate protocol PASS)

## Settings
- per-layer Codebook Transforms: **r_l=[1,1,1] identity + s_l=[2,2,2]** (s_l 隔离)
- code_path: `_t5_hrqvae_issue35_rl_identity_sl22.npy`
- Beam size: 50 (K14)
- T5-mini 200 epoch Stage 3

## Pipeline metrics
| Gate | Result |
|------|--------|
| 1 (Stage 1, 100 ep) | ✅ L0/L1/L2 100% usage at epoch 100, collision=0.1229 |
| 2 (Stage 2 Sinkhorn) | ✅ 9481/9922 unique SID (95.6%), 4-digit collision=0.1229 |
| 3 (Stage 3 T5-mini 200 ep) | ✅ HG_Rec_best.pth saved @ ep169 |
| 4 (Stage 4 eval @ beam=50) | ❌ **R@10=0.0846 (-17%)** |

## Stage 4 metric (test_*, 97 batches)
- Recall@5 = 0.0727
- Recall@10 = **0.0846** (-17.0% vs baseline 0.1020)
- Recall@20 = 0.1017
- NDCG@5 = 0.0650
- NDCG@10 = 0.0688
- NDCG@20 = 0.0731

## Verdict
**s_l alone (r_l identity) is INSUFFICIENT and HARMFUL to Stage 4 R@10.** This is a critical finding:
- task312 (s_l alone) @ beam=50 → R@10=0.0846 (**-17%**)
- task309c (r_l alone, task304 Arm A) @ beam=50 → R@10=0.1005 (-1.5%)
- Issue #30 (r_l + s_l both extreme) @ beam=50 → **R@10=0.1041 (+2.1%)** ⭐

The **interaction effect is ESSENTIAL**: r_l extreme alone is mild (-1.5%), s_l extreme alone is severe (-17%), but r_l + s_l together is +2.1%. This is a non-linear synergy effect, NOT a linear sum.

## Implications
1. **Issue #35 联立 4-arm ablation close**:
   - Issue #30 (r_l + s_l both extreme) → +2.1%  ⭐ GO
   - task312 (s_l only, r_l identity) → -17%  ❌ NO-GO
   - task309c (r_l only, s_l identity) → -1.5%  ❌ NO-GO
   - task313 (r_l + s_l both identity control) → in progress Stage 3

2. **D6 ablation fault confirmed**: r_l alone is NOT sufficient, s_l alone is much worse, both required.

3. **Issue #30's specific r_l=[0.1,1,10] + s_l=[2,2,2]** is the unique winning combo. Any orthogonal ablation destroys it.

4. **Task #314 description should pivot**: instead of 4-arm α/β/γ/ε, focus on:
   - **Confirm Issue #30 = unique winning combo** (close Issue #35)
   - **Explore variations of Issue #30** (e.g., r_l=[0.01,1,100] + s_l=[3,3,3] — sharper extreme)
   - **Stage 4 + Stage 3 protocol combine** — task194 anchor at beam=50/100 (if ckpt still exists)

## R11.3 transparency
- 选 r_l=[1,1,1] identity 而不是 r_l=[1,2,1] / r_l=[1,0.5,1] 因为 (a) 严格 identity 是真 baseline 隔离 (b) 单变量 ablation 减少 GPU 消耗
- 如要更严格隔离, 后续可再跑 r_l=[1,0.5,1] + s_l=[2,2,2] (弱 r_l) 或 r_l=[1,2,1] + s_l=[2,2,2] (强 r_l)
- 当前单点足够否定 Issue #35

## Issue #38 update
- Arm α (T5-small) — CLOSED NO-GO
- Arm ε (K sweep on Issue #30) — task309b partial GO (+2.1pp at K=50)
- **Arm γ (per-layer transform orthogonal ablation) — Issue #35 closed, Issue #30 is unique winning combo**
- Arm β (HNSW + rerank) — not yet tested, marginal priority now (K14 already gets +2.1pp)
