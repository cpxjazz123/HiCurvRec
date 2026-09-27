# S02_HYPOTHESIS — canonical source packet

STAGE_ID=S02_HYPOTHESIS
ROUND=1
ITERATION=30

## Objective
Independently register one falsifiable, protocol-valid hypothesis for the S14-approved prospective matched-seed comparison. Use only the two already-tested FCCR-1 fixed-curvature mappings, their approved common inputs, and seeds 43/44/45. Do not tune constants, introduce a third mapping, alter other mechanisms, authorize GPU work, or treat unresolved S03 provenance as passed.

## Governing sources
1. Root `/home/wlia0047/ar57/wenyu/GeneRec/CLAUDE.md` (auto-supplied repository rule; especially stage output roots, no CLI/env overrides, Stage2 gradient-path check, no quality gates, commit/push rules).
2. Active `skill://curvature-rqvae-iter`, FCCR-1 and S02 hypothesis requirements.
3. Judge-approved S00 `stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/source_snapshot_iter30.md`.
4. Judge-approved S01 `stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/protocol_manifest_iter30.md` and `.../S01_PROTOCOL_LOCK/round_1/judge.md`.
5. For primary historical details, iter29 `logs/hypothesis_iter29.md`, `logs/protocol_manifest_iter29.md`, `logs/mechanism_manifest_iter29.md`, `scripts/compute_closed_form_curvature.py`, `scripts/computed_behavior_branching.json`, and iter26 corresponding mapping sources. Read direct files as needed; distinguish source facts from inference.

## Locked question and arms
Prospective replication of whether the existing iter29 fixed mapping changes retrieval relative to the existing iter26 fixed mapping under three new matched seeds (43, 44, 45). Iter26 mapping is the fresh paired control; iter29 mapping is the candidate. There are six complete Stage2→Stage3 pipelines. The only conceptual contrast is the mapping. Do not pool seed 42 into the primary estimate. Historical reference values: iter26 seed-42 R@10 `0.057017009349048554`; iter29 seed-42 R@10 `0.05921064085377531`; both historical `n_eval=57439`; their single difference is not a prospective effect or variance estimate.

The user success threshold is inclusive `test_recall@10 >= 0.065`. Classify it per protocol-valid final result. It is distinct from the paired mapping-effect hypothesis; no run or block may stop early due to the threshold. Finish all three predeclared pairs before reporting the primary paired estimate.

## Shared inputs and provenance status
Layer order `[L0,L1,L2]`:
- `B=[19.324911558712664,1.4605688962651735,1.0148104414712726]` (recorded behavior-branching vector).
- `m_raw=[1.0,0.10941,0.09331]` (historical `raw_residual_medians`; **provisional pending independent S03 provenance adjudication**).
- Do not substitute normalized layer scales `[0.001,0.932889,1.0]`.
- Iter26 historical implementation consumed the ambiguous JSON key `residual_norm`; the same record contains the explicit `raw_residual_medians` vector with identical numbers, but that alias alone does not prove raw-residual semantics. New iter30 mapping inputs must use the explicit raw-residual field only. Preserve S00/S01 medium-confidence historical provenance limits. S03 failure blocks subsequent mapping approval/MVG/Stage2.

## Exact registered mappings (no alternatives)
Both are FCCR-1: compute once before Stage2; curvature is fixed, non-trainable and time-invariant; no cyclic schedule, curvature regularization, curvature-conditioned optimizer or auxiliary loss. Existing Stage2/Stage3 settings, inputs, warm-start and evaluation protocol are otherwise held fixed.

**Control: iter26 mapping**
```text
s_l = log1p(B_l) / log1p(m_l_raw / min(m_raw))
z_l = (s_l - mean(s)) / (std(s) + 1e-12)
c_l = clip(0.5 * exp(0.2 * z_l), 0.05, 1.5)
s=[1.2238118890466054,1.1604516044603779,1.010644112691917]
z=[1.03129493309937,0.3223997103378134,-1.3536946434371857]
c=[0.6145357379232853,0.5333020920777128,0.3814078098431606]
```

**Candidate: iter29 mapping**
```text
x_l=B_l/(B_l+2.0)
y_l=m_l_raw/(m_l_raw+0.1)
u_l=(x_l+y_l)/2
c_l=0.05+1.45*u_l
x=[0.9062129756321139,0.42206034326942476,0.3366083742817442]
y=[0.9090909090909091,0.5224678859653312,0.48269618747090165]
u=[0.9076519423615115,0.4722641146173780,0.40965228087632294]
c=[1.3660953164241916,0.7347829661951981,0.6439958072706683]
```
Candidate constants: `B_ref=2.0`, `m_ref=0.1`, `c_min=0.05`, `c_max=1.50`. No cross-layer normalization for candidate. These substitutions reproduce recorded decimal values only; they do not establish the residual provenance or historic byte identity.

## Protocol lock
- Canonical implementation/source-lineage parent: iter29, current root code revision `ecd01e4712a1badd38a0338255f4b2ec7b030aff`; paired control is iter26 mapping, not iter29.
- Three pairs at seeds 43, 44, 45; same Stage2 and Stage3 seed within each pair; six total runs; serial, complete-block execution.
- Stage2: 100000 steps; 3 quantizer layers ×256 entries; input 768; hidden `[512,256,128]`; embedding 32; batch 640/GPU; 4 ranks; commitment 1.0; Sinkhorn `sk_eps=0.05`/3 iterations; AdamW `lr=1e-3`, weight decay `1e-4`; existing losses/data/runtime held fixed. Fixed-curvature buffers must not receive gradients.
- Stage3 unchanged trainer SHA256 `9861eb6b2a5b0bfac546524c50d402aba5827e5079a228e2ae5501909bc937cb`, seed matched per pair, 150 configured max epochs preserving existing `NO_EVAL=True` / train-loss patience behavior, final test required (`SKIP_TEST=False`), beam 20, top K `[5,10]`, batch 4096, inference batch 1024, 4 ranks, port 50201. No Stage3 trainer change.
- Locked paths are the six exact run roots in S01 `protocol_manifest_iter30.md` table: `results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/{iter26_mapping|iter29_mapping}_seed{43|44|45}/` and matching `results/stage3_T5Train/curvature_RQ-VAE_iter30/.../`. Source/config/logs stay in the iter30 source subtree; all products stay under the short roots; no overwriting.
- Rehash all locked inputs and warm-start immediately before every applicable invocation; prove actual consumer/resolved-path use. Warm-start absent/mismatched/malformed/incompatible or zero compatible transferred tensors must fail closed. Every Stage2 run needs its own one-checkpoint/one-batch gradient-path check before training. S00/S01 did not authorize execution.

## Required S02 artifact
Propose exactly one concise, falsifiable hypothesis conforming to skill §6, with (A) research question, (B) exact two-arm equations/constants and one-factor contrast, (C) actual numeric substitution for both arms, with raw-residual status expressly provisional, (D) direct effects/invariants and preregistered measurable counterfactual, (E) downstream rationale clearly labeled inference, (F) falsification and inclusive target classification. Distinguish mapping-effect estimation from target attainment; include `d_s=R29_s-R26_s`, the three-seed mean and descriptive uncertainty reporting, no early stop/omission/pooling. Do not claim S03 approval, execution authorization, or expected threshold success as fact.

## Candidate independence and execution boundary
Agent A and Agent B receive this identical packet and independently write only their candidate artifact under `logs/deliberation/S02_HYPOTHESIS/round_1/`. Do not read or react to the other candidate. Do not edit canonical files, code, configs, or launch jobs. Judge C adjudicates against primary evidence; only Judge-approved `logs/hypothesis_iter30.md` may propagate.