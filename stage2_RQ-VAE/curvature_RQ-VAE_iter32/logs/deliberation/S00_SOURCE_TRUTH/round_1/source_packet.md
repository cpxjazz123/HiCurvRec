# Iter32 S00 Source Packet — HRA Request and Current Workflow State

```text
STAGE_ID=S00_SOURCE_TRUTH
ROUND=1
ITERATION=32
PACKET_STATUS=FROZEN_FOR_INDEPENDENT_REVIEW
ITERATION_PURPOSE=PERFORMANCE_SEEKING_MECHANISM
SWEEP_OR_REPLICATION_ITERATION=NO
ROOT_CAUSE_ITERATION=NO
USER_INPUT_REQUIRED=NO
PARALLEL_GROUP_1=Agent A and Agent B independently assess the same packet and primary repository facts.
PARALLEL_GROUP_2=Judge C adjudicates after both candidate reports are complete.
SERIAL_DEPENDENCIES=Judge waits for both independent reports; no source mutation or GPU execution is authorized by S00.
```

## S00 objective

Independently determine whether the user's proposed Iter32 HRA experiment is admissible to register now, or whether workflow policy requires completing the repairable Iter31 S08 provenance verification first. Establish the authoritative current contract, candidate parent/comparator facts, explicit one-factor boundary, and unresolved prerequisites. This packet authorizes no code edit, MVG, Stage2, Stage3, GPU use, contract change, parent selection, or iteration launch.

## User-proposed direction (proposal, not yet canonical)

The user proposes a new performance-seeking experiment named `Iter32: Hyperbolic Residual Aggregation (HRA)`, with `Iter18` as the parent and the existing Euclidean Step6 sum replaced by a common-reference-curvature HRA operation:

```text
z_E = e_0 + e_1 + e_2
q_l = exp0^{c_l}(e_l)
q_l^0 = exp0^{c_0}(log0^{c_l}(q_l))
h = q_0^0 ⊕_{c_0} (q_1^0 ⊕_{c_0} q_2^0)
z = log0^{c_0}(h)
```

The proposal says all other Iter18 components remain unchanged: layer-wise dynamic curvature, Poincaré quantization, Möbius residuals, cross-curvature transport, Sinkhorn, commitment, behavior loss, optimizer, Stage1, Stage3, seed/dataset/protocol. It frames the sole change as Step6 aggregation and proposes a separate later Iter33 for Structure-Preserving Cross-Curvature Transport. The claimed rationale is that HRA was implemented and numerically active in Iter31 but never received Stage2/Stage3 performance evaluation. The proposal describes Iter18 as the strong baseline and requests formal Iter32 establishment.

## Current controlling research contract

The checked-in skill at `.claude/skills/curvature-rqvae-iter/SKILL.md` declares FCCR-1 active: fixed, closed-form, non-trainable, time-invariant curvature computed from behavior branching and raw residual magnitude. Dynamic/cyclic curvature is forbidden unless changed by an explicit higher-priority instruction or canonically adjudicated between-iteration contract transition. A scientific decision/transition requires parallel A/B and Judge C. Missing MVG batch IDs, hashes, or runtime device are explicitly repairable evidence-capture failures: invalidate the incomplete attempt, repair instrumentation, and rerun S08 in the same iteration; they are not a feasibility abort when faithful rerun is possible.

Repository root `CLAUDE.md` additionally requires fixed output roots under `results/stage2_RQ-VAE/curvature_RQ-VAE_iter<N>/` and `results/stage3_T5Train/curvature_RQ-VAE_iter<N>/`, no CLI overrides, a nonzero-gradient path check before actual Stage2 training, GitHub-only `main`, and push/hash verification for each completed or aborted iteration. Stage2 quality proxies are descriptive only; the adoption target is strict `test_R@10 > 0.065`.

## Iter31 primary facts and current S08 state

- Iter31 has a canonical HRA Step6 contract (`logs/hra_step6_contract_iter31.json`) specifying the common-reference-curvature aggregation above while preserving the inherited FCCR-1 fixed-curvature contract (`logs/mechanism_contract_iter31.json`). Its canonical one-factor diff names Iter29 as parent and Iter29 as sole comparator. Iter31 Stage2/Stage3 have not run.
- Iter31 S08 raw MVG output (`logs/deliberation/S08_MVG/round_2/parent_mvg_results.md`) reports finite, in-domain HRA outputs of shape `[640,32]`, a finite nonzero direct Step6 effect, fixed-curvature invariance, and finite nonzero total/component gradients. It does not contemporaneously print the actual ordered source/future IDs or actual runtime device string.
- The prior round-3 Judge authorized one data-only reconstruction; round-4 Judge then declared `ABORT_ITERATION`, based on the old single-invocation restriction. The abort note records the provenance gap and no Stage2/Stage3 execution.
- The just-pulled deliberation gate now allows operational-repair rounds and no longer caps round numbers. A no-argument run from Iter31 failed earlier at S00 because `forbidden_action_is_positive()` still treats the phrase `without ... reviving matched-seed replication` as positive: it reported the `seed replication` pattern. The current tool output is captured in the conversation; the gate did not yet reach S08.
- This deterministic S00 false positive is checker-repairable. Any gate/checker repair must preserve historical deliberation records and use the current operational-repair process; no historical round may be rewritten to make a check pass.

## Parent and exact performance evidence

- Iter18's exact `test_final.json` reports `n_eval=57439`, `test_recall@10=0.05988962203380978`, R@5 `0.040460314420515675`, NDCG@5 `0.02697174940916111`, and NDCG@10 `0.03323035190128754`. This misses the strict `0.065` target. The Iter31 S01 protocol explicitly classifies Iter18 as `HISTORICAL_NONCOMPARABLE`; no `protocol_manifest_iter18.md` exists at the expected Iter18 path.
- Iter18 source (`curvature_RQ-VAE.py`, `curvature_config.py`, `modules/rqvae.py`) confirms cyclic curvature (`C_CYCLIC_MIN=0.05`, `C_CYCLIC_MAX=1.5`, period `100000`), curvature-conditioned AdamW beta2 (`ADAMW_BETA2_SPAN=0.009`), Poincaré/Sinkhorn quantization, Möbius residual update, cross-layer curvature transport, behavior loss, and Euclidean Step6 sum at `modules/rqvae.py:302-309`.
- Iter29's exact `test_final.json` reports `n_eval=57439`, `test_recall@10=0.05921064085377531`, R@5 `0.03953759640662268`, NDCG@5 `0.026252776900288842`, and NDCG@10 `0.03257647953179143`. Iter29 is Iter31's canonically selected parent/comparator under its existing S01/S05 records and uses FCCR-1 fixed curvature. Both values are below target; matching `n_eval` alone does not prove protocol compatibility.

## Required review questions

1. Does the explicit user proposal supersede FCCR-1 for a new iteration, or does the skill's same-iteration S08 evidence-repair rule require completing Iter31's registered experiment first?
2. If Iter32 can be registered, is Iter18 an admissible parent and protocol-compatible direct baseline, given its missing protocol manifest and `HISTORICAL_NONCOMPARABLE` classification? What must S01 establish before code changes?
3. Does the proposed Iter32 one-factor stack truly change only Step6 relative to Iter18, and can all listed components/protocol identities be held fixed and verified from primary evidence?
4. What is the canonical autonomous next action? No Stage2/Stage3 authorization is in scope for S00.

## Evidence index

- Current skill: `.claude/skills/curvature-rqvae-iter/SKILL.md` §§2, 2.5A, 2.10, 3–5, 17–19.
- Project rules: root `CLAUDE.md` §§0–13, especially Stage2/Stage3 roots, gradient check, target, and Git rules.
- Iter31 HRA/FCCR contracts and selected parent/protocol: `stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/hra_step6_contract_iter31.json`, `logs/mechanism_contract_iter31.json`, `logs/protocol_manifest_iter31.md`, `logs/one_factor_diff_iter31.md`.
- Iter31 S08: `logs/deliberation/S08_MVG/round_2/parent_mvg_results.md`, `round_3/judge.md`, `round_4/judge.md`, `logs/iteration_abort_iter31.md`.
- Iter18 implementation and exact outcome: `stage2_RQ-VAE/curvature_RQ-VAE_iter18/{curvature_RQ-VAE.py,curvature_config.py,modules/rqvae.py,logs/stage3_outcome_iter18.md}` and `results/stage3_T5Train/curvature_RQ-VAE_iter18/logs/Amazon_2023_Instruments/Sep-26-2026_04-45-57/test_final.json`.
- Iter29 exact outcome and closure: `results/stage3_T5Train/curvature_RQ-VAE_iter29/logs/Amazon_2023_Instruments/Sep-27-2026_06-11-34/test_final.json` and `stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/git_closure_iter29.md`.
- Current checker: `.claude/skills/curvature-rqvae-iter/scripts/deliberation_gate.py`.
