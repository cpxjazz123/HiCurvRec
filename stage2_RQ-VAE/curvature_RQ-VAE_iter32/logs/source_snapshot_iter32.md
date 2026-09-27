# Iter32 S00 Source Snapshot — Judge C

```text
STAGE_ID=S00_SOURCE_TRUTH
ROUND=1
VERDICT=MERGE_AB
ITERATION=32
ITER32_PROPOSAL_STATUS=REGISTERED_DIRECTION_DEFERRED_NOT_CANONICALLY_REGISTERED
ITERATION_PURPOSE=PERFORMANCE_SEEKING_MECHANISM
SWEEP_OR_REPLICATION_ITERATION=NO
ROOT_CAUSE_ITERATION=NO
USER_INPUT_REQUIRED=NO
```

## Canonical S00 decision and boundary

The user's Iter32 HRA proposal is recorded in the frozen S00 packet as a **registered direction for review**, but the Iter32 scientific experiment is **deferred and not canonically registered**. S00 does not authorize implementation, an FCCR-1 amendment, MVG, checker/model execution, GPU use, Stage2, or Stage3.

The autonomous next action is the current skill's fast operational repair in **Iter31**, not proceeding with Iter32: repair the documented deliberation-gate false positive and add contemporaneous S08 provenance capture in a single repair batch; document the repair; rerun affected deterministic checks and Iter31 S08 with its registered science/protocol held fixed; then have a repair verifier adjudicate the record. Discard the defective S08 attempt for gate purposes. Do not rewrite historical Iter31 artifacts. This follows current skill §§2.2 and 2.5A. The earlier Iter31 abort was issued under a single-invocation restriction that the current explicit repair-and-rerun rule supersedes; current primary evidence identifies a provenance-capture defect, not mechanism infeasibility.

Only after Iter31 reaches a valid terminal adjudication may a renewed Iter32 proposal proceed through its own canonical between-iteration contract adjudication and protocol lock. No later action is left to user choice.

## Verified facts

### Governing contract and transition scope

- Root `CLAUDE.md` is the highest-priority repository rule. The checked-in curvature skill is controlling research policy. Under current skill §§2.2/2.5A, a repairable missing-provenance failure requires repair and rerun of the same stage in the same iteration; the incomplete attempt is not a gate pass. Before MVG, provenance must include actual ordered batch/sample/item IDs, actual runtime device, required input/checkpoint identity, batch shape/count, mechanism diagnostics, and complete run status.
- FCCR-1 is the current curvature contract: closed-form, non-trainable, time-invariant curvature derived from behavior branching and raw residual magnitude; cyclic/scheduled curvature is forbidden unless changed by a higher-priority instruction or a canonically adjudicated between-iteration transition (skill §§3.1–3.3).
- Iter31's `hra_step6_contract_iter31.json` says its HRA Step6 transition is active for Iter31 only and explicitly preserves the FCCR-1 fixed-curvature subcontract. Its `mechanism_contract_iter31.json` records fixed closed-form curvatures `[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]`. That transition does not activate Iter18-style dynamic curvature for Iter32.

### Iter31 registration and S08

- Iter31 S01 selects Iter29 as parent and sole direct comparator. Iter31 S05 conditionally registers the exact common-reference-curvature HRA Step6 replacement of Euclidean summation as its sole model-behavior change.
- Iter31 S08 raw output records finite `[640,32]` HRA and Euclidean outputs, a finite nonzero direct Step6 effect, valid domains, fixed-curvature invariance, and finite nonzero total/component gradients. It does **not** contemporaneously record actual ordered selected IDs or actual runtime device string. The prior post-hoc reconstruction cannot establish historical input identity or runtime device.
- Iter31's round-4 Judge and abort record ended Iter31 under its then-applied one-invocation restriction; neither records a Stage2 or Stage3 run. The active skill now explicitly provides a repair route for exactly this missing-evidence class. The current deliberation gate supports operational-repair rounds without redundant A/B and has no maximum-round cap. The frozen packet reports that the gate invocation failed at S00 on a negated “seed replication” phrase; source inspection confirms a bounded negation-prefix heuristic. Repair that checker defect without rewriting historical deliberation.
- This disposition does not label the existing S08 as a valid pass and does not infer HRA performance. Rerun S08 only as a same-iteration operational verification repair, capturing the required evidence contemporaneously and preserving Iter31's registered equation, constants, parent, protocol, seed/selection rule, data identity, and evaluation definition.

### Iter18 outcome, source, and comparison limits

- Iter18's exact `test_final.json` records `n_eval=57439`; R@10 `0.05988962203380978`; R@5 `0.040460314420515675`; NDCG@5 `0.02697174940916111`; NDCG@10 `0.03323035190128754`. The recorded R@10 misses the strict `>0.065` target by `0.005110377966190224` (Iter18 outcome record).
- Iter18 source sets cyclic curvature bounds `0.05` to `1.5` with a `100000`-step period and `ADAMW_BETA2_SPAN=0.009`; the Iter18 outcome/bridge identifies P18B as fixed initial-curvature-conditioned AdamW beta2 while retaining cyclic curvature. Its `modules/rqvae.py` Step6 sums quantized layer embeddings in Euclidean coordinates.
- Iter31 S01 classifies Iter18 `HISTORICAL_NONCOMPARABLE`. The expected `stage2_RQ-VAE/curvature_RQ-VAE_iter18/logs/protocol_manifest_iter18.md` is absent. Iter29's exact result is R@10 `0.05921064085377531`, also with `n_eval=57439`; equal evaluation counts do **not** establish protocol compatibility or make Iter18 a valid parent/comparator. Iter18's metric is a historical observation, not causal evidence for HRA.

### Proposed Iter32 direction

The packet proposes replacing Iter18 Euclidean Step6 with common-reference-curvature HRA while keeping Iter18's dynamic/cyclic curvature, optimizer and other listed components. The requested Step6 replacement is a plausible structural intervention **as a proposal**; it is not a verified one-factor experiment, canonical mechanism contract, or protocol lock. Keeping dynamic/cyclic curvature conflicts with current FCCR-1. A user request identifies the direction to consider, but this packet contains no canonical between-iteration contract transition authorizing the proposed contract. No claim that matching `n_eval` cures that conflict or Iter18's noncomparability is accepted.

## Inference and limits

- **Inference:** the recorded Iter31 S08 numbers make a faithful same-science evidence rerun appear operationally plausible, but they establish neither a valid provenance gate nor downstream performance. The current skill requires rerun rather than accepting this inference as a pass.
- **Inference:** HRA may be a forward performance-seeking structural mechanism, but the Iter18 outcome neither validates the proposed cross-curvature operation nor shows its causal effect. Protocol, contract, and one-factor admissibility remain unresolved for any future Iter32 registration.
- The expected Iter18 protocol manifest's absence was checked at its canonical path; no current Iter32 protocol compatibility claim is made.

## Canonical artifact and next action

`logs/deliberation/S00_SOURCE_TRUTH/round_1/judge.md` is the S00 adjudication. The sole next action is the consolidated Iter31 fast operational-repair round described above. After its verifier reaches a valid terminal result, any continued HRA direction must first undergo a fresh canonical between-iteration contract transition and S01 protocol lock; Iter32 remains deferred until then. No S00 authorization for implementation, MVG, Stage2, or Stage3 is granted.

## Primary evidence reviewed

- `.claude/skills/curvature-rqvae-iter/SKILL.md`, §§2.2, 2.5A, 3.1–3.3.
- `.claude/skills/curvature-rqvae-iter/scripts/deliberation_gate.py`, highest-round selection, negation heuristic, and operational-repair routing.
- `stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/protocol_manifest_iter31.md`, `one_factor_diff_iter31.md`, `hra_step6_contract_iter31.json`, `mechanism_contract_iter31.json`.
- `stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/deliberation/S08_MVG/round_2/parent_mvg_results.md`, `round_4/judge.md`, and `logs/iteration_abort_iter31.md`.
- `stage2_RQ-VAE/curvature_RQ-VAE_iter18/curvature_RQ-VAE.py`, `curvature_config.py`, `modules/rqvae.py`, `logs/iteration_bridge.md`, and `logs/stage3_outcome_iter18.md`.
- `results/stage3_T5Train/curvature_RQ-VAE_iter18/logs/Amazon_2023_Instruments/Sep-26-2026_04-45-57/test_final.json`.
- `stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/deliberation/S01_PROTOCOL_LOCK` canonical evidence and `results/stage3_T5Train/curvature_RQ-VAE_iter29/logs/Amazon_2023_Instruments/Sep-27-2026_06-11-34/test_final.json`.
- Frozen packet and completed independent candidates in `logs/deliberation/S00_SOURCE_TRUTH/round_1/`.
